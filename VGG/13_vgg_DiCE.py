import os
import torch
import math
from dotenv import load_dotenv

import torch.nn.functional as F
from utils.data_module import DataModule
from utils.model import Model

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

# >>>
# Configuration

load_dotenv()

NAME = "13_VGG_DiCE"
SEED = int(os.getenv('TRAINING_SEED'))
ONE_CYCLE_LR = (os.getenv('ONE_CYCLE_LR') == 'true')
MAX_EPOCHS = int(os.getenv('MAX_EPOCHS')) if ONE_CYCLE_LR else -1

def channel_shuffle(x, groups):

    batch, channels, height, width = x.size()

    channels_per_group = channels // groups
    x = x.view(batch, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batch, channels, height, width)
    return x

class ChannelShuffle(torch.nn.Module):
    def __init__(self, channelsNum, groupsNum):
        super(ChannelShuffle, self).__init__()
        if channelsNum % groupsNum != 0:
            raise ValueError('channels must be divisible by groups')
        self.groups = groupsNum

    def forward(self, x):
        return channel_shuffle(x, self.groups)

class groupPointwiseConv(torch.nn.Module):
    def __init__(self, nin, nout, stride=1, bias=False, groups=4, isbn=False, isrelu=True, shuffle=True):
        super().__init__()

        self.shuffle = shuffle
        self.isrelu = isrelu
        self.isbn = isbn

        if nin > 24:
            self.groups = groups
        else:
            self.groups = 1
            self.shuffle = False
    

        self.conv1 = torch.nn.Conv2d(nin, nout, kernel_size=1, stride=stride, groups=self.groups, bias=bias)
        if self.isbn:
            self.bn1 = torch.nn.BatchNorm2d(nout)
        if self.isrelu:
            self.relu1 = torch.nn.ReLU(inplace=True)
        if self.shuffle:
            self.shuffle = ChannelShuffle(nout, groups)

    def forward(self, x):
        out = self.conv1(x)
        if self.isbn:
            out = self.bn1(out)
        if self.isrelu:
            out = self.relu1(out)
        if self.shuffle:
            out = self.shuffle(out)
        return out

class pointwiseConv(torch.nn.Module):
    def __init__(self, nin, nout, stride=1, bias=False, isbn=False, isrelu=False):
        super().__init__()

        self.isrelu = isrelu
        self.isbn = isbn

        self.conv1 = torch.nn.Conv2d(nin, nout, kernel_size=1, stride=stride, bias=bias)
        if self.isbn:
            self.bn1 = torch.nn.BatchNorm2d(nout)
        
        if self.isrelu:
            self.relu1 = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)

        if self.isbn:
            out = self.bn1(out)
        
        if self.isrelu:
            out = self.relu1(out)
        return out

# https://github.com/sacmehta/EdgeNets/tree/master

class Shuffle(torch.nn.Module):
    '''
    This class implements Channel Shuffling
    '''
    def __init__(self, groups):
        '''
        :param groups: # of groups for shuffling
        '''
        super().__init__()
        self.groups = groups

    def forward(self, x):
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // self.groups
        x = x.view(batchsize, self.groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)
        return x
    
class DICE(torch.nn.Module):
    '''
    This class implements the volume-wise seperable convolutions
    '''
    def __init__(self, channel_in, channel_out, height, width, kernel_size=3, dilation=[1, 1, 1], shuffle=True, isbn=False):
        '''
        :param channel_in: # of input channels
        :param channel_out: # of output channels
        :param height: Height of the input volume
        :param width: Width of the input volume
        :param kernel_size: Kernel size. We use the same kernel size of 3 for each dimension. Larger kernel size would increase the FLOPs and Parameters
        :param dilation: It's a list with 3 elements, each element corresponding to a dilation rate for each dimension.
        :param shuffle: Shuffle the feature maps in the volume-wise separable convolutions
        '''
        super().__init__()
        assert len(dilation) == 3
        padding_1 = int((kernel_size - 1) / 2) *dilation[0]
        padding_2 = int((kernel_size - 1) / 2) *dilation[1]
        padding_3 = int((kernel_size - 1) / 2) *dilation[2]
        self.conv_channel = torch.nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=1, groups=channel_in,
                                      padding=padding_1, bias=False, dilation=dilation[0])
        self.conv_width = torch.nn.Conv2d(width, width, kernel_size=kernel_size, stride=1, groups=width,
                               padding=padding_2, bias=False, dilation=dilation[1])
        self.conv_height = torch.nn.Conv2d(height, height, kernel_size=kernel_size, stride=1, groups=height,
                               padding=padding_3, bias=False, dilation=dilation[2])
        
        

        self.br_act = torch.nn.Sequential(
            torch.nn.BatchNorm2d(3*channel_in),
            torch.nn.ReLU(inplace=True),
        ) if isbn else torch.nn.Sequential(
            torch.nn.ReLU(inplace=True),
        )
        self.weight_avg_layer = torch.nn.Sequential(
            torch.nn.Conv2d(3*channel_in, channel_in, kernel_size=1, stride=1, padding=1, bias=False, groups=channel_in),
            torch.nn.BatchNorm2d(channel_in),
            torch.nn.ReLU(inplace=True),
        ) if isbn else torch.nn.Sequential(
            torch.nn.Conv2d(3*channel_in, channel_in, kernel_size=1, stride=1, padding=1, bias=False, groups=channel_in),
            torch.nn.ReLU(inplace=True),
        )


        # project from channel_in to Channel_out
        groups_proj = math.gcd(channel_in, channel_out)
        self.proj_layer = torch.nn.Sequential(
            torch.nn.Conv2d(channel_in, channel_out, kernel_size=3, stride=1, padding=0, bias=False, groups=groups_proj),
            torch.nn.BatchNorm2d(channel_out),
            torch.nn.ReLU(inplace=True),
        ) if isbn else torch.nn.Sequential(
            torch.nn.Conv2d(channel_in, channel_out, kernel_size=3, stride=1, padding=0, bias=False, groups=groups_proj),
            torch.nn.ReLU(inplace=True),
        )
        self.linear_comb_layer = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(output_size=1),
            torch.nn.Conv2d(channel_in, channel_in // 4, kernel_size=1, bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(channel_in //4, channel_out, kernel_size=1, bias=False),
            torch.nn.Sigmoid()
        )

        self.vol_shuffle = Shuffle(3)

        self.width = width
        self.height = height
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.shuffle = shuffle
        self.ksize=kernel_size
        self.dilation = dilation

    def forward(self, x):
        '''
        :param x: input of dimension C x H x W
        :return: output of dimension C1 x H x W
        '''
        bsz, channels, height, width = x.size()
        # process across channel. Input: C x H x W, Output: C x H x W
        out_ch_wise = self.conv_channel(x)

        # process across height. Input: H x C x W, Output: C x H x W
        x_h_wise = x.clone()
        if height != self.height:
            if height < self.height:
                x_h_wise = F.interpolate(x_h_wise, mode='bilinear', size=(self.height, width), align_corners=True)
            else:
                x_h_wise = F.adaptive_avg_pool2d(x_h_wise, output_size=(self.height, width))

        x_h_wise = x_h_wise.transpose(1, 2).contiguous()
        out_h_wise = self.conv_height(x_h_wise).transpose(1, 2).contiguous()

        h_wise_height = out_h_wise.size(2)
        if height != h_wise_height:
            if h_wise_height < height:
                out_h_wise = F.interpolate(out_h_wise, mode='bilinear', size=(height, width), align_corners=True)
            else:
                out_h_wise = F.adaptive_avg_pool2d(out_h_wise, output_size=(height, width))

        # process across width: Input: W x H x C, Output: C x H x W
        x_w_wise = x.clone()
        if width != self.width:
            if width < self.width:
                x_w_wise = F.interpolate(x_w_wise, mode='bilinear', size=(height, self.width), align_corners=True)
            else:
                x_w_wise = F.adaptive_avg_pool2d(x_w_wise, output_size=(height, self.width))

        x_w_wise = x_w_wise.transpose(1, 3).contiguous()
        out_w_wise = self.conv_width(x_w_wise).transpose(1, 3).contiguous()
        w_wise_width = out_w_wise.size(3)
        if width != w_wise_width:
            if w_wise_width < width:
                out_w_wise = F.interpolate(out_w_wise, mode='bilinear', size=(height, width), align_corners=True)
            else:
                out_w_wise = F.adaptive_avg_pool2d(out_w_wise, output_size=(height, width))

        # Merge. Output will be 3C x H X W
        outputs = torch.cat((out_ch_wise, out_h_wise, out_w_wise), 1)
        outputs = self.br_act(outputs)
        if self.shuffle:
            outputs = self.vol_shuffle(outputs)
        outputs = self.weight_avg_layer(outputs)
        linear_wts = self.linear_comb_layer(outputs)
        outputs = self.proj_layer(outputs)
        return outputs * linear_wts

    def __repr__(self):
        s = '{name}(in_channels={channel_in}, out_channels={channel_out}, kernel_size={ksize}, vol_shuffle={shuffle}, ' \
            'width={width}, height={height}, dilation={dilation})'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class Fire(torch.nn.Module):

    def __init__(self,
                 nin,
                 squeeze,
                 expand1x1_out,
                 expand3x3_out,
                 height,
                 width,
                 stride = 1,
                 bn=False):
        super(Fire, self).__init__()
        self.nin = nin

        squeeze_groups = math.gcd(nin, squeeze)
        self.squeeze = groupPointwiseConv(nin, squeeze, stride=1, groups=squeeze_groups, isbn=bn, isrelu=True)
        self.expand1x1 = pointwiseConv(squeeze, expand1x1_out, stride=stride, isrelu=False, isbn=bn)
        self.expand3x3 = DICE(squeeze, expand3x3_out, height, width, isbn=bn)

        self.activation = torch.nn.ReLU(inplace=True)
        

    def forward(self, x):

        out = self.squeeze(x)

        ex1 = self.expand1x1(out)
        ex3 = self.expand3x3(out)

        out = torch.cat([
            ex1,
            ex3
        ], 1)
        
        out = self.activation(out)

        return out

# https://github.com/weiaicunzai/pytorch-cifar100/blob/master/models/vgg.py

cfg = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,         ],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,         ],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,    ],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512]
}

class VGG(torch.nn.Module):
    def __init__(self, features, num_class=100, bn=False):
        super().__init__()
        self.features = features

        if bn:
            self.reduce = torch.nn.Sequential(
                torch.nn.Conv2d(512, num_class, kernel_size=1),
                torch.nn.BatchNorm2d(num_class),
                torch.nn.ReLU(inplace=True)
            )
        else:
            self.reduce = torch.nn.Sequential(
                torch.nn.Conv2d(512, num_class, kernel_size=1),
                torch.nn.ReLU(inplace=True)
            )

        self.gap = torch.nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        output = self.features(x)
        output = self.reduce(output)
        output = self.gap(output)
        output = torch.flatten(output, 1)

        return output

def make_layers(cfg, squeeze_ratio, batch_norm=False):
    layers = []

    input_channel = 3
    width = 32
    height = 32
    for l in cfg:
        if l == 'M':
            layers += [torch.nn.MaxPool2d(kernel_size=2, stride=2)]
            width = int(width/2)
            height = int(height/2)
            continue

        layers += [Fire(input_channel, l//squeeze_ratio, l//2, l//2, height, width, bn=batch_norm)]

        input_channel = l

    return torch.nn.Sequential(*layers)

def get_model(num_classes):
    model = VGG(make_layers(cfg['E'], squeeze_ratio=8, batch_norm=True), num_class=num_classes, bn=True) #VGG19_bn
    return model

# ^^^

def main(model, dataset_name, device_num):    
    if dataset_name == "cifar10":
        batch_size = int(os.getenv('CIFAR10_BATCH_SIZE'))
    elif dataset_name == "cifar100":
        batch_size = int(os.getenv('CIFAR100_BATCH_SIZE'))
    name = NAME + "_{}".format(dataset_name)

    devices = [device_num]

    seed_everything(SEED, workers=True)

    logs_path = os.path.join(os.getcwd(), "logs")
    logger = TensorBoardLogger(save_dir=logs_path, name=name, default_hp_metric=False)
    lr_monitor = LearningRateMonitor(logging_interval='step')

    data = DataModule(batch_size, dataset_name)
    lightning_model = Model(name, model)

    if ONE_CYCLE_LR:
        trainer = Trainer(max_epochs=MAX_EPOCHS, accelerator="gpu", devices=devices, logger=logger, callbacks=[lr_monitor], deterministic=True)
    else:
        early_stopping = EarlyStopping(monitor="Loss/train", mode="min", patience=10)
        trainer = Trainer(max_epochs=MAX_EPOCHS, accelerator="gpu", devices=devices, logger=logger, callbacks=[lr_monitor, early_stopping], deterministic=True)
    
    trainer.fit(lightning_model, datamodule=data)
    trainer.save_checkpoint(lightning_model.saved_model_path, weights_only=True)
    trainer.test(lightning_model, data.val_dataloader())


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--dataset', required=True, choices=["cifar10", "cifar100"])
    parser.add_argument('-g', '--gpu_num', required=True)

    args = vars(parser.parse_args())

    dataset_name = args['dataset']
    if dataset_name == "cifar10":
        num_classes = 10
    elif dataset_name == "cifar100":
        num_classes = 100
    
    model = get_model(num_classes)

    main(model, dataset_name, int(args['gpu_num']))