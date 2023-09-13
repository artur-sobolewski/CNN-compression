import os
import torch
import torchvision
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

NAME = "15_invResB_dice"
SEED = int(os.getenv('TRAINING_SEED'))
ONE_CYCLE_LR = (os.getenv('ONE_CYCLE_LR') == 'true')
MAX_EPOCHS = int(os.getenv('MAX_EPOCHS')) if ONE_CYCLE_LR else -1



class DWSconv(torch.nn.Module):
    def __init__(self, nin, nout, kernel_size=3, padding=1, stride = 1, bias=False, pointwise=False):
        super(DWSconv, self).__init__()

        self.pointwise = pointwise

        self.depthwise = torch.nn.Conv2d(nin, nin, kernel_size=kernel_size, padding=padding, stride=(stride, stride), groups=nin, bias=bias)
        
        if self.pointwise:
            self.pointwise = torch.nn.Conv2d(nin, nout, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        if self.pointwise:
            out = self.pointwise(out)
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
    def __init__(self, channel_in, channel_out, height, width, kernel_size=3, dilation=[1, 1, 1], shuffle=True):
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
        )
        self.weight_avg_layer = torch.nn.Sequential(
            torch.nn.Conv2d(3*channel_in, channel_in, kernel_size=1, stride=1, padding=1, bias=False, groups=channel_in),
            torch.nn.BatchNorm2d(channel_in),
            torch.nn.ReLU(inplace=True),
        )


        # project from channel_in to Channel_out
        groups_proj = math.gcd(channel_in, channel_out)
        self.proj_layer = torch.nn.Sequential(
            torch.nn.Conv2d(channel_in, channel_out, kernel_size=3, stride=1, padding=0, bias=False, groups=groups_proj),
            torch.nn.BatchNorm2d(channel_out),
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


class StridedDICE(torch.nn.Module):
    '''
    This class implements the strided volume-wise seperable convolutions
    '''
    def __init__(self, channel_in, channel_out, height, width, kernel_size=3, dilation=[1,1,1], shuffle=True):
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

        chan_out = round(channel_out / 2)
        self.left_layer = torch.nn.Sequential(
            # depthwise
            torch.nn.Conv2d(channel_in, channel_in, kernel_size=3, stride=2, padding=1, bias=False, groups=channel_in),
            torch.nn.BatchNorm2d(channel_in),
            torch.nn.ReLU(inplace=True),
            # pointhwise
            torch.nn.Conv2d(channel_in, chan_out, kernel_size=1, stride=1, padding=0, bias=False, groups=1),
            torch.nn.BatchNorm2d(chan_out),
            torch.nn.ReLU(inplace=True),
        )
        self.right_layer =  torch.nn.Sequential(
            torch.nn.AvgPool2d(kernel_size=3, padding=1, stride=2),
            DICE(channel_in, chan_out, height, width, kernel_size=kernel_size, dilation=dilation,
                 shuffle=shuffle),
        )
        self.shuffle = Shuffle(groups=2)

        self.width = width
        self.height = height
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.ksize = kernel_size

    def forward(self, x):
        x_left = self.left_layer(x)
        x_right = self.right_layer(x)
        concat = torch.cat([x_left, x_right], 1)
        return self.shuffle(concat)

    def __repr__(self):
        s = '{name}(in_channels={channel_in}, out_channels={channel_out}, kernel_size={ksize}, ' \
            'width={width}, height={height})'
        return s.format(name=self.__class__.__name__, **self.__dict__)
    


class diceBlock(torch.nn.Module):
    def __init__(self, nin, nexpand, nout, height, width, stride = 1, padding = 1, downsample=False):
        super(diceBlock, self).__init__()

        self.downsample = downsample
        self.nin = nin
        self.nexpand = nexpand
        self.nout = nout

        if self.downsample:
            self.downsampleSkip = torch.nn.Sequential(
                torch.nn.Conv2d(nin, nout, kernel_size=(1, 1), stride=stride, bias=False),
                torch.nn.BatchNorm2d(nout)
            )
            self.downsampleLayer = torch.nn.Sequential(
                torch.nn.Conv2d(nexpand, nexpand, kernel_size=3, padding=padding, stride=(stride, stride) , groups=nexpand, bias=False),
                torch.nn.BatchNorm2d(nexpand)
            )
        else:
            self.downsampleSkip = None
            self.downsampleLayer = None

        self.conv1 = torch.nn.Conv2d(nin, nexpand, kernel_size=1, stride=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(nexpand)
        self.relu1 = torch.nn.ReLU(inplace=True)

        if self.downsample and stride != 1:
            self.dice = StridedDICE(nexpand, nout, height, width)
        else:
            self.dice = DICE(nexpand, nout, height, width)
            



    def forward(self, x):
        skipConnection = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        # if self.downsample:
        #     out = self.downsampleLayer(out)

        out = self.dice(out)

        if self.downsample:
            skipConnection = self.downsampleSkip(skipConnection)

        out += skipConnection

        return out

def get_model(num_classes):
    model = torchvision.models.resnet101(num_classes=num_classes)
    model.conv1 = DWSconv(3, 64, pointwise=True)
    model.maxpool = torch.nn.Identity()

    #layer1
    height = 32
    width = 32

    model.layer1[0] = diceBlock(64, 64, 10, height, width, downsample=True)
    model.layer1[1] = diceBlock(10, 64, 10, height, width)
    model.layer1[2] = diceBlock(10, 64, 10, height, width)

    #layer2
    height = int(height / 2)
    width = int(width / 2)

    model.layer2[0] = diceBlock(10, 64, 22, height, width, stride=2, downsample=True)
    model.layer2[1] = diceBlock(22, 128, 22, height, width)
    model.layer2[2] = diceBlock(22, 128, 22, height, width)
    model.layer2[3] = diceBlock(22, 128, 22, height, width)

    #layer3
    height = int(height / 2)
    width = int(width / 2)

    model.layer3[0] = diceBlock(22, 128, 42, height, width, stride=2, downsample=True)
    model.layer3[1] = diceBlock(42, 256, 42, height, width)
    model.layer3[2] = diceBlock(42, 256, 42, height, width)
    model.layer3[3] = diceBlock(42, 256, 42, height, width)
    model.layer3[4] = diceBlock(42, 256, 42, height, width)
    model.layer3[5] = diceBlock(42, 256, 42, height, width)
    model.layer3[6] = diceBlock(42, 256, 42, height, width)
    model.layer3[7] = diceBlock(42, 256, 42, height, width)
    model.layer3[8] = diceBlock(42, 256, 42, height, width)
    model.layer3[9] = diceBlock(42, 256, 42, height, width)
    model.layer3[10] = diceBlock(42, 256, 42, height, width)
    model.layer3[11] = diceBlock(42, 256, 42, height, width)
    model.layer3[12] = diceBlock(42, 256, 42, height, width)
    model.layer3[13] = diceBlock(42, 256, 42, height, width)
    model.layer3[14] = diceBlock(42, 256, 42, height, width)
    model.layer3[15] = diceBlock(42, 256, 42, height, width)
    model.layer3[16] = diceBlock(42, 256, 42, height, width)
    model.layer3[17] = diceBlock(42, 256, 42, height, width)
    model.layer3[18] = diceBlock(42, 256, 42, height, width)
    model.layer3[19] = diceBlock(42, 256, 42, height, width)
    model.layer3[20] = diceBlock(42, 256, 42, height, width)
    model.layer3[21] = diceBlock(42, 256, 42, height, width)
    model.layer3[22] = diceBlock(42, 256, 42, height, width)

    #layer4
    height = int(height / 2)
    width = int(width / 2)

    model.layer4 = torch.nn.Sequential(
        diceBlock(42, 512, 86, height, width, stride=2, downsample=True),
        diceBlock(86, 512, 86, height, width),
        diceBlock(86, 512, 86, height, width)
    )

    model.fc = torch.nn.Linear(86, num_classes, bias=False)

    # summary(model.cpu(), (3, 32, 32), device="cpu")
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