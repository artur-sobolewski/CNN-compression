import os
import torch
from dotenv import load_dotenv

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

NAME = "12_VGG_fire_Gpointwise_shuffle_SE"
SEED = int(os.getenv('TRAINING_SEED'))
ONE_CYCLE_LR = (os.getenv('ONE_CYCLE_LR') == 'true')
MAX_EPOCHS = int(os.getenv('MAX_EPOCHS')) if ONE_CYCLE_LR else -1

class SEConnection(torch.nn.Module):
    def __init__(self, nin, nhidden, nout):
        super(SEConnection, self).__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(nin, nhidden, bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(nhidden, nout, bias=False),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)

        y = y.expand_as(x)
        out = x * y

        return out

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
    def __init__(self, nin, nout, stride=1, bias=False, groups=4, shuffle=True, isrelu=False, bn=False):
        super().__init__()

        self.shuffle = shuffle
        self.isrelu = isrelu
        self.bn = bn

        if nin > 24:
            self.groups = groups
        else:
            self.groups = 1
            self.shuffle = False

        self.conv1 = torch.nn.Conv2d(nin, nout, kernel_size=1, stride=stride, groups=self.groups, bias=bias)
        if self.bn:
            self.bn1 = torch.nn.BatchNorm2d(nout)
        if self.isrelu:
            self.relu1 = torch.nn.ReLU(inplace=True)
        if self.shuffle:
            self.shuffle = ChannelShuffle(nout, groups)
        

    def forward(self, x):
        out = self.conv1(x)
        if self.bn:
            out = self.bn1(out)
        if self.isrelu:
            out = self.relu1(out)
        if self.shuffle:
            out = self.shuffle(out)
        return out

class DWSconv(torch.nn.Module):
    def __init__(self, nin, nout, kernel_size=3, padding=1, stride = 1, bias=False, pointwise=False, bn=False):
        super(DWSconv, self).__init__()

        self.pointwise = pointwise
        self.bn = bn
        
        if self.pointwise:
            self.depthwise = torch.nn.Conv2d(nin, nin, kernel_size=kernel_size, padding=padding, stride=(stride, stride), groups=nin, bias=bias)
            self.pointwise = torch.nn.Conv2d(nin, nout, kernel_size=1, bias=bias)
        else:
            self.depthwise = torch.nn.Conv2d(nin, nout, kernel_size=kernel_size, padding=padding, stride=(stride, stride), groups=nin, bias=bias)
        
        if bn:
            self.bn1 = torch.nn.BatchNorm2d(nout)

    def forward(self, x):
        out = self.depthwise(x)
        if self.pointwise:
            out = self.pointwise(out)
        if self.bn:
            self.bn1(out)
        return out


class Fire(torch.nn.Module):

    def __init__(self, nin, squeeze,
                 expand1x1_out, expand3x3_out,
                 stride = 1, bn=False):
        super(Fire, self).__init__()
        self.nin = nin

        self.squeeze = groupPointwiseConv(nin, squeeze, groups=4, isrelu=True, bn=bn)
        self.expand1x1 = groupPointwiseConv(squeeze, expand1x1_out, stride=stride, shuffle=False, isrelu=False, bn=bn)
        self.expand3x3 = DWSconv(squeeze, expand3x3_out, stride=stride, bn=bn)
        

        self.shuffle = ChannelShuffle(expand1x1_out + expand3x3_out, groupsNum=8)
        self.activation = torch.nn.ReLU(inplace=True)

    def forward(self, x):

        out = self.squeeze(x)

        ex1 = self.expand1x1(out)
        ex3 = self.expand3x3(out)
        
        out = torch.cat([
            ex1,
            ex3
        ], 1)
        out = self.shuffle(out)
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
    for l in cfg:
        if l == 'M':
            layers += [torch.nn.MaxPool2d(kernel_size=2, stride=2)]
            continue

        layers += [Fire(input_channel, l//squeeze_ratio, expand1x1_out=l//2, expand3x3_out=l//2, bn=batch_norm)]
        layers += [SEConnection(l, int(l*0.25), l)]

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