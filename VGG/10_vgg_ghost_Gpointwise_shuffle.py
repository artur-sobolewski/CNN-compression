import torch

from utils.data_module import DataModule
from utils.model import Model

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# >>>
# Configuration

NAME = "10_VGG_ghost_Gpointwise_shuffle"
DEVICES = [0]
SEED = 0
BATCH_SIZE = 128
MAX_EPOCHS = -1



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

class ghostModule(torch.nn.Module):
    def __init__(self, nin, nout, stride = 1, padding = 1, relu=True, bn=True, shuffle=True, groups=4):
        super(ghostModule, self).__init__()

        self.nin = nin
        self.nghost = nout//2
        self.nout = nout
        self.shuffle = shuffle

        if shuffle:
            self.groups = groups
        else:
            self.groups = 1
            self.shuffle = False
        
        self.conv1 = groupPointwiseConv(nin, self.nghost, stride=stride, shuffle=self.shuffle, groups=self.groups, isrelu=False)
        self.conv2 = torch.nn.Conv2d(self.nghost, self.nghost, kernel_size=3, padding=padding, stride=(stride, stride) , groups=self.nghost, bias=False)
        if bn:
            self.bn = torch.nn.BatchNorm2d(nout)
        else:
            self.bn = None
        if relu:
            self.relu = torch.nn.ReLU6(inplace=True)
        else:
            self.relu = None

        
    def forward(self, x):

        out = self.conv1(x)
        ghost = self.conv2(out)

        out = torch.cat((ghost, out), axis=1)

        if self.bn:
            out = self.bn(out)

        if self.relu:
            out = self.relu(out)

        return out
    
class Fire(torch.nn.Module):

    def __init__(self, nin, squeeze,
                 expand1x1_out, expand3x3_out,
                 stride = 1, bn=False):
        super(Fire, self).__init__()
        self.nin = nin

        self.squeeze = ghostModule(nin, squeeze, shuffle=False, bn=bn)
        self.expand1x1 = pointwiseConv(squeeze, expand1x1_out, stride=stride, isrelu=False, isbn=bn)
        self.expand3x3 = ghostModule(squeeze, expand3x3_out, stride=stride, shuffle=True, groups=4, relu=False, bn=bn)
        
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
    for l in cfg:
        if l == 'M':
            layers += [torch.nn.MaxPool2d(kernel_size=2, stride=2)]
            continue

        layers += [Fire(input_channel, l//squeeze_ratio, l//2, l//2, bn=batch_norm)]

        input_channel = l

    return torch.nn.Sequential(*layers)
    
model = VGG(make_layers(cfg['E'], squeeze_ratio=8, batch_norm=True)) #VGG19_bn

# ^^^

def main(model):      
    seed_everything(SEED, workers=True)

    logger = TensorBoardLogger(save_dir="./logs/", name=NAME, default_hp_metric=False)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    early_stopping = EarlyStopping(monitor="Loss/train", mode="min", patience=10)

    data = DataModule(batch_size=BATCH_SIZE)
    lightning_model = Model(NAME, model)

    trainer = Trainer(max_epochs=MAX_EPOCHS, accelerator="gpu", devices=DEVICES, logger=logger, callbacks=[lr_monitor, early_stopping], deterministic=True)
    trainer.fit(lightning_model, datamodule=data)
    trainer.save_checkpoint(lightning_model.saved_model_path, weights_only=True)
    trainer.test(lightning_model, data.val_dataloader())
    
if __name__ == '__main__':
    main(model)    