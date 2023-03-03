import torch

from utils.data_module import DataModule
from utils.model import Model

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# >>>
# Configuration

NAME = "09_VGG_ghostModule"
DEVICES = [0]
SEED = 0
BATCH_SIZE = 128
MAX_EPOCHS = -1

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
    def __init__(self, nin, nout, stride = 1, padding = 1, relu=True, bn=True):
        super(ghostModule, self).__init__()

        self.nin = nin
        self.nghost = nout//2
        self.nout = nout
        self.isbn = bn
        self.isrelu = relu
        
        self.conv1 = torch.nn.Conv2d(nin, self.nghost, kernel_size=1, stride=1, bias=False)
        self.conv2 = torch.nn.Conv2d(self.nghost, self.nghost, kernel_size=3, padding=padding, stride=(stride, stride) , groups=self.nghost, bias=False)
        if self.isbn:
            self.bn1 = torch.nn.BatchNorm2d(nout)
        if self.isrelu:
            self.relu = torch.nn.ReLU6(inplace=True)

        

    def forward(self, x):

        out = self.conv1(x)
        ghost = self.conv2(out)

        out = torch.cat((ghost, out), axis=1)

        if self.isbn:
            out = self.bn1(out)

        if self.isrelu:
            out = self.relu(out)

        return out

class Fire(torch.nn.Module):

    def __init__(self, nin, squeeze,
                 expand1x1_out, expand3x3_out,
                 stride = 1, bn=False):
        super(Fire, self).__init__()
        self.nin = nin

        self.squeeze = ghostModule(nin, squeeze, bn=bn)
        self.expand1x1 = pointwiseConv(squeeze, expand1x1_out, stride=stride, isrelu=False, isbn=bn)
        self.expand3x3 = ghostModule(squeeze, expand3x3_out, stride=stride, relu=False, bn=bn)

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