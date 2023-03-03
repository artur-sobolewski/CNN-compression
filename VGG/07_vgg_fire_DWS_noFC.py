import torch

from utils.data_module import DataModule
from utils.model import Model

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# >>>
# Configuration

NAME = "07_VGG_fire_DWS_noFC"
DEVICES = [0]
SEED = 0
BATCH_SIZE = 128
MAX_EPOCHS = -1

class DWSconv(torch.nn.Module):
    def __init__(self, nin, nout, kernel_size=3, padding=1, stride = 1, bias=False, pointwise=False):
        super(DWSconv, self).__init__()

        self.pointwise = pointwise
        
        if self.pointwise:
            self.depthwise = torch.nn.Conv2d(nin, nin, kernel_size=kernel_size, padding=padding, stride=(stride, stride), groups=nin, bias=bias)
            self.pointwise = torch.nn.Conv2d(nin, nout, kernel_size=1, bias=bias)
        else:
            self.depthwise = torch.nn.Conv2d(nin, nout, kernel_size=kernel_size, padding=padding, stride=(stride, stride), groups=nin, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        if self.pointwise:
            out = self.pointwise(out)
        return out

class Fire(torch.nn.Module):

    def __init__(self, nin, squeeze,
                 expand1x1_out, expand3x3_out,
                 stride = 1, bn=False):
        super(Fire, self).__init__()
        self.nin = nin
        self.bn = bn

        if bn:
            self.squeeze_bn = torch.nn.BatchNorm2d(squeeze)
            self.expand1x1_bn = torch.nn.BatchNorm2d(expand1x1_out)
            self.expand3x3_bn = torch.nn.BatchNorm2d(expand3x3_out)

        self.squeeze = torch.nn.Conv2d(nin, squeeze, kernel_size=1)
        self.squeeze_activation = torch.nn.ReLU(inplace=True)
        self.expand1x1 = torch.nn.Conv2d(squeeze, expand1x1_out,
                                   kernel_size=1, stride = stride)
        self.expand3x3 = DWSconv(squeeze, expand3x3_out, stride = stride, pointwise=False)
        self.activation = torch.nn.ReLU(inplace=True)

    def forward(self, x):

        out = self.squeeze(x)
        if self.bn:
            out = self.squeeze_bn(out)
        out = self.squeeze_activation(out)

        ex1 = self.expand1x1(out)
        ex3 = self.expand3x3(out)
        
        if self.bn:
            ex1 = self.expand1x1_bn(ex1)
            ex3 = self.expand3x3_bn(ex3)
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

        layers += [Fire(input_channel, l//squeeze_ratio, expand1x1_out=l//2, expand3x3_out=l//2, bn=batch_norm)]

        input_channel = l

    return torch.nn.Sequential(*layers)
    
model = VGG(make_layers(cfg['E'], squeeze_ratio=8, batch_norm=True), bn=True) #VGG19_bn

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