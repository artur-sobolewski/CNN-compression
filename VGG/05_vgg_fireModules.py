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

NAME = "05_VGG_fireModules"
SEED = int(os.getenv('TRAINING_SEED'))
ONE_CYCLE_LR = (os.getenv('ONE_CYCLE_LR') == 'true')
MAX_EPOCHS = int(os.getenv('MAX_EPOCHS')) if ONE_CYCLE_LR else -1

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
        self.expand1x1_activation = torch.nn.ReLU(inplace=True)
        self.expand3x3 = torch.nn.Conv2d(squeeze, expand3x3_out,
                                   kernel_size=3, padding=1, stride = stride)
        self.expand3x3_activation = torch.nn.ReLU(inplace=True)

    def forward(self, x):

        out = self.squeeze(x)
        ex1 = self.expand1x1(out)
        ex3 = self.expand3x3(out)
        if self.bn:
            out = self.squeeze_bn(out)
            ex1 = self.expand1x1_bn(ex1)
            ex3 = self.expand3x3_bn(ex3)
        out = self.squeeze_activation(out)
        out = torch.cat([
            self.expand1x1_activation(ex1),
            self.expand3x3_activation(ex3)
        ], 1)

        return out

# https://github.com/weiaicunzai/pytorch-cifar100/blob/master/models/vgg.py

cfg = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,         ],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,         ],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,    ],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512]
}

class VGG(torch.nn.Module):
    def __init__(self, features, num_class=100):
        super().__init__()
        self.features = features

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(512, num_class),
            torch.nn.Dropout()
        )

        self.gap = torch.nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        output = self.features(x)
        output = self.gap(output)
        output = torch.flatten(output, 1)
        output = self.classifier(output)

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
    

def get_model(num_classes):
    model = VGG(make_layers(cfg['E'], squeeze_ratio=8, batch_norm=True), num_class=num_classes) #VGG19_bn
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