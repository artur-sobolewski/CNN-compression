import os
import torch
import torchvision
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

NAME = "10_invResB_ghostBlocks"
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
    
class ghostModule(torch.nn.Module):
    def __init__(self, nin, nout, stride = 1, padding = 1, relu=True):
        super(ghostModule, self).__init__()

        self.nin = nin
        self.nghost = nout//2
        self.nout = nout
        
        self.conv1 = torch.nn.Conv2d(nin, self.nghost, kernel_size=1, stride=1, bias=False)
        self.conv2 = torch.nn.Conv2d(self.nghost, self.nghost, kernel_size=3, padding=padding, stride=(stride, stride) , groups=self.nghost, bias=False)
        self.bn = torch.nn.BatchNorm2d(nout)
        if (relu):
            self.relu = torch.nn.ReLU6(inplace=True)
        else:
            self.relu = None

        

    def forward(self, x):

        out = self.conv1(x)
        ghost = self.conv2(out)

        out = torch.cat((ghost, out), axis=1)

        out = self.bn(out)
        if (self.relu):
            out = self.relu(out)

        return out


class ghostBlock(torch.nn.Module):
    def __init__(self, nin, nexpand, nout, stride = 1, padding = 1, downsample=False):
        super(ghostBlock, self).__init__()

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

        self.ghost1 = ghostModule(nin, nexpand)
        self.ghost2 = ghostModule(nexpand, nout, relu=False)

    def forward(self, x):
        skipConnection = x

        out = self.ghost1(x)
        if self.downsample:
            out = self.downsampleLayer(out)

        out = self.ghost2(out)

        if self.downsample:
            skipConnection = self.downsampleSkip(skipConnection)

        out += skipConnection

        return out

def get_model(num_classes):
    model = torchvision.models.resnet101(num_classes=num_classes)
    model.conv1 = DWSconv(3, 64, pointwise=True)
    model.maxpool = torch.nn.Identity()

    #layer1
    model.layer1[0] = ghostBlock(64, 64, 10, downsample=True)
    model.layer1[1] = ghostBlock(10, 64, 10)
    model.layer1[2] = ghostBlock(10, 64, 10)
    #layer2
    model.layer2[0] = ghostBlock(10, 64, 22, stride=2, downsample=True)
    model.layer2[1] = ghostBlock(22, 128, 22)
    model.layer2[2] = ghostBlock(22, 128, 22)
    model.layer2[3] = ghostBlock(22, 128, 22)
    #layer3
    model.layer3[0] = ghostBlock(22, 128, 42, stride=2, downsample=True)
    model.layer3[1] = ghostBlock(42, 256, 42)
    model.layer3[2] = ghostBlock(42, 256, 42)
    model.layer3[3] = ghostBlock(42, 256, 42)
    model.layer3[4] = ghostBlock(42, 256, 42)
    model.layer3[5] = ghostBlock(42, 256, 42)
    model.layer3[6] = ghostBlock(42, 256, 42)
    model.layer3[7] = ghostBlock(42, 256, 42)
    model.layer3[8] = ghostBlock(42, 256, 42)
    model.layer3[9] = ghostBlock(42, 256, 42)
    model.layer3[10] = ghostBlock(42, 256, 42)
    model.layer3[11] = ghostBlock(42, 256, 42)
    model.layer3[12] = ghostBlock(42, 256, 42)
    model.layer3[13] = ghostBlock(42, 256, 42)
    model.layer3[14] = ghostBlock(42, 256, 42)
    model.layer3[15] = ghostBlock(42, 256, 42)
    model.layer3[16] = ghostBlock(42, 256, 42)
    model.layer3[17] = ghostBlock(42, 256, 42)
    model.layer3[18] = ghostBlock(42, 256, 42)
    model.layer3[19] = ghostBlock(42, 256, 42)
    model.layer3[20] = ghostBlock(42, 256, 42)
    model.layer3[21] = ghostBlock(42, 256, 42)
    model.layer3[22] = ghostBlock(42, 256, 42)
    #layer4
    model.layer4 = torch.nn.Sequential(
        ghostBlock(42, 512, 86, stride=2, downsample=True),
        ghostBlock(86, 512, 86),
        ghostBlock(86, 512, 86)
    )

    model.fc = torch.nn.Linear(86, num_classes, bias=False)
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