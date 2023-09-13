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

NAME = "07_invResB_noFC"
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

class invertedResidual(torch.nn.Module):
    def __init__(self, nin, nhidden, nout, stride = 1, downsample=False):
        super(invertedResidual, self).__init__()

        self.downsample = downsample

        self.conv1 = torch.nn.Conv2d(nin, nhidden, kernel_size=1, stride=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(nhidden)
        self.relu1 = torch.nn.ReLU6(inplace=True)

        self.conv2 = DWSconv(nhidden, nhidden, kernel_size = 3, stride=stride, padding = 1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(nhidden)
        self.relu2 = torch.nn.ReLU6(inplace=True)

        self.conv3 = torch.nn.Conv2d(nhidden, nout, kernel_size = 1, stride=1, bias=False)
        self.bn3 = torch.nn.BatchNorm2d(nout)

        if self.downsample:
            self.downsampleLayer = torch.nn.Sequential(
                torch.nn.Conv2d(nin, nout, kernel_size=(1, 1), stride=stride, bias=False),
                torch.nn.BatchNorm2d(nout)
            )


    def forward(self, x):
        skipconnection = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample:
            skipconnection = self.downsampleLayer(skipconnection)

        return out + skipconnection

def get_model(num_classes):
    model = torchvision.models.resnet101(num_classes=num_classes)
    model.conv1 = DWSconv(3, 64, pointwise=True)
    model.maxpool = torch.nn.Identity()

    #layer1
    model.layer1[0] = invertedResidual(64, 64, 10, downsample=True)
    model.layer1[1] = invertedResidual(10, 64, 10)
    model.layer1[2] = invertedResidual(10, 64, 22, stride=2, downsample=True)
    #layer2
    model.layer2[0] = invertedResidual(22, 128, 22)
    model.layer2[1] = invertedResidual(22, 128, 22)
    model.layer2[2] = invertedResidual(22, 128, 22)
    model.layer2[3] = invertedResidual(22, 128, 42, stride=2, downsample=True)
    #layer3
    model.layer3[0] = invertedResidual(42, 256, 42)
    model.layer3[1] = invertedResidual(42, 256, 42)
    model.layer3[2] = invertedResidual(42, 256, 42)
    model.layer3[3] = invertedResidual(42, 256, 42)
    model.layer3[4] = invertedResidual(42, 256, 42)
    model.layer3[5] = invertedResidual(42, 256, 42)
    model.layer3[6] = invertedResidual(42, 256, 42)
    model.layer3[7] = invertedResidual(42, 256, 42)
    model.layer3[8] = invertedResidual(42, 256, 42)
    model.layer3[9] = invertedResidual(42, 256, 42)
    model.layer3[10] = invertedResidual(42, 256, 42)
    model.layer3[11] = invertedResidual(42, 256, 42)
    model.layer3[12] = invertedResidual(42, 256, 42)
    model.layer3[13] = invertedResidual(42, 256, 42)
    model.layer3[14] = invertedResidual(42, 256, 42)
    model.layer3[15] = invertedResidual(42, 256, 42)
    model.layer3[16] = invertedResidual(42, 256, 42)
    model.layer3[17] = invertedResidual(42, 256, 42)
    model.layer3[18] = invertedResidual(42, 256, 42)
    model.layer3[19] = invertedResidual(42, 256, 42)
    model.layer3[20] = invertedResidual(42, 256, 42)
    model.layer3[21] = invertedResidual(42, 256, 42)
    model.layer3[22] = invertedResidual(42, 256, 86, stride=2, downsample=True)
    #layer4
    model.layer4 = torch.nn.Sequential(
            invertedResidual(86, 512, 86),
            invertedResidual(86, 512, 86),
            invertedResidual(86, 512, 512, downsample=True),
            torch.nn.Conv2d(512, num_classes, kernel_size=1, stride=1, bias=False),
            torch.nn.BatchNorm2d(num_classes),
            torch.nn.ReLU6(inplace=True)
    )
    model.fc = torch.nn.Identity()
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