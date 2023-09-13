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

NAME = "02_firemodules"
SEED = int(os.getenv('TRAINING_SEED'))
ONE_CYCLE_LR = (os.getenv('ONE_CYCLE_LR') == 'true')
ONE_CYCLE_LR = (os.getenv('ONE_CYCLE_LR') == 'true')
MAX_EPOCHS = int(os.getenv('MAX_EPOCHS')) if ONE_CYCLE_LR else -1 if ONE_CYCLE_LR else -1

class Fire(torch.nn.Module):

    def __init__(self, nin, squeeze,
                 expand1x1_out, expand3x3_out,
                 stride = 1, downsample=False):
        super(Fire, self).__init__()
        self.nin = nin
        self.downsample = downsample
        self.squeeze = torch.nn.Conv2d(nin, squeeze, kernel_size=1)
        self.squeeze_activation = torch.nn.ReLU(inplace=True)
        self.expand1x1 = torch.nn.Conv2d(squeeze, expand1x1_out,
                                   kernel_size=1, stride = stride)
        self.expand1x1_activation = torch.nn.ReLU(inplace=True)
        self.expand3x3 = torch.nn.Conv2d(squeeze, expand3x3_out,
                                   kernel_size=3, padding=1, stride = stride)
        self.expand3x3_activation = torch.nn.ReLU(inplace=True)

        if downsample:
            if stride > 1:
                self.downsample1x1 = torch.nn.Conv2d(nin, expand1x1_out + expand3x3_out, kernel_size=1, stride=2)
            else:
                self.downsample1x1 = torch.nn.Conv2d(nin, expand1x1_out + expand3x3_out, kernel_size=1)
            self.downsampleBN = torch.nn.BatchNorm2d(expand1x1_out + expand3x3_out)
        else:
            self.downsample1x1 = None
            self.downsampleBN = None

    def forward(self, x):
        shortcut = x

        x = self.squeeze_activation(self.squeeze(x))
        x = torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)

        if self.downsample:
            shortcut = self.downsample1x1(shortcut)
            shortcut = self.downsampleBN(shortcut)

        return shortcut + x

def get_model(num_classes):
    model = torchvision.models.resnet101(num_classes=num_classes)
    model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = torch.nn.Identity()

    #layer1
    model.layer1[0] = Fire(64, 64, 128, 128, downsample=True)
    model.layer1[1] = Fire(256, 64, 128, 128)
    model.layer1[2] = Fire(256, 64, 128, 128)
    #layer2
    model.layer2[0] = Fire(256, 128, 256, 256, 2, downsample=True)
    model.layer2[1] = Fire(512, 128, 256, 256)
    model.layer2[2] = Fire(512, 128, 256, 256)
    model.layer2[3] = Fire(512, 128, 256, 256)
    #layer3
    model.layer3[0] = Fire(512, 256, 512, 512, 2, downsample=True)
    model.layer3[1] = Fire(1024, 256, 512, 512)
    model.layer3[2] = Fire(1024, 256, 512, 512)
    model.layer3[3] = Fire(1024, 256, 512, 512)
    model.layer3[4] = Fire(1024, 256, 512, 512)
    model.layer3[5] = Fire(1024, 256, 512, 512)
    model.layer3[6] = Fire(1024, 256, 512, 512)
    model.layer3[7] = Fire(1024, 256, 512, 512)
    model.layer3[8] = Fire(1024, 256, 512, 512)
    model.layer3[9] = Fire(1024, 256, 512, 512)
    model.layer3[10] = Fire(1024, 256, 512, 512)
    model.layer3[11] = Fire(1024, 256, 512, 512)
    model.layer3[12] = Fire(1024, 256, 512, 512)
    model.layer3[13] = Fire(1024, 256, 512, 512)
    model.layer3[14] = Fire(1024, 256, 512, 512)
    model.layer3[15] = Fire(1024, 256, 512, 512)
    model.layer3[16] = Fire(1024, 256, 512, 512)
    model.layer3[17] = Fire(1024, 256, 512, 512)
    model.layer3[18] = Fire(1024, 256, 512, 512)
    model.layer3[19] = Fire(1024, 256, 512, 512)
    model.layer3[20] = Fire(1024, 256, 512, 512)
    model.layer3[21] = Fire(1024, 256, 512, 512)
    model.layer3[22] = Fire(1024, 256, 512, 512)
    #layer4
    model.layer4[0] = Fire(1024, 512, 1024, 1024, 2, downsample=True)
    model.layer4[1] = Fire(2048, 512, 1024, 1024)
    model.layer4[2] = Fire(2048, 512, 1024, 1024)

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