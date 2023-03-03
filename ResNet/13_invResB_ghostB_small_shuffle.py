import torch
import torchvision

from utils.data_module import DataModule
from utils.model import Model

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# >>>
# Configuration

NAME = "13_invResB_ghostBlocks_small_shuffle"
DEVICES = [1]
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
    def __init__(self, nin, nout, bias=False, groups=3, shuffle=True, isrelu=False, isBN=False):
        super().__init__()

        self.shuffle = shuffle
        self.isrelu = isrelu
        self.isBN = isBN
        self.groups = groups
        
        self.conv1 = torch.nn.Conv2d(nin, nout, kernel_size=1, groups=self.groups, bias=bias)
        if self.isBN:
            self.bn1 = torch.nn.BatchNorm2d(nout)
        if self.isrelu:
            self.relu1 = torch.nn.ReLU(inplace=True)
        if self.shuffle:
            self.shuffle = ChannelShuffle(nout, groups)

    def forward(self, x):
        out = self.conv1(x)
        if self.isBN:
            out = self.bn1(out)
        if self.isrelu:
            out = self.relu1(out)
        if self.shuffle:
            out = self.shuffle(out)
        return out

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
    def __init__(self, nin, nout, stride = 1, padding = 1, isrelu=True):
        super(ghostModule, self).__init__()

        self.isrelu = isrelu

        self.nin = nin
        self.nghost = nout//2
        self.nout = nout
        
        self.conv1 = groupPointwiseConv(nin, self.nghost, groups=4)
        self.conv2 = torch.nn.Conv2d(self.nghost, self.nghost, kernel_size=3, padding=padding, stride=(stride, stride), groups=self.nghost, bias=False)
        self.bn = torch.nn.BatchNorm2d(nout)
        if (self.isrelu):
            self.relu = torch.nn.ReLU6(inplace=True)

    def forward(self, x):

        out = self.conv1(x)
        ghost = self.conv2(out)

        out = torch.cat((ghost, out), axis=1)

        out = self.bn(out)
        if (self.isrelu):
            out = self.relu(out)

        return out


class ghostBlock(torch.nn.Module):
    def __init__(self, nin, nexpand, nout, stride = 1, padding = 1, downsample=False):
        super(ghostBlock, self).__init__()

        self.downsample = downsample
        self.nin = nin
        self.nexpand = nexpand
        self.nout = nout
        self.stride = stride

        if self.downsample:
            self.downsampleSkip = torch.nn.Sequential(
                torch.nn.Conv2d(nin, nout, kernel_size=(1, 1), stride=stride, bias=False),
                torch.nn.BatchNorm2d(nout)
            )
        if stride > 1:
            self.downsampleLayer = torch.nn.Sequential(
                torch.nn.Conv2d(nexpand, nexpand, kernel_size=3, padding=padding, stride=(stride, stride) , groups=nexpand, bias=False),
                torch.nn.BatchNorm2d(nexpand)
            )

        self.ghost1 = ghostModule(nin, nexpand)
        self.ghost2 = ghostModule(nexpand, nout, isrelu=False)

    def forward(self, x):
        skipConnection = x

        out = self.ghost1(x)
        if self.stride > 1:
            out = self.downsampleLayer(out)

        out = self.ghost2(out)

        if self.downsample:
            skipConnection = self.downsampleSkip(skipConnection)

        out += skipConnection

        return out



model = torchvision.models.resnet101(num_classes=10)
model.conv1 = DWSconv(3, 16, pointwise=True)
model.bn1 = torch.nn.BatchNorm2d(16)
model.maxpool = torch.nn.Identity()

#layer1
model.layer1[0] = ghostBlock(16, 64, 16, downsample=True)
model.layer1[1] = ghostBlock(16, 64, 16)
model.layer1[2] = ghostBlock(16, 64, 16)
#layer2
model.layer2[0] = ghostBlock(16, 64, 24, stride=2, downsample=True)
model.layer2[1] = ghostBlock(24, 128, 24)
model.layer2[2] = ghostBlock(24, 128, 24)
model.layer2[3] = ghostBlock(24, 128, 24)
#layer3
model.layer3[0] = ghostBlock(24, 128, 40, stride=2, downsample=True)
model.layer3[1] = ghostBlock(40, 256, 40)
model.layer3[2] = ghostBlock(40, 256, 40)
model.layer3[3] = ghostBlock(40, 256, 40)
model.layer3[4] = ghostBlock(40, 256, 40)
model.layer3[5] = ghostBlock(40, 256, 40)
model.layer3[6] = ghostBlock(40, 256, 40)
model.layer3[7] = ghostBlock(40, 256, 40)
model.layer3[8] = ghostBlock(40, 256, 40)
model.layer3[9] = ghostBlock(40, 256, 40)
model.layer3[10] = ghostBlock(40, 256, 40)
model.layer3[11] = ghostBlock(40, 256, 40)
model.layer3[12] = ghostBlock(40, 256, 40)
model.layer3[13] = ghostBlock(40, 256, 40)
model.layer3[14] = ghostBlock(40, 256, 40)
model.layer3[15] = ghostBlock(40, 256, 40)
model.layer3[16] = ghostBlock(40, 256, 40)
model.layer3[17] = ghostBlock(40, 256, 40)
model.layer3[18] = ghostBlock(40, 256, 40)
model.layer3[19] = ghostBlock(40, 256, 40)
model.layer3[20] = ghostBlock(40, 256, 40)
model.layer3[21] = ghostBlock(40, 256, 40)
model.layer3[22] = ghostBlock(40, 256, 40)
#layer4
model.layer4 = torch.nn.Sequential(
    ghostBlock(40, 512, 88, stride=2, downsample=True),
    ghostBlock(88, 512, 88),

    torch.nn.Conv2d(88, 10, kernel_size=1, stride=1, bias=False),
    torch.nn.BatchNorm2d(10),
    torch.nn.ReLU6(inplace=True)
)

model.fc = torch.nn.Identity()

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