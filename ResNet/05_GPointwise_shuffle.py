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

NAME = "05_GPointwise_shuffle"
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
    def __init__(self, nin, nout, bias=False, groups=3, shuffle=True, isrelu=False):
        super().__init__()

        self.shuffle = shuffle
        self.isrelu = isrelu

        if nin > 24:
            self.groups = groups
        else:
            self.groups = 1

        self.conv1 = torch.nn.Conv2d(nin, nout, kernel_size=1, groups=self.groups, bias=bias)
        self.bn1 = torch.nn.BatchNorm2d(nout)
        if self.isrelu:
            self.relu1 = torch.nn.ReLU(inplace=True)
        if self.shuffle:
            self.shuffle = ChannelShuffle(nout, groups)

    def forward(self, x):
        out = self.conv1(x)
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
    
class Bottleneck(torch.nn.Module):
    def __init__(self, nin, nhidden, nout, kernel_size=3, padding=1, stride=1, bias=False, groups=4, downsample=False):
        super(Bottleneck, self).__init__()

        self.downsample = downsample

        self.conv1 = groupPointwiseConv(nin, nhidden, groups=groups, isrelu=True)
        self.conv2 = DWSconv(nhidden, nhidden, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn2 = torch.nn.BatchNorm2d(nhidden)
        self.conv3 = groupPointwiseConv(nhidden, nout, groups=groups)
        self.reluout = torch.nn.ReLU(inplace=True)

        if downsample:
            self.downsample = torch.nn.Sequential(
                torch.nn.Conv2d(nin, nout, kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(nout)
            )
    
    def forward(self, x):
        skipconnection = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.conv3(out)
        if self.downsample:
            skipconnection = self.downsample(skipconnection)
        return self.reluout(out + skipconnection)

model = torchvision.models.resnet101(num_classes=10)
model.conv1 = DWSconv(3, 64, pointwise=True)
model.maxpool = torch.nn.Identity()

#layer1
model.layer1[0] = Bottleneck(64, 64, 256, groups=4, downsample=True)
model.layer1[1] = Bottleneck(256, 64, 256, groups=4)
model.layer1[2] = Bottleneck(256, 64, 256, groups=4)
#layer2
model.layer2[0] = Bottleneck(256, 128, 512, groups=4, stride=2, downsample=True)
model.layer2[1] = Bottleneck(512, 128, 512, groups=4)
model.layer2[2] = Bottleneck(512, 128, 512, groups=4)
model.layer2[3] = Bottleneck(512, 128, 512, groups=4)
#layer3
model.layer3[0] = Bottleneck(512, 256, 1024, groups=4, stride=2, downsample=True)
model.layer3[1] = Bottleneck(1024, 256, 1024, groups=4)
model.layer3[2] = Bottleneck(1024, 256, 1024, groups=4)
model.layer3[3] = Bottleneck(1024, 256, 1024, groups=4)
model.layer3[4] = Bottleneck(1024, 256, 1024, groups=4)
model.layer3[5] = Bottleneck(1024, 256, 1024, groups=4)
model.layer3[6] = Bottleneck(1024, 256, 1024, groups=4)
model.layer3[7] = Bottleneck(1024, 256, 1024, groups=4)
model.layer3[8] = Bottleneck(1024, 256, 1024, groups=4)
model.layer3[9] = Bottleneck(1024, 256, 1024, groups=4)
model.layer3[10] = Bottleneck(1024, 256, 1024, groups=4)
model.layer3[11] = Bottleneck(1024, 256, 1024, groups=4)
model.layer3[12] = Bottleneck(1024, 256, 1024, groups=4)
model.layer3[13] = Bottleneck(1024, 256, 1024, groups=4)
model.layer3[14] = Bottleneck(1024, 256, 1024, groups=4)
model.layer3[15] = Bottleneck(1024, 256, 1024, groups=4)
model.layer3[16] = Bottleneck(1024, 256, 1024, groups=4)
model.layer3[17] = Bottleneck(1024, 256, 1024, groups=4)
model.layer3[18] = Bottleneck(1024, 256, 1024, groups=4)
model.layer3[19] = Bottleneck(1024, 256, 1024, groups=4)
model.layer3[20] = Bottleneck(1024, 256, 1024, groups=4)
model.layer3[21] = Bottleneck(1024, 256, 1024, groups=4)
model.layer3[22] = Bottleneck(1024, 256, 1024, groups=4)
#layer4
model.layer4[0] = Bottleneck(1024, 512, 2048, groups=4, stride=2, downsample=True)
model.layer4[1] = Bottleneck(2048, 512, 2048, groups=4)
model.layer4[2] = Bottleneck(2048, 512, 2048, groups=4)

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