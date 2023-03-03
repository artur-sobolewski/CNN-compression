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

NAME = "06_invResBottleneck"
DEVICES = [1]
SEED = 0
BATCH_SIZE = 128
MAX_EPOCHS = -1

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

model = torchvision.models.resnet101(num_classes=10)
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
    torch.nn.Conv2d(512, 2048, kernel_size=1, stride=1, bias=False),
    torch.nn.BatchNorm2d(2048),
    torch.nn.ReLU6(inplace=True)
)

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