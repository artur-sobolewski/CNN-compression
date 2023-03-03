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

NAME = "04_DWSconv"
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

model = torchvision.models.resnet101(num_classes=10)
model.conv1 = DWSconv(3, 64, pointwise=True)
model.maxpool = torch.nn.Identity()

#layer1
model.layer1[0].conv2 = DWSconv(64, 64)
model.layer1[1].conv2 = DWSconv(64, 64)
model.layer1[2].conv2 = DWSconv(64, 64)
#layer2
model.layer2[0].conv2 = DWSconv(128, 128, stride=2)
model.layer2[1].conv2 = DWSconv(128, 128)
model.layer2[2].conv2 = DWSconv(128, 128)
model.layer2[3].conv2 = DWSconv(128, 128)
#layer3
model.layer3[0].conv2 = DWSconv(256, 256, stride=2)
model.layer3[1].conv2 = DWSconv(256, 256)
model.layer3[2].conv2 = DWSconv(256, 256)
model.layer3[3].conv2 = DWSconv(256, 256)
model.layer3[4].conv2 = DWSconv(256, 256)
model.layer3[5].conv2 = DWSconv(256, 256)
model.layer3[6].conv2 = DWSconv(256, 256)
model.layer3[7].conv2 = DWSconv(256, 256)
model.layer3[8].conv2 = DWSconv(256, 256)
model.layer3[9].conv2 = DWSconv(256, 256)
model.layer3[10].conv2 = DWSconv(256, 256)
model.layer3[11].conv2 = DWSconv(256, 256)
model.layer3[12].conv2 = DWSconv(256, 256)
model.layer3[13].conv2 = DWSconv(256, 256)
model.layer3[14].conv2 = DWSconv(256, 256)
model.layer3[15].conv2 = DWSconv(256, 256)
model.layer3[16].conv2 = DWSconv(256, 256)
model.layer3[17].conv2 = DWSconv(256, 256)
model.layer3[18].conv2 = DWSconv(256, 256)
model.layer3[19].conv2 = DWSconv(256, 256)
model.layer3[20].conv2 = DWSconv(256, 256)
model.layer3[21].conv2 = DWSconv(256, 256)
model.layer3[22].conv2 = DWSconv(256, 256)
#layer4
model.layer4[0].conv2 = DWSconv(512, 512, stride=2)
model.layer4[1].conv2 = DWSconv(512, 512)
model.layer4[2].conv2 = DWSconv(512, 512)

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