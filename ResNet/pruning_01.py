import torch
import torchvision

from utils.data_module import DataModule
from utils.model_pruning import PruningModel

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# >>>
# Configuration

NAME = "01_ResNet_baseline"
SAVE_NAME = "01_ResNet_baseline_pruned"
DEVICES = [0]
SEED = 0
BATCH_SIZE = 128
MAX_EPOCHS = 1 # Max epoch num per iteration
ITERATIONS = 21 # To 0.99 sparsity
PRUNING_RATE = 0.2

model = torchvision.models.resnet101(num_classes=10)
model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = torch.nn.Identity()

# ^^^

def main(model):      
    seed_everything(SEED, workers=True)

    checkpoint = torch.load("./saved_models/{}.ckpt".format(NAME))
    loaded_model = PruningModel(SAVE_NAME, model, pruning_amount=PRUNING_RATE)
    loaded_model.load_state_dict(checkpoint['state_dict'])

    for iter in range(ITERATIONS):
        print("STARTED {} ITER".format(iter))
        loaded_model.set_model_name(NAME + "_pruning_{}".format(iter))
        logger = TensorBoardLogger(save_dir="./logs/", name=NAME + "_pruning_{}".format(iter), default_hp_metric=False)
        lr_monitor = LearningRateMonitor(logging_interval='step')
        early_stopping = EarlyStopping(monitor="Loss/train", mode="min", patience=5)

        data = DataModule(batch_size=BATCH_SIZE)

        trainer = Trainer(max_epochs=MAX_EPOCHS, accelerator="gpu", devices=DEVICES, logger=logger, callbacks=[lr_monitor, early_stopping], deterministic=True)
        
        trainer.fit(loaded_model, datamodule=data)

        trainer.save_checkpoint(loaded_model.saved_model_path, weights_only=True)
        trainer.test(loaded_model, data.val_dataloader())


if __name__ == '__main__':
    main(model)