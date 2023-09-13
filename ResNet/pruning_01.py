import os
import torch
import torchvision
from dotenv import load_dotenv

from utils.data_module import DataModule
from utils.model_pruning import PruningModel

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

# >>>
# Configuration

load_dotenv()

NAME = "01_baseline"
SAVE_NAME = "01_baseline_pruned"
DEVICES = [0]
SEED = 0
MAX_EPOCHS = 50 # Max epoch num per iteration
ITERATIONS = 10
PRUNING_RATE = 0.2

def get_model(num_classes):
    model = torchvision.models.resnet101(num_classes=num_classes)
    model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = torch.nn.Identity()

    return model

# ^^^

def main(model, dataset_name):
    if dataset_name == "cifar10":
        batch_size = int(os.getenv('CIFAR10_BATCH_SIZE'))
    elif dataset_name == "cifar100":
        batch_size = int(os.getenv('CIFAR100_BATCH_SIZE'))
    name = NAME + "_{}".format(dataset_name)
    save_name = SAVE_NAME + "_{}".format(dataset_name)
    seed_everything(SEED, workers=True)

    checkpoints_path = os.path.join(os.getcwd(), "saved_models")
    checkpoint = torch.load(os.path.join(checkpoints_path, "{}.ckpt".format(name)))
    loaded_model = PruningModel(save_name, model, pruning_amount=PRUNING_RATE)
    loaded_model.load_state_dict(checkpoint['state_dict'])

    for iter in range(ITERATIONS):
        print("STARTED {} ITER".format(iter))
        loaded_model.set_model_name(name + "_pruning_{}".format(iter))
        logs_path = os.path.join(os.getcwd(), "logs")
        logger = TensorBoardLogger(save_dir=logs_path, name=name + "_pruning_{}".format(iter), default_hp_metric=False)
        lr_monitor = LearningRateMonitor(logging_interval='step')
        early_stopping = EarlyStopping(monitor="Loss/train", mode="min", patience=5)

        data = DataModule(batch_size, dataset_name)

        trainer = Trainer(max_epochs=MAX_EPOCHS, accelerator="gpu", devices=DEVICES, logger=logger, callbacks=[lr_monitor, early_stopping], deterministic=True)
        
        trainer.fit(loaded_model, datamodule=data)

        trainer.save_checkpoint(loaded_model.saved_model_path, weights_only=True)
        trainer.test(loaded_model, data.val_dataloader())


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--dataset', required=True, choices=["cifar10", "cifar100"])

    args = vars(parser.parse_args())

    dataset_name = args['dataset']
    if dataset_name == "cifar10":
        num_classes = 10
    elif dataset_name == "cifar100":
        num_classes = 100
    
    model = get_model(num_classes)

    main(model, dataset_name)