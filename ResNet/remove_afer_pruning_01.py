import os
import torch
import torchvision

from utils.data_module import DataModule
from utils.model_pruning import PruningModel

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
# >>>
# Configuration

LOAD_NAME = "01_baseline_cifar100_pruning_3"
SAVE_NAME = "01_ResNet_baseline_cifar100_3_pruned"
DEVICES = [0]
SEED = 0
PRUNING_RATE = 0.2
BATCH_SIZE = 128
MAX_EPOCHS = 0

def get_model(num_classes):
    model = torchvision.models.resnet101(num_classes=num_classes)
    model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = torch.nn.Identity()

    return model

# ^^^

def main(model, dataset_name):      
    seed_everything(SEED, workers=True)

    checkpoints_path = os.path.join(os.getcwd(), "saved_models", "pruning")
    checkpoint = torch.load(os.path.join(checkpoints_path, "{}.ckpt".format(LOAD_NAME)))
    loaded_model = PruningModel(SAVE_NAME, model, pruning_amount=PRUNING_RATE)
    
    
    logger = TensorBoardLogger(save_dir="./logs/", name=SAVE_NAME, default_hp_metric=False)
    data = DataModule(batch_size=BATCH_SIZE, dataset_name=dataset_name)

    trainer = Trainer(max_epochs=MAX_EPOCHS, accelerator="cpu", logger=logger, deterministic=True)

    loaded_model.set_model_name(SAVE_NAME)

    trainer.fit(loaded_model, datamodule=data)

    loaded_model.prune_model(amount=0)
    loaded_model.load_state_dict(checkpoint['state_dict'])

    num_zeros, num_elements, sparsity = loaded_model.measure_global_sparsity(
            weight = True, bias = False,
            conv2d_use_mask = True,
            linear_use_mask = False)

    print(num_zeros, num_elements, sparsity)

    loaded_model.remove_parameters()

    trainer.save_checkpoint(loaded_model.saved_model_path, weights_only=True)

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