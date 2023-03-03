import torch
import torchvision

from utils.data_module import DataModule
from utils.model_pruning import PruningModel

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger

# >>>
# Configuration

LOAD_NAME = "01_ResNet_baseline_pruning_20"
SAVE_NAME = "01_ResNet_baseline_20_pruned"
DEVICES = [0]
SEED = 0
PRUNING_RATE = 0.2
BATCH_SIZE = 128
MAX_EPOCHS = 0

model = torchvision.models.resnet101(num_classes=10)
model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = torch.nn.Identity()

# ^^^

def main(model):      
    seed_everything(SEED, workers=True)

    checkpoint = torch.load("./saved_models/pruning/01/{}.ckpt".format(LOAD_NAME))
    loaded_model = PruningModel(SAVE_NAME, model, pruning_amount=PRUNING_RATE)
    
    
    logger = TensorBoardLogger(save_dir="./logs/", name=SAVE_NAME, default_hp_metric=False)
    data = DataModule(batch_size=BATCH_SIZE)

    trainer = Trainer(max_epochs=MAX_EPOCHS, accelerator="gpu", devices=DEVICES, logger=logger, deterministic=True)

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
    main(model)