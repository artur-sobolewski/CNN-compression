import os
import torch
import torchvision

from utils.data_module import DataModule
from utils.model import Model

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

# >>>
# Configuration

LOAD_NAME = "01_ResNet_baseline_cifar100_3_pruned"
SAVE_NAME = "01_ResNet_baseline_3_pruned_quantized"
LOAD_NAME_2 = "01_baseline_cifar100"
SAVE_NAME_2 = "01_baseline_cifar100_quantized"
DEVICES = [0]
SEED = 0
BATCH_SIZE = 128
MAX_EPOCHS = 0

class Bottleneck(torch.nn.Module):
    def __init__(self, nin, nhidden, nout, kernel_size=3, stride=1, padding=1, bias=False, downsample=False):
        super(Bottleneck, self).__init__()

        self.isdownsample = downsample

        self.conv1 = torch.nn.Conv2d(nin, nhidden, kernel_size=1, stride=1, bias=bias)
        self.bn1 = torch.nn.BatchNorm2d(nhidden)
        self.conv2 = torch.nn.Conv2d(nhidden, nhidden, kernel_size=kernel_size, padding=padding, stride=stride, bias=bias)
        self.bn2 = torch.nn.BatchNorm2d(nhidden)
        self.conv3 = torch.nn.Conv2d(nhidden, nout, kernel_size=1, stride=1, bias=bias)
        self.bn3 = torch.nn.BatchNorm2d(nout)
        self.relu = torch.nn.ReLU(inplace=True)

        if downsample:
            self.downsample = torch.nn.Sequential(
                torch.nn.Conv2d(nin, nout, kernel_size=1, stride=stride, bias=bias),
                torch.nn.BatchNorm2d(nout)
            )
        
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
    
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.isdownsample:
            identity = self.downsample(x)

        out = self.dequant(out)
        identity = self.dequant(identity)
        out += identity
        out = self.quant(out)
        out = self.relu(out)

        return out

def get_model(num_classes):
    model = torchvision.models.resnet101(num_classes=num_classes)
    model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = torch.nn.Identity()

    #layer1
    model.layer1[0] = Bottleneck(64, 64, 256, downsample=True)
    model.layer1[1] = Bottleneck(256, 64, 256)
    model.layer1[2] = Bottleneck(256, 64, 256)
    #layer2
    model.layer2[0] = Bottleneck(256, 128, 512, stride=2, downsample=True)
    model.layer2[1] = Bottleneck(512, 128, 512)
    model.layer2[2] = Bottleneck(512, 128, 512)
    model.layer2[3] = Bottleneck(512, 128, 512)
    #layer3
    model.layer3[0] = Bottleneck(512, 256, 1024, stride=2, downsample=True)
    model.layer3[1] = Bottleneck(1024, 256, 1024)
    model.layer3[2] = Bottleneck(1024, 256, 1024)
    model.layer3[3] = Bottleneck(1024, 256, 1024)
    model.layer3[4] = Bottleneck(1024, 256, 1024)
    model.layer3[5] = Bottleneck(1024, 256, 1024)
    model.layer3[6] = Bottleneck(1024, 256, 1024)
    model.layer3[7] = Bottleneck(1024, 256, 1024)
    model.layer3[8] = Bottleneck(1024, 256, 1024)
    model.layer3[9] = Bottleneck(1024, 256, 1024)
    model.layer3[10] = Bottleneck(1024, 256, 1024)
    model.layer3[11] = Bottleneck(1024, 256, 1024)
    model.layer3[12] = Bottleneck(1024, 256, 1024)
    model.layer3[13] = Bottleneck(1024, 256, 1024)
    model.layer3[14] = Bottleneck(1024, 256, 1024)
    model.layer3[15] = Bottleneck(1024, 256, 1024)
    model.layer3[16] = Bottleneck(1024, 256, 1024)
    model.layer3[17] = Bottleneck(1024, 256, 1024)
    model.layer3[18] = Bottleneck(1024, 256, 1024)
    model.layer3[19] = Bottleneck(1024, 256, 1024)
    model.layer3[20] = Bottleneck(1024, 256, 1024)
    model.layer3[21] = Bottleneck(1024, 256, 1024)
    model.layer3[22] = Bottleneck(1024, 256, 1024)
    #layer4
    model.layer4[0] = Bottleneck(1024, 512, 2048, stride=2, downsample=True)
    model.layer4[1] = Bottleneck(2048, 512, 2048)
    model.layer4[2] = Bottleneck(2048, 512, 2048)

    return model

class QuantizedResNet(torch.nn.Sequential):
    def __init__(self, model):
        super(QuantizedResNet, self).__init__()

        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        self.model = model

    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x

# ^^^

def main(model, dataset_name):      
    seed_everything(SEED, workers=True)

    print(model)

    # Quantizing pruned model

    checkpoints_path = os.path.join(os.getcwd(), "saved_models", "pruning")
    checkpoint = torch.load(os.path.join(checkpoints_path, "{}.ckpt".format(LOAD_NAME)))
    loaded_model = Model(SAVE_NAME, model)
    loaded_model.load_state_dict(checkpoint['state_dict'])

    model.load_state_dict(loaded_model.model.state_dict())

    quant_model = Model(SAVE_NAME, QuantizedResNet(model))

    # quant_model = Model(SAVE_NAME, model)

    logger = TensorBoardLogger(save_dir="./logs/", name=SAVE_NAME, default_hp_metric=False)
    data = DataModule(batch_size=BATCH_SIZE, dataset_name=dataset_name)

    trainer = Trainer(max_epochs=MAX_EPOCHS, accelerator="cpu", logger=logger, deterministic=True)

    trainer.fit(quant_model, datamodule=data)

    quant_model.prepare_quantization()
    trainer.validate(quant_model, data.val_dataloader())
    quant_model.quantize()

    trainer.save_checkpoint(quant_model.saved_model_path, weights_only=True)
    trainer.test(quant_model, data.val_dataloader())


    # Quantizing not pruned model

    checkpoints_path_2 = os.path.join(os.getcwd(), "saved_models")
    checkpoint_2 = torch.load(os.path.join(checkpoints_path_2, "{}.ckpt".format(LOAD_NAME_2)))
    loaded_model_2 = Model(SAVE_NAME_2, model)
    loaded_model_2.load_state_dict(checkpoint_2['state_dict'])

    model.load_state_dict(loaded_model_2.model.state_dict())

    quant_model_2 = Model(SAVE_NAME_2, QuantizedResNet(model))


    logger = TensorBoardLogger(save_dir="./logs/", name=SAVE_NAME, default_hp_metric=False)
    data = DataModule(batch_size=BATCH_SIZE, dataset_name=dataset_name)

    trainer = Trainer(max_epochs=MAX_EPOCHS, accelerator="cpu", logger=logger, deterministic=True)

    trainer.fit(quant_model_2, datamodule=data)

    quant_model_2.prepare_quantization()
    trainer.validate(quant_model_2, data.val_dataloader())
    quant_model_2.quantize()

    trainer.save_checkpoint(quant_model_2.saved_model_path, weights_only=True)
    trainer.test(quant_model_2, data.val_dataloader())

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