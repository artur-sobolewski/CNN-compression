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

LOAD_NAME = "11_invResB_ghostBlocks_small_3_pruned"
SAVE_NAME = "11_invResB_ghostBlocks_small_3_pruned_quantized"
LOAD_NAME_2 = "11_invResB_ghostBlocks_small_cifar100"
SAVE_NAME_2 = "11_invResB_ghostBlocks_small_cifar100_quantized"
DEVICES = [0]
SEED = 0
BATCH_SIZE = 128
MAX_EPOCHS = 0

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
    def __init__(self, nin, nout, stride = 1, padding = 1, relu=True):
        super(ghostModule, self).__init__()

        self.nin = nin
        self.nghost = nout//2
        self.nout = nout
        
        self.conv1 = torch.nn.Conv2d(nin, self.nghost, kernel_size=1, stride=1, bias=False)
        self.conv2 = torch.nn.Conv2d(self.nghost, self.nghost, kernel_size=3, padding=padding, stride=(stride, stride) , groups=self.nghost, bias=False)
        self.bn = torch.nn.BatchNorm2d(nout)
        if (relu):
            self.relu = torch.nn.ReLU6(inplace=False)
        else:
            self.relu = None

        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):

        out = self.conv1(x)
        ghost = self.conv2(out)

        out = self.dequant(out)
        ghost = self.dequant(ghost)
        out = torch.cat((ghost, out), axis=1)
        out = self.quant(out)

        out = self.bn(out)
        if (self.relu):
            out = self.relu(out)

        return out


class ghostBlock(torch.nn.Module):
    def __init__(self, nin, nexpand, nout, stride = 1, padding = 1, downsample=False):
        super(ghostBlock, self).__init__()

        self.downsample = downsample
        self.nin = nin
        self.nexpand = nexpand
        self.nout = nout

        if self.downsample:
            self.downsampleSkip = torch.nn.Sequential(
                torch.nn.Conv2d(nin, nout, kernel_size=(1, 1), stride=stride, bias=False),
                torch.nn.BatchNorm2d(nout)
            )
            self.downsampleLayer = torch.nn.Sequential(
                torch.nn.Conv2d(nexpand, nexpand, kernel_size=3, padding=padding, stride=(stride, stride) , groups=nexpand, bias=False),
                torch.nn.BatchNorm2d(nexpand)
            )
        else:
            self.downsampleSkip = None
            self.downsampleLayer = None

        self.ghost1 = ghostModule(nin, nexpand)
        self.ghost2 = ghostModule(nexpand, nout, relu=False)

        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        skipConnection = x

        out = self.ghost1(x)
        if self.downsample:
            out = self.downsampleLayer(out)

        out = self.ghost2(out)

        if self.downsample:
            skipConnection = self.downsampleSkip(skipConnection)

        out = self.dequant(out)
        skipConnection = self.dequant(skipConnection)
        out += skipConnection
        out = self.quant(out)

        return out

def get_model(num_classes):
    model = torchvision.models.resnet101(num_classes=num_classes)
    model.conv1 = DWSconv(3, 16, pointwise=True)
    model.relu = torch.nn.ReLU6(inplace=False)
    model.bn1 = torch.nn.BatchNorm2d(16)
    model.maxpool = torch.nn.Identity()

    #layer1
    model.layer1[0] = ghostBlock(16, 64, 10, downsample=True)
    model.layer1[1] = ghostBlock(10, 64, 10)
    model.layer1[2] = ghostBlock(10, 64, 10)
    #layer2
    model.layer2[0] = ghostBlock(10, 64, 22, stride=2, downsample=True)
    model.layer2[1] = ghostBlock(22, 128, 22)
    model.layer2[2] = ghostBlock(22, 128, 22)
    model.layer2[3] = ghostBlock(22, 128, 22)
    #layer3
    model.layer3[0] = ghostBlock(22, 128, 42, stride=2, downsample=True)
    model.layer3[1] = ghostBlock(42, 256, 42)
    model.layer3[2] = ghostBlock(42, 256, 42)
    model.layer3[3] = ghostBlock(42, 256, 42)
    model.layer3[4] = ghostBlock(42, 256, 42)
    model.layer3[5] = ghostBlock(42, 256, 42)
    model.layer3[6] = ghostBlock(42, 256, 42)
    model.layer3[7] = ghostBlock(42, 256, 42)
    model.layer3[8] = ghostBlock(42, 256, 42)
    model.layer3[9] = ghostBlock(42, 256, 42)
    model.layer3[10] = ghostBlock(42, 256, 42)
    model.layer3[11] = ghostBlock(42, 256, 42)
    model.layer3[12] = ghostBlock(42, 256, 42)
    model.layer3[13] = ghostBlock(42, 256, 42)
    model.layer3[14] = ghostBlock(42, 256, 42)
    model.layer3[15] = ghostBlock(42, 256, 42)
    model.layer3[16] = ghostBlock(42, 256, 42)
    model.layer3[17] = ghostBlock(42, 256, 42)
    model.layer3[18] = ghostBlock(42, 256, 42)
    model.layer3[19] = ghostBlock(42, 256, 42)
    model.layer3[20] = ghostBlock(42, 256, 42)
    model.layer3[21] = ghostBlock(42, 256, 42)
    model.layer3[22] = ghostBlock(42, 256, 42)
    #layer4
    model.layer4 = torch.nn.Sequential(
        ghostBlock(42, 512, 86, stride=2, downsample=True),
        ghostBlock(86, 512, 86),

        torch.nn.Conv2d(86, num_classes, kernel_size=1, stride=1, bias=False),
        torch.nn.BatchNorm2d(num_classes),
        torch.nn.ReLU6(inplace=False)
    )

    model.fc = torch.nn.Identity()

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


    # Quantizing pruned model
    checkpoints_path = os.path.join(os.getcwd(), "saved_models", "pruning")
    checkpoint = torch.load(os.path.join(checkpoints_path, "{}.ckpt".format(LOAD_NAME)))
    loaded_model = Model(SAVE_NAME, model)
    loaded_model.load_state_dict(checkpoint['state_dict'])

    model.load_state_dict(loaded_model.model.state_dict())

    quant_model = Model(SAVE_NAME, QuantizedResNet(model))


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