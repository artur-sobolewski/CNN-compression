import torch
import torchvision

from pytorch_lightning import LightningDataModule

class DataModule(LightningDataModule):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.mean = (0.5071, 0.4867, 0.4408)
        self.std = (0.2675, 0.2565, 0.2761)    

    def train_dataloader(self):
        train_transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.mean, self.std)
        ])
        
        train_set = torchvision.datasets.CIFAR100(root="./data/", train=True, download=True, transform=train_transforms)
        return torch.utils.data.DataLoader(train_set, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        test_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.mean, self.std)
        ])
        
        test_set = torchvision.datasets.CIFAR100(root="./data/", train=False, download=True, transform=test_transforms)
        return torch.utils.data.DataLoader(test_set, batch_size=self.batch_size, shuffle=False)