import os
import torch
import torchvision

from pytorch_lightning import LightningDataModule

class DataModule(LightningDataModule):
    def __init__(self, batch_size, dataset_name):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        
        
        if self.dataset_name == "cifar10":
            self.mean = (0.4914, 0.4822, 0.4465)
            self.std = (0.2470, 0.2435, 0.2616)   
        elif self.dataset_name == "cifar100":
            self.mean = (0.5071, 0.4867, 0.4408)
            self.std = (0.2675, 0.2565, 0.2761)    

    def train_dataloader(self):
        train_transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.mean, self.std)
        ])
        
        root_path = os.path.join(os.getcwd(), "data")
        if self.dataset_name == "cifar10":
            train_set = torchvision.datasets.CIFAR10(root=root_path, train=True, download=True, transform=train_transforms)
        elif self.dataset_name == "cifar100":
            train_set = torchvision.datasets.CIFAR100(root=root_path, train=True, download=True, transform=train_transforms)
        return torch.utils.data.DataLoader(train_set, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        test_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.mean, self.std)
        ])
        
        root_path = os.path.join(os.getcwd(), "data")
        if self.dataset_name == "cifar10":
            test_set = torchvision.datasets.CIFAR10(root=root_path, train=False, download=True, transform=test_transforms)
        elif self.dataset_name == "cifar100":
            test_set = torchvision.datasets.CIFAR100(root=root_path, train=False, download=True, transform=test_transforms)
        return torch.utils.data.DataLoader(test_set, batch_size=self.batch_size, shuffle=False)