import os
import torch
import numpy as np
from dotenv import load_dotenv

from pytorch_lightning.core.module import LightningModule
from fvcore.nn import FlopCountAnalysis

load_dotenv()

class Model(LightningModule):
    def __init__(self, name, model):
        super().__init__()
        self.name = name
        self.model = model
        self.onecyclelr = (os.getenv('ONE_CYCLE_LR') == 'true')

        self.loss_module = torch.nn.CrossEntropyLoss()

        self.test_device = torch.device('cuda')
        
        self.cwd_path = os.getcwd()

        if not os.path.exists(os.path.join(self.cwd_path, "results")):
            os.mkdir(os.path.join(self.cwd_path, "results"))
        if not os.path.exists(os.path.join(self.cwd_path, "saved_models")):
            os.mkdir(os.path.join(self.cwd_path, "saved_models"))

        self.result_file_path = os.path.join(self.cwd_path, "results", "{}.txt".format(self.name))
        self.saved_model_path = os.path.join(self.cwd_path, "saved_models", "{}.ckpt".format(self.name))
    
    def set_device(self, device):
        self.test_device = device
    
    def set_model_name(self, new_name):
        self.name = new_name
        self.result_file_path = os.path.join(self.cwd_path, "results", "pruning", "{}.txt".format(self.name))
        self.saved_model_path = os.path.join(self.cwd_path, "saved_models", "pruning", "{}.ckpt".format(self.name))

    def forward(self, x):
        return self.model(x)   
    
    def configure_optimizers(self):
        if self.onecyclelr:
            optimizer = torch.optim.SGD(self.parameters(), lr=0.0125, momentum=0.9, weight_decay=2e-05)
            stepping_batches = self.trainer.estimated_stepping_batches
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.0125, pct_start=0.1, total_steps=stepping_batches)
            lr_schedulers = {"scheduler": scheduler, "interval": "step"}
        else:
            optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, min_lr=1e-05)
            lr_schedulers = {"scheduler": scheduler, "monitor": "Loss/train"}
        
        return [optimizer], [lr_schedulers]
    
    def get_model_flops(self):
        x = torch.rand(1, 3, 32, 32).to(self.test_device)
        flops = FlopCountAnalysis(self.model, x)
        return flops.total()
        
    def get_number_of_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def get_model_size_in_mb(self):
        return os.path.getsize(self.saved_model_path) / (1024*1024.0)
    
    def save_result_file(self, acc, loss, flops, parameters, model_size):
        with open(self.result_file_path, 'w') as file:
            file.write("{}\n\nAcc = {:.4f}\nLoss = {:.4f}\nFLOPs: {}\nParameters: {}\nSize: {:.4f} MB".format(
                    self.name,
                    acc,
                    loss,
                    flops,
                    parameters,
                    model_size
                ))    
    
    def calculate_acc_loss(self, batch):
        imgs, labels = batch
        preds = self.model(imgs)
        loss = self.loss_module(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()
   
        return acc, loss
    
    def log_acc_loss(self, acc, loss, name):
        self.log("Acc/{}".format(name), acc, on_step=False, on_epoch=True, logger=True)
        self.log("Loss/{}".format(name), loss, on_step=False, on_epoch=True, logger=True)       
                
    def training_step(self, batch, batch_idx):
        acc, loss = self.calculate_acc_loss(batch)
        self.log_acc_loss(acc, loss, "train")
        return loss
    
    def validation_step(self, batch, batch_idx):
        acc, loss = self.calculate_acc_loss(batch)
        self.log_acc_loss(acc, loss, "val")
           
    def test_step(self, batch, batch_idx):
        acc, loss = self.calculate_acc_loss(batch)
        return [acc, loss]
    
    def test_epoch_end(self, step_outputs):
        acc = torch.stack([x[0] for x in step_outputs]).mean().item()
        loss = torch.stack([x[1] for x in step_outputs]).mean().item()
        
        
        parameters = self.get_number_of_parameters()
        model_size = self.get_model_size_in_mb()

        if self._device != torch.device('cpu'):
            flops = self.get_model_flops()
            self.logger.experiment.add_scalar('flops', flops, 0)
        else:
            flops = '-'
        self.logger.experiment.add_scalar('parameters', parameters, 0)
        self.logger.experiment.add_scalar('model_size', model_size, 0)
        
        self.save_result_file(acc, loss, flops, parameters, model_size)
    
    def prepare_quantization(self):
        self.set_device(torch.device('cpu'))
        self.model.to(torch.device('cpu'))
        self.model.eval()
        quantization_config = torch.quantization.get_default_qconfig("qnnpack")
        self.model.qconfig = quantization_config

        self.model = torch.quantization.prepare(self.model)
    
    def quantize(self):
        self.model.eval()
        self.model = torch.quantization.convert(self.model)
