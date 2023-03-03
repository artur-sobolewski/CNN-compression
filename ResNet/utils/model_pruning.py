import os
import torch
import numpy as np

from pytorch_lightning.core.module import LightningModule
from fvcore.nn import FlopCountAnalysis

import torch.nn.utils.prune as prune

class PruningModel(LightningModule):
    def __init__(self, name, model, pruning_amount=0.2):
        super().__init__()
        self.name = name
        self.model = model
        self.amount = pruning_amount
        self.loss_module = torch.nn.CrossEntropyLoss()
        
        self.result_file_path = "./results/pruning/{}.txt".format(self.name)
        self.saved_model_path = "./saved_models/pruning/{}.ckpt".format(self.name)
    
    def set_model_name(self, new_name):
        self.name = new_name
        self.result_file_path = "./results/pruning/{}.txt".format(self.name)
        self.saved_model_path = "./saved_models/pruning/{}.ckpt".format(self.name)

    def forward(self, x):
        return self.model(x)   
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-04, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, min_lr=1e-05)
        lr_schedulers = {"scheduler": scheduler, "monitor": "Loss/train"}
        return [optimizer], [lr_schedulers]          
    
    def get_model_flops(self):
        x = torch.rand(1, 3, 32, 32).cuda()
        flops = FlopCountAnalysis(self.model, x)
        return flops.total()
        
    def get_number_of_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def get_model_size_in_mb(self):
        return os.path.getsize(self.saved_model_path) / (1024*1024.0)
    
    def save_result_file(self, acc, loss, flops, parameters, not_zeroed, sparsity, model_size):
        with open(self.result_file_path, 'w') as file:
            file.write("{}\n\nAcc = {:.4f}\nLoss = {:.4f}\nFLOPs: {}\nParameters: {}\nNot zeroed parameters: {}\nsparsity: {}\nSize: {:.4f} MB".format(
                    self.name,
                    acc,
                    loss,
                    flops,
                    parameters,
                    not_zeroed,
                    sparsity,
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
    
    def log_sparsity(self, spar, not_zeroed_params):
        self.log("Not zeroed params", not_zeroed_params, on_step=False, on_epoch=True, logger=True)
        self.log("sparsity", spar, on_step=False, on_epoch=True, logger=True)


    def on_fit_start(self) -> None:
        self.prune_model(amount=self.amount)
        return super().on_fit_start()
                
    def training_step(self, batch, batch_idx):
        acc, loss = self.calculate_acc_loss(batch)
        self.log_acc_loss(acc, loss, "train")
        return loss
    
    def validation_step(self, batch, batch_idx):
        acc, loss = self.calculate_acc_loss(batch)
        self.log_acc_loss(acc, loss, "val")
        num_zeros, num_elements, sparsity = self.measure_global_sparsity(
            weight = True, bias = False,
            conv2d_use_mask = True,
            linear_use_mask = False)
        self.log_sparsity(num_elements - num_zeros, sparsity * 100)

           
    def test_step(self, batch, batch_idx):
        acc, loss = self.calculate_acc_loss(batch)
        return [acc, loss]
    
    def test_epoch_end(self, step_outputs):
        acc = torch.stack([x[0] for x in step_outputs]).mean().item()
        loss = torch.stack([x[1] for x in step_outputs]).mean().item()
        
        flops = self.get_model_flops()
        parameters = self.get_number_of_parameters()
        model_size = self.get_model_size_in_mb()

        num_zeros, num_elements, sparsity = self.measure_global_sparsity(
            weight = True, bias = False,
            conv2d_use_mask = True,
            linear_use_mask = False)

        self.logger.experiment.add_scalar('flops', flops, 0)
        self.logger.experiment.add_scalar('parameters', parameters, 0)
        self.logger.experiment.add_scalar('model_size', model_size, 0)
        
        self.save_result_file(acc, loss, flops, parameters, num_elements - num_zeros, sparsity,  model_size)
    
    def measure_module_sparsity(self, module, weight=True, bias=False, use_mask=False):

        num_zeros = 0
        num_elements = 0

        if use_mask == True:
            for buffer_name, buffer in module.named_buffers():
                if "weight_mask" in buffer_name and weight == True:
                    num_zeros += torch.sum(buffer == 0).item()
                    num_elements += buffer.nelement()
                if "bias_mask" in buffer_name and bias == True:
                    num_zeros += torch.sum(buffer == 0).item()
                    num_elements += buffer.nelement()
        else:
            
            for param_name, param in module.named_parameters():
                if "weight" in param_name and weight == True:
                    num_zeros += torch.sum(param == 0).item()
                    num_elements += param.nelement()
                if "bias" in param_name and bias == True:
                    num_zeros += torch.sum(param == 0).item()
                    num_elements += param.nelement()

        return num_zeros, num_elements
        
    def measure_global_sparsity(
        self, weight = True,
        bias = False, conv2d_use_mask = False,
        linear_use_mask = False):

        num_zeros = 0
        num_elements = 0

        for module_name, module in self.model.named_modules():
            

            if isinstance(module, torch.nn.Conv2d):
                module_num_zeros, module_num_elements = self.measure_module_sparsity(
                    module, weight=weight, bias=bias, use_mask=conv2d_use_mask)
                num_zeros += module_num_zeros
                num_elements += module_num_elements

            elif isinstance(module, torch.nn.Linear):

                module_num_zeros, module_num_elements = self.measure_module_sparsity(
                    module, weight=weight, bias=bias, use_mask=linear_use_mask)
                num_zeros += module_num_zeros
                num_elements += module_num_elements

        if num_elements == 0:
            sparsity = 0
        else:
            sparsity = num_zeros / num_elements

        return num_zeros, num_elements, sparsity

    def get_parameter_to_prune(self):
        parameter_to_prune = [
            (module, "weight") for module in filter(lambda m: type(m) == torch.nn.Conv2d, self.model.modules())
        ]
        return parameter_to_prune

    def prune_model(self, amount):
        prune.global_unstructured(
            self.get_parameter_to_prune(),
            pruning_method=prune.L1Unstructured,
            amount=amount,
        )
    
    def remove_parameters(self):

        for module_name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                try:
                    prune.remove(module, "weight")
                except:
                    pass
                try:
                    prune.remove(module, "bias")
                except:
                    pass
            elif isinstance(module, torch.nn.Linear):
                try:
                    prune.remove(module, "weight")
                except:
                    pass
                try:
                    prune.remove(module, "bias")
                except:
                    pass
