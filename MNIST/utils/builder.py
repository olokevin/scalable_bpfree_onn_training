import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from collections import OrderedDict
from core.tensor_fwd_bwd.tensorized_linear import TensorizedLinear
from core.TFONet.modules import FCBlock, TensorizedFCBlock, ONN_FCBlock, TensorizedONN_FCBlock
from core.ZO_Estim import ZO_Estim_MC
from core.optimizer import FLOPSOptimizer

def build_model(config):
    if config.model.network == 'dnn':
        model = FCBlock(
            in_features     = config.model.in_features, 
            out_features    = config.model.out_features, 
            hidden_features = config.model.hidden_features,
            num_layers      = config.model.num_layers, 
            nonlinearity    = config.model.nonlinearity, 
            nl_last_layer   = config.model.nl_last_layer, 
            bias            = config.model.bias, 
            device          = config.GraSP.device, 
            dtype           = torch.float32
        )
    elif config.model.network == 'tensorized_dnn':
        model = TensorizedFCBlock(
            in_features     = config.model.in_features, 
            out_features    = config.model.out_features, 
            hidden_features = config.model.hidden_features,
            num_layers      = config.model.num_layers, 
            nonlinearity    = config.model.nonlinearity, 
            shape_list      = config.model.shape_list,
            order           = config.model.order, 
            min_dim         = config.model.min_dim,
            factorization   = config.model.factorization, 
            rank            = config.model.rank, 
            tensorize_first = config.model.tensorize_first,
            tensorize_last  = config.model.tensorize_last, 
            bias            = config.model.bias, 
            device          = config.GraSP.device, 
            dtype           = torch.float32
        )
    elif config.model.network == 'onn':
        model = ONN_FCBlock(
            in_features     = config.model.in_features, 
            out_features    = config.model.out_features, 
            hidden_features = config.model.hidden_features,
            num_layers      = config.model.num_layers, 
            nonlinearity    = config.model.nonlinearity, 
            bias            = config.model.bias, 
            device          = config.GraSP.device, 
            dtype           = torch.float32,
            
            in_bit          = config.quantize.input_bit,
            w_bit           = config.quantize.weight_bit,
            mode            = config.onn_model.mode,
            v_max           = config.quantize.v_max,
            v_pi            = config.quantize.v_pi,
            act_thres       = config.onn_model.act_thres,
            photodetect     = False
        )
    elif config.model.network == 'tonn':
        model = TensorizedONN_FCBlock(
            in_features     = config.model.in_features, 
            out_features    = config.model.out_features, 
            hidden_features = config.model.hidden_features,
            num_layers      = config.model.num_layers, 
            nonlinearity    = config.model.nonlinearity, 
            shape_list      = config.model.shape_list,
            order           = config.model.order, 
            min_dim         = config.model.min_dim,
            factorization   = config.model.factorization, 
            rank            = config.model.rank, 
            tensorize_first = config.model.tensorize_first,
            tensorize_last  = config.model.tensorize_last, 
            bias            = config.model.bias, 
            device          = config.GraSP.device, 
            dtype           = torch.float32,
            
            in_bit          = config.quantize.input_bit,
            w_bit           = config.quantize.weight_bit,
            mode            = config.onn_model.mode,
            v_max           = config.quantize.v_max,
            v_pi            = config.quantize.v_pi,
            act_thres       = config.onn_model.act_thres,
            photodetect     = False,
        )

        return model
    else:
        raise ValueError(f"Wrong network_name {config.model.network}")
  
    return model

def build_obj_fn_classifier(data, target, model, criterion):
    def _obj_fn():
        y = model(data)
        return y, criterion(y, target)
    
    return _obj_fn

def build_obj_fn_classifier_row(data, target, model, criterion):
    criterion = nn.CrossEntropyLoss(reduction='none')
    def _obj_fn(row=-1, get_batch_sz=False):
        if get_batch_sz == True:
            return data.size()[0]
        else:
            if row == -1:
                y = model(data)
                return y, criterion(y, target)
            else:
                y = model(data[row].unsqueeze(0)).squeeze()
                return y, criterion(y, target[row])
            
    return _obj_fn

def build_obj_fn(obj_fn_type, **kwargs):
    if obj_fn_type == 'classifier':
        obj_fn = build_obj_fn_classifier(**kwargs)
    elif obj_fn_type == 'classifier_row':
        obj_fn = build_obj_fn_classifier_row(**kwargs)
    else:
        return NotImplementedError
    return obj_fn

def build_ZO_Estim(config, model, obj_fn, named_masks):
    model.requires_grad_(False)
    if config.name == 'ZO_Estim_MC':
        ZO_Estim = ZO_Estim_MC(
            model = model, 
            obj_fn = obj_fn,

            sigma = config.sigma,
            n_sample  = config.n_sample,
            signSGD = config.signSGD,
            mask_method = config.mask_method,
            estimate_method = config.estimate_method,
            perturb_method = config.perturb_method,
            sample_method = config.sample_method,
            prior_method = config.prior_method
        )
        return ZO_Estim
    else:
        return NotImplementedError
    
    
    
def build_optimizer(config_optim, net, criterion, named_masks, learning_rate):
    
    if config_optim.name == 'sgd':
        net.requires_grad_(True)
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=float(config_optim.momentum))
    elif config_optim.name == 'adam':
        net.requires_grad_(True)
        if hasattr(config_optim, 'betas') == False:
            betas = (0.9, 0.98)
        else:
            betas = tuple(config_optim.betas)
        optimizer = optim.Adam(net.parameters(), betas=betas, eps=1e-06, lr = learning_rate)
        optimizer.param_groups[0]['capturable'] = True
    elif config_optim.name == "flops":
        optimizer = FLOPSOptimizer(
            net,
            lr=learning_rate,
            sigma=config_optim.sigma,
            n_sample=config_optim.n_sample,
            criterion=criterion,
        )
    else:
        raise ValueError(f"Wrong optimizer_name {config_optim.name}") 
    
    return optimizer

def build_scheduler(config_sched, optimizers, learning_rate):
    if config_sched.name == 'ExponentialLR':
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizers, gamma=config_sched.gamma)
        return lr_scheduler
    elif config_sched.name == 'steplr':
        epochs_til_decay = config_sched.epochs_til_decay
        gamma=config_sched.gamma
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizers, step_size=epochs_til_decay, gamma=gamma)
        return lr_scheduler
    elif config_sched.name == 'CosineAnnealingLR':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizers, T_max=int(config_sched.T_max), eta_min=float(config_sched.eta_min))
        return lr_scheduler
    else:
        raise ValueError(f"Wrong scheduler_name {config_sched.name}")