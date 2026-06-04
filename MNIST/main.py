import argparse
import json
import math
import os
import sys
sys.path.append(".")
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
import time
import copy
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils.common_utils import get_logger, makedirs, set_torch_deterministic
from utils.data_utils import get_dataloader
from utils.builder import build_model, build_optimizer, build_scheduler
from utils.yml_config import configs

from core.ZO_Estim import build_ZO_Estim, build_obj_fn
from core.optimizer import FLOPSOptimizer

def init_logger(config):
    # set logger
    path = os.path.dirname(os.path.abspath(__file__))
    logger = get_logger('log.log', logpath=config.summary_dir)
    # logger.info(dict(config))
    writer = SummaryWriter(config.summary_dir)
    return logger, writer

def prepare_logger(config_dir):
    name_append = '_bz{}_lr{}'.format(configs.GraSP.batch_size, configs.GraSP.learning_rate)
    if hasattr(configs, 'ZO_Estim'):
        name_append = configs.ZO_Estim.name + name_append
    else:
        name_append = 'FO' + name_append
    summn = os.path.join(
            "./runs",
            configs.GraSP.dataset,
            configs.model.network, 
            name_append, 
            time.strftime("%Y%m%d-%H%M%S")+'-'+str(os.getpid())
        )
    

    # summary_dir, ckpt_dir is path (with / in the end)
    configs.GraSP.summary_dir = os.path.join(summn, 'run/')
    configs.GraSP.checkpoint_dir = os.path.join(summn, 'run/')
    print("=> config.summary_dir:    %s" % configs.GraSP.summary_dir)
    # print("=> config.checkpoint_dir: %s" % configs.GraSP.checkpoint_dir)

    # save .yml to directory
    makedirs(configs.GraSP.summary_dir)
    makedirs(configs.GraSP.checkpoint_dir)
    shutil.copy(config_dir, configs.GraSP.summary_dir)
    
    logger, writer = init_logger(configs.GraSP)
    logger.info(os.getpid())
    logger.info(torch.initial_seed())

    return logger, writer

def set_nonlinearity_from_init(yml_config, model):
    # deterministic phase bias
    if(yml_config.noise.phase_bias):
        model.assign_random_phase_bias(random_state=int(yml_config.noise.random_state))
    # deterministic phase shifter gamma noise
    model.set_gamma_noise(float(yml_config.noise.gamma_noise_std),
                          random_state=int(yml_config.noise.random_state))
    # deterministic phase shifter crosstalk
    model.set_crosstalk_factor(float(yml_config.noise.crosstalk_factor))
    # deterministic phase quantization
    model.set_weight_bitwidth(int(yml_config.quantize.weight_bit))
    # enable/disable noisy identity
    model.set_noisy_identity(int(yml_config.sl.noisy_identity))

def main():
    # ================== .yml parser ========================== 
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', metavar='FILE', help='config file')
    args = parser.parse_args()

    configs.load(args.config, recursive=False)  

    if hasattr(configs.GraSP, 'seed'):
        seed = configs.GraSP.seed
    else:
        seed = 42
    set_torch_deterministic(seed)
    print(torch.initial_seed())

    logger, writer = prepare_logger(args.config)

    learning_rate = float(configs.GraSP.learning_rate)
    num_epochs = configs.GraSP.epoch

    # ====================================== Prepare model ======================================
    # has pretrained model
    if hasattr(configs, 'pretrained') and configs.pretrained.incre == True:
        model_state = torch.load(configs.pretrained.load_model_path)
        # model = model_state['model']
        model = build_model(configs)
        model.load_state_dict(model_state['model'])
        logger.info('Pre-trained model accuracy: %.4f ' % model_state['acc'])
    # from scratch
    else: 
        model = build_model(configs)
    logger.info("Num of params: {}".format(sum(p.numel() for p in model.parameters())))

    named_masks = None
    
    # ================= Setup nonlinearity =================
    if configs.model.network in ('onn', 'tonn'):
        if configs.onn_model.mode == 'phase':
            model.switch_mode_to("usv")
            model.sync_parameters(src="phase")
        if configs.model.mzi_noise == True:
            set_nonlinearity_from_init(configs, model)
    else:
        pass
      
    # ====================================== get dataloader ======================================
    trainloader, testloader = get_dataloader(configs.GraSP.dataset, configs.GraSP.batch_size, 256, 4, resize=configs.GraSP.resize)
    criterion = nn.CrossEntropyLoss(reduction='mean')
    optimizer = build_optimizer(configs.optimizer, model, criterion, named_masks, learning_rate)
    lr_scheduler = build_scheduler(configs.scheduler, optimizer, learning_rate)

    if hasattr(configs, 'ZO_Estim') and configs.ZO_Estim.en:
        obj_fn = None
        ZO_Estim = build_ZO_Estim(configs.ZO_Estim, model=model)
        logger.info('trainable params:')
        for param in ZO_Estim.splited_param_list:
            logger.info(f'{param.name}')
    else:
        ZO_Estim = None

    best_acc = 0
    best_epoch = 0
    ZO_forward_num = 0 
    num_iterations = 0
    total_num_forward = 0
    
    ##### Fix data
    if configs.GraSP.fix_data == True:
        fix_inputs, fix_targets = next(iter(trainloader))
    
    for epoch in range(num_epochs):
        # ====================================== Training =====================================
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        now_lr = optimizer.state_dict()['param_groups'][0]['lr']
        desc = ('[LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                (now_lr, 0, 0, correct, total))
        prog_bar = tqdm(enumerate(trainloader), total=len(trainloader), desc=desc, leave=True)

        for batch_idx, (inputs, targets) in prog_bar:
            ##### Fix data
            if configs.GraSP.fix_data == True:
                inputs = fix_inputs
                targets = fix_targets
            
            inputs, targets = inputs.to(configs.GraSP.device), targets.to(configs.GraSP.device)
            optimizer.zero_grad()

            if ZO_Estim is not None: 
                with torch.no_grad():
                    obj_fn = build_obj_fn(configs.ZO_Estim.obj_fn_type, data=inputs, target=targets, model=model, criterion=criterion)
                    ZO_Estim.update_obj_fn(obj_fn)
                    outputs, loss = obj_fn()
                    ZO_Estim.estimate_grad(old_loss=loss)
                ##### Update parameters or not    
                optimizer.step()
            else:
                if isinstance(optimizer, (FLOPSOptimizer,)):
                    with torch.no_grad():
                        outputs, loss = optimizer.step(inputs, targets)
                else:   
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
          
            ##### metrics
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            desc = ('[LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                    (now_lr, train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
            prog_bar.set_description(desc, refresh=True)

            num_iterations = num_iterations + 1
            # for batch ends

        train_loss = train_loss / (batch_idx + 1)
        train_acc  = 100. * correct / total
  
        logger.info('epoch[%d],\n \t lr: %.4f, train_loss: %.4f, train_acc: %.4f' % (epoch, now_lr, train_loss, train_acc))  
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/acc', train_acc, epoch)
        
        # ====================================== Validation =====================================
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        desc = ('Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss / (0 + 1), 0, correct, total))

        prog_bar = tqdm(enumerate(testloader), total=len(testloader), desc=desc, leave=True)
        with torch.no_grad():
            for batch_idx, (inputs, targets) in prog_bar:
                # inputs, targets = inputs.cuda(), targets.cuda()
                inputs, targets = inputs.to(configs.GraSP.device), targets.to(configs.GraSP.device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                desc = ('Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
                prog_bar.set_description(desc, refresh=True)

        test_loss = test_loss / (batch_idx + 1)
        test_acc = 100. * correct / total

        logger.info('\t test_loss: %.4f, test_acc: %.4f' % (test_loss, test_acc))

        writer.add_scalar('test/loss', test_loss, epoch)
        writer.add_scalar('test/acc', test_acc, epoch)
    
        lr_scheduler.step()

        # if ZO_Estim is not None:
        #     num_forward = ZO_Estim.get_forward_cnt() 
        #     writer.add_scalar('test/acc_forwards', test_acc, num_forward * configs.GraSP.batch_size)
        #     logger.info('\t num_forward * batch_size: %d' % (num_forward * configs.GraSP.batch_size))
        
        # ====================================== Save checkpoints =====================================
        if test_acc > best_acc:
            print('Saving..')
            state = {
                'model': model.state_dict(),
                'acc': test_acc,
                'epoch': epoch,
                'named_masks': named_masks,
            }
            if hasattr(configs.GraSP, 'best_path'):
                # Delete the file
                os.remove(configs.GraSP.best_path)

            configs.GraSP.best_path = os.path.join(configs.GraSP.checkpoint_dir, 'finetune_epoch-%d_test-%.2f_best.pth.tar' % (epoch, test_acc))
            torch.save(state, configs.GraSP.best_path)
            best_acc = test_acc
            best_epoch = epoch
        
        # for epoch ends


if __name__ == '__main__':
    main()