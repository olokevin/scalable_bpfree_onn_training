import os
import time
import json
import logging

import torch
import numpy as np
import random

from pprint import pprint
from easydict import EasyDict as edict

def load_model(model, model_dir, epoch):
    checkpoint_dir = os.path.join(model_dir, 'checkpoints', 'model_epoch_{}.pth'.format(epoch))
    checkpoint = torch.load(checkpoint_dir)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model


def to_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        return tensor
    else:
        raise TypeError('Unknown type of input, expected torch.Tensor or '
                        'np.ndarray, but got {}'.format(type(tensor)))


def to_device(data, device, requires_grad=False):
    return torch.tensor(data, requires_grad=requires_grad).float().to(device)

def get_learning_rate(optimizer):
    return optimizer.param_groups[0]["lr"]

def get_random_state():
    return np.random.get_state()[1][0]

def set_torch_deterministic(random_state: int = 0) -> None:
    random_state = int(random_state) % (2 ** 32)
    # random_state = int(random_state)
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(random_state)
    random.seed(random_state)

def set_torch_stochastic():
    seed = int(time.time() * 1000) % (2 ** 32)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = False
        torch.cuda.manual_seed_all(seed)

def get_logger(name, logpath, filepath='', package_files=[],
               displaying=True, saving=True):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    log_path = logpath + name
    makedirs(log_path)
    if saving:
        info_file_handler = logging.FileHandler(log_path)
        info_file_handler.setLevel(logging.INFO)
        logger.addHandler(info_file_handler)
    # logger.info(filepath)
    # with open(filepath, 'r') as f:
    #     logger.info(f.read())

    # for f in package_files:
    #     logger.info(f)
    #     with open(f, 'r') as package_f:
    #         logger.info(package_f.read())
    # if displaying:
    #     console_handler = logging.StreamHandler()
    #     console_handler.setLevel(logging.INFO)
    #     logger.addHandler(console_handler)

    return logger


def makedirs(filename):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))


def str_to_list(src, delimiter, converter):
    """Conver a string to list.
    """
    src_split = src.split(delimiter)
    res = [converter(_) for _ in src_split]
    return res


def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file:
    :return: config(namespace) or config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)
    config = edict(config_dict)

    return config, config_dict


def process_config(json_file, runs=None):
    """Process a json file into a config file.
    Where we can access the value using .xxx
    Note: we will need to create a similar directory as the config file.
    """
    config, _ = get_config_from_json(json_file)
    paths = json_file.split('/')[1:-1]

    summn = [config.exp_name]
    chekn = [config.exp_name]
    if runs is not None:
        summn.append('run_%s' % runs)
        chekn.append('run_%s' % runs)
    summn.append("summary/")
    chekn.append("checkpoint/")
    summary_dir = ["./runs/pruning"] + paths + summn
    ckpt_dir = ["./runs/pruning"] + paths + chekn
    config.summary_dir = os.path.join(*summary_dir)
    config.checkpoint_dir = os.path.join(*ckpt_dir)
    print("=> config.summary_dir:    %s" % config.summary_dir)
    print("=> config.checkpoint_dir: %s" % config.checkpoint_dir)
    return config


def try_contiguous(x):
    if not x.is_contiguous():
        x = x.contiguous()

    return x


def try_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return x


def tensor_to_list(tensor):
    if len(tensor.shape) == 1:
        return [tensor[_].item() for _ in range(tensor.shape[0])]
    else:
        return [tensor_to_list(tensor[_]) for _ in range(tensor.shape[0])]


# =====================================================
# For learning rate schedule
# =====================================================
class StairCaseLRScheduler(object):
    def __init__(self, start_at, interval, decay_rate):
        self.start_at = start_at
        self.interval = interval
        self.decay_rate = decay_rate

    def __call__(self, optimizer, iteration):
        start_at = self.start_at
        interval = self.interval
        decay_rate = self.decay_rate
        if (start_at >= 0) \
                and (iteration >= start_at) \
                and (iteration + 1) % interval == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= decay_rate
                print('[%d]Decay lr to %f' % (iteration, param_group['lr']))

    @staticmethod
    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
            return lr


class PresetLRScheduler(object):
    """Using a manually designed learning rate schedule rules.
    """
    def __init__(self, decay_schedule):
        # decay_schedule is a dictionary
        # which is for specifying iteration -> lr
        self.decay_schedule = decay_schedule
        print('=> Using a preset learning rate schedule:')
        pprint(decay_schedule)
        self.for_once = True

    def __call__(self, optimizer, iteration):
        for param_group in optimizer.param_groups:
            lr = self.decay_schedule.get(iteration, param_group['lr'])
            param_group['lr'] = lr

    @staticmethod
    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
            return lr


# =======================================================
# For math computation
# =======================================================
def prod(l):
    val = 1
    if isinstance(l, list):
        for v in l:
            val *= v
    else:
        val = val * l

    return val