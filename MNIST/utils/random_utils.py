import os
import torch
import numpy as np
import random
import time


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