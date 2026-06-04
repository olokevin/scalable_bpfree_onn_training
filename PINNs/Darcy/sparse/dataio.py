import numpy as np
import copy
import torch
from torch.utils.data import Dataset
import scipy.io

def dataio_init_from_config(config, mode):
    device = config['basic'].get('device')

    return DarcyDataIO(device=device)


def to_device(data, device, dtype=torch.float32):
    if dtype == torch.float32:
        return torch.tensor(data, requires_grad=True).float().to(device)
    elif dtype == torch.float16:
        return torch.tensor(data, requires_grad=True).half().to(device)


class DarcyDataIO(Dataset):

    def __init__(self, num_interior=1200, device=None, dtype=torch.float32):
        super(DarcyDataIO, self).__init__()
        
        data = scipy.io.loadmat('piececonst_r241_N1024_smooth2.mat')
        self.num_interior = num_interior

        x = np.linspace(0, 1, 241)
        y = np.linspace(0, 1, 241)
        xx, yy = np.meshgrid(x, y, indexing='ij')
        self.x_train = np.stack((xx.flatten(), yy.flatten()), axis=-1)
        self.x_test = self.x_train
       
        self.u_test = data['sol'][0].reshape(-1,1)

        self.mode = 'train'
        self.device = device
        self.dtype = dtype
        self.c = data["coeff"][0].reshape(-1,1)
        self.dcdx = data["Kcoeff_y"][0].reshape(-1,1)
        self.dcdy =  data["Kcoeff_x"][0].reshape(-1,1)
        self.c = to_device(self.c, self.device, dtype=self.dtype)
        self.dcdx = to_device(self.dcdx, self.device, dtype=self.dtype)
        self.dcdy = to_device(self.dcdy, self.device, dtype=self.dtype)

    def __len__(self):
        return 1

    def __getitem__(self, item):
        if self.mode == 'train':
            idx = np.random.choice(241*241,(self.num_interior,),replace=False)
            c = self.c[idx,:]
            dcdx = self.dcdx[idx,:]
            dcdy = self.dcdy[idx,:]
            return c, dcdx, dcdy, to_device(copy.deepcopy(self.x_train[idx,:]), self.device, dtype=self.dtype)
        elif self.mode == 'eval':
            return to_device(copy.deepcopy(self.x_test), self.device, dtype=self.dtype), self.u_test

    def train(self):
        self.mode = 'train'
        return next(iter(self))

    def eval(self):
        self.mode = 'eval'
        return next(iter(self))