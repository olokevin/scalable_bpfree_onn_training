import numpy as np
import copy
import torch
from torch.utils.data import Dataset


def dataio_init_from_config(config, mode):
    device = config['basic'].get('device')

    return BurgersDataIO(device=device)


def to_device(data, device, dtype=torch.float32):
    if dtype == torch.float32:
        return torch.tensor(data, requires_grad=True).float().to(device)
    elif dtype == torch.float16:
        return torch.tensor(data, requires_grad=True).half().to(device)
    


class BurgersDataIO(Dataset):
    def __init__(self, num_interior=1200, num_boundary=100, device=None, dtype=torch.float32):
        super(BurgersDataIO, self).__init__()
        
        self.num_interior = num_interior
        self.num_boundary = num_boundary
        self.device = device
        self.dtype = dtype
        self.mode = 'train'

    def __len__(self):
        return 1

    def __getitem__(self, item):
        if self.mode == 'train':
            # Interior points
            x_interior = np.random.uniform(-1, 1, (self.num_interior, 1))
            t_interior = np.random.uniform(0, 1, (self.num_interior, 1))
            interior_points = np.hstack((x_interior,t_interior))

            # Initial condition points
            x_initial = np.random.uniform(-1, 1, (self.num_boundary, 1))
            t_initial = np.full((self.num_boundary, 1), 0)
            initial_points = np.hstack((x_initial, t_initial))

            # Boundary condition at x = -1
            x_boundary_min = np.full((self.num_boundary, 1), -1)
            t_boundary_min = np.random.uniform(0, 1, (self.num_boundary, 1))
            boundary_min_points = np.hstack((x_boundary_min, t_boundary_min))

            # Boundary condition at x = 1
            x_boundary_max = np.full((self.num_boundary, 1), 1)
            t_boundary_max = np.random.uniform(0, 1, (self.num_boundary, 1))
            boundary_max_points = np.hstack((x_boundary_max, t_boundary_max))

            all_points = np.vstack((interior_points, initial_points, boundary_min_points, boundary_max_points))
            
            return to_device(copy.deepcopy(all_points), self.device, dtype=self.dtype)
        
        elif self.mode == 'eval':
            # Create a grid for evaluation
            x = np.linspace(-1, 1, 101)
            t = np.linspace(0, 1, 11)
            X, T = np.meshgrid(x, t, indexing='ij')
            eval_points = np.stack((X.flatten(), T.flatten()), axis=-1)
            eval_tensor = to_device(copy.deepcopy(eval_points), self.device, dtype=self.dtype)
            
            # Calculate analytical solution
            u_test = np.loadtxt('Burgers_1D_Simulation_Data.csv',delimiter=',',skiprows=1)[:,1:].reshape(-1,1)
            
            return eval_tensor, u_test

    def train(self):
        self.mode = 'train'
        return next(iter(self))

    def eval(self):
        self.mode = 'eval'
        return next(iter(self))

# class BurgersDataIO(Dataset):
#     def __init__(self, num_interior=8192, num_boundary=2048, device=None, dtype=torch.float32):
#         super(BurgersDataIO, self).__init__()
        
#         self.num_interior = num_interior
#         self.num_boundary = num_boundary
#         self.device = device
#         self.dtype = dtype
#         self.mode = 'train'
        
        
#         # Interior points
#         x_interior = np.random.uniform(-1, 1, (self.num_interior, 1))
#         t_interior = np.random.uniform(0, 1, (self.num_interior, 1))
#         interior_points = np.hstack((x_interior,t_interior))

#         # Initial condition points
#         x_initial = np.random.uniform(-1, 1, (self.num_boundary, 1))
#         t_initial = np.full((self.num_boundary, 1), 0)
#         initial_points = np.hstack((x_initial, t_initial))

#         # Boundary condition at x = -1
#         x_boundary_min = np.full((self.num_boundary, 1), -1)
#         t_boundary_min = np.random.uniform(0, 1, (self.num_boundary, 1))
#         boundary_min_points = np.hstack((x_boundary_min, t_boundary_min))

#         # Boundary condition at x = 1
#         x_boundary_max = np.full((self.num_boundary, 1), 1)
#         t_boundary_max = np.random.uniform(0, 1, (self.num_boundary, 1))
#         boundary_max_points = np.hstack((x_boundary_max, t_boundary_max))

#         all_points = np.vstack((interior_points, initial_points, boundary_min_points, boundary_max_points))
#         self.all_points = to_device(copy.deepcopy(all_points), self.device, dtype=self.dtype)

#     def __len__(self):
#         return 1

#     def __getitem__(self, item):
#         if self.mode == 'train':
#             return self.all_points
        
#         elif self.mode == 'eval':
#             # Create a grid for evaluation
#             x = np.linspace(-1, 1, 101)
#             t = np.linspace(0, 1, 11)
#             X, T = np.meshgrid(x, t, indexing='ij')
#             eval_points = np.stack((X.flatten(), T.flatten()), axis=-1)
#             eval_tensor = to_device(copy.deepcopy(eval_points), self.device, dtype=self.dtype)
            
#             # Calculate analytical solution
#             u_test = np.loadtxt('Burgers_1D_Simulation_Data.csv',delimiter=',',skiprows=1)[:,1:].reshape(-1,1)
            
#             return eval_tensor, u_test

#     def train(self):
#         self.mode = 'train'
#         return next(iter(self))

#     def eval(self):
#         self.mode = 'eval'
#         return next(iter(self))
