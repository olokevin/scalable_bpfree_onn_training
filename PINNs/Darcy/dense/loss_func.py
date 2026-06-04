import torch
from core.TFONet import gradients
import numpy as np
import os

def gaussian_augment(x:torch.Tensor, std, N_sample):

    # sample_x.shape: (N_sample*batch_size, 2)
    sample_x = torch.cat(N_sample*[x])
    e = torch.normal(mean=0, std=std, size=sample_x.shape, device=sample_x.device)
    return sample_x+e, e


def direction_vector(X:torch.Tensor,h,ith_dim):
    '''
    h: hyperparameter in FD method
    '''
    e = torch.zeros(X.shape,device=X.device)
    e[:,ith_dim:ith_dim+1] = 1
    return h*e

def gradients_auto_diff(X,f):

    u_pred = f(X)
    u_g = gradients(u_pred, X)[0]
    u_x, u_y = u_g[:, :1], u_g[:, 1:]
    u_xx, u_yy = gradients(u_x, X)[0][:, :1], gradients(u_y, X)[0][:, 1:]
    
    return u_x, u_y, u_xx, u_yy

def gradients_cd2(X,f,h):

    u = f(X)
    e_x = direction_vector(X,h,0)
    u_x = (f(X+e_x)-f(X-e_x)) / (2*h)
    u_xx = (f(X+e_x)+f(X-e_x)-2*u) / (h**2)
    e_y = direction_vector(X,h,1)
    u_y = (f(X+e_y)-f(X-e_y)) / (2*h)
    u_yy = (f(X+e_y)+f(X-e_y)-2*u) / (h**2)
  
    return u_x, u_y, u_xx, u_yy


def gradients_stein(X,f,sigma,N_sample):

    batch_size = X.shape[0]
    sample_X, e_X = gaussian_augment(X, sigma, N_sample)
    sample_X_plus = sample_X
    sample_X_minus = sample_X - 2*e_X

    sample_u_plus = f(sample_X_plus).reshape(N_sample,batch_size,1)
    sample_u_minus = f(sample_X_minus).reshape(N_sample,batch_size,1)

    e_X = e_X.reshape(N_sample,batch_size,2)
    
    u_X = ((sample_u_plus-sample_u_minus)*e_X)/(2*sigma*sigma)
    u_X = torch.mean(u_X, dim=0)
    u_x, u_y = u_X[:,0:1], u_X[:,1:2]
    u_XX = (e_X**2-sigma**2)*(sample_u_plus+sample_u_minus-2*f(X).reshape(1,batch_size,1))/(2*(sigma**4))
    u_XX = torch.mean(u_XX, dim=0)
    u_xx, u_yy = u_XX[:,0:1], u_XX[:,1:2]
    return u_x, u_y, u_xx, u_yy


def gradients_sparse_gird(X,f,sigma,level,rule):

    if rule == 'GQN':
        data_path = os.path.join(os.getcwd(), 'GQN_d2', 'GQN_d2_l' + str(level) + '.asc')
        data = np.loadtxt(data_path,delimiter=',')
        delta = data[:,0:-1]
        weight = data[:,-1:]
 
    elif rule == 'KPN':
        data_path = os.path.join(os.getcwd(), 'KPN_d2', 'KPN_d2_l' + str(level) + '.asc')
        data = np.loadtxt(data_path,delimiter=',')
        delta = data[:,0:-1]
        weight = data[:,-1:]
    
    delta = torch.tensor(delta,device=X.device,dtype=X.dtype) * sigma
    weight = torch.tensor(weight,device=X.device,dtype=X.dtype)
    
    n_node = delta.shape[0]
    batch_size = X.shape[0]
    dim = X.shape[1]

    delta_expand = delta.view(n_node,1,dim).repeat(1,batch_size,1)
    weight_expand = weight.view(n_node,1,1).repeat(1,batch_size,1)

    X_expand = X.view(1,batch_size,dim).repeat(n_node,1,1)
    X_plus = (X_expand + delta_expand).reshape(n_node*batch_size,dim)
    X_minus = (X_expand - delta_expand).reshape(n_node*batch_size,dim)
    u_plus = f(X_plus).reshape(n_node,batch_size,1)
    u_minus = f(X_minus).reshape(n_node,batch_size,1)
    
    u_X = delta_expand*(u_plus-u_minus)/(2*(sigma**2))
    u_X = torch.sum(weight_expand*u_X,dim=0)
    u_x,u_y = u_X[:,0:1],u_X[:,1:2]
    
    u_XX = (delta_expand**2 - sigma*sigma)*(u_plus + u_minus - 2*f(X).reshape(1,batch_size,1))/(2*(sigma**4))
    u_XX = torch.sum(weight_expand*u_XX,dim=0)
    u_xx, u_yy = u_XX[:,0:1], u_XX[:,1:2]
    
    return u_x, u_y, u_xx, u_yy



def loss_func_init_from_config(config):


    def loss_func(model, dataset, inputs=None, return_loss_reduction='mean'):
        if inputs is not None:
            x = inputs
        else:
            x = dataset.train()
        c = dataset.c
        dcdx = dataset.dcdx
        dcdy = dataset.dcdy
        
        f_model = lambda X: X[:, :1] * X[:, 1:] * (X[:, :1] - 1) * (X[:, 1:] - 1) * model(X)
        
        

        if config['loss_func'].get('gradient_func') == 'gradients_cd2':
            u_x, u_y, u_xx, u_yy = gradients_cd2(x,f_model,h=1e-2)
        elif config['loss_func'].get('gradient_func') == 'gradients_stein':
            u_x, u_y, u_xx, u_yy = gradients_stein(x,f_model,sigma=0.001,N_sample=2048)
        elif config['loss_func'].get('gradient_func') == 'gradients_sparse_grid':
            u_x, u_y, u_xx, u_yy = gradients_sparse_gird(x,f_model,sigma=0.001,level=3,rule='GQN')
        elif config['loss_func'].get('gradient_func') == 'gradients_auto_diff':
            u_x, u_y, u_xx, u_yy = gradients_auto_diff(x,f_model)
        else:
            raise NotImplementedError
        
        # loss = 1 + dcdx*u_x + c*u_xx + dcdy*u_y + c*u_yy
        loss = 1 + c*u_xx + c*u_yy
        return (loss ** 2).mean()

    return loss_func