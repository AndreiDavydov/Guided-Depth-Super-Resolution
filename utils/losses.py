import torch as th
import numpy as np

'''
This file holds TV-based losses computation and overall criterion composition.
'''

def deriv_x(x, preserve_size=True):
    delta_x = x[...,1:] - x[...,:-1]
    if preserve_size:
        delta_x = th.nn.functional.pad(delta_x, (1,0,0,0), mode='constant')
    return delta_x

def deriv_y(x, preserve_size=True):
    delta_y = x[...,1:,:] - x[...,:-1,:]
    if preserve_size:
        delta_y = th.nn.functional.pad(delta_y, (0,0,1,0), mode='constant')
    return delta_y

def get_div1(x):
    delta_y, delta_x = deriv_y(x), deriv_x(x)
    return delta_y, delta_x

def get_div2(x):
    delta_y, delta_x = deriv_y(x), deriv_x(x)
    delta_yy = deriv_y(delta_y)
    delta_xx = deriv_x(delta_x)
    delta_xy = deriv_y(delta_x)
    return delta_yy, delta_xx, delta_xy

class TVLoss(th.nn.Module):
    '''
    first-order TV
    '''
    def __init__(self,TVLoss_weight=0, p=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight
        self.p = p

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

    def forward(self,x):
        batch_size = x.size()[0]
        y_size = self._tensor_size(x[:,:,1:,:])
        x_size = self._tensor_size(x[:,:,:,1:])
        delta_y, delta_x = th.abs(deriv_y(x)), th.abs(deriv_x(x))

        loss = th.pow(delta_y/y_size, self.p) + th.pow(delta_x/x_size, self.p)
        return self.TVLoss_weight*loss.sum()/batch_size

class TV2Loss(th.nn.Module):
    '''
    second-order TV
    '''
    def __init__(self,TV2Loss_weight=0, p=1):
        super(TV2Loss,self).__init__()
        self.TV2Loss_weight = TV2Loss_weight
        self.p = p

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

    def forward(self,x):
        batch_size = x.size()[0]
        yy_size = self._tensor_size(x[:,:,1:,:][:,:,1:,:])
        xx_size = self._tensor_size(x[:,:,:,1:][:,:,:,1:])
        xy_size = self._tensor_size(x[:,:,:,1:][:,:,1:,:])
        
        delta_yy, delta_xx, delta_xy = th.abs(deriv_y(deriv_y(x))),\
                                       th.abs(deriv_x(deriv_x(x))),\
                                       th.abs(deriv_y(deriv_x(x)))

        loss = th.pow(delta_yy/yy_size, self.p) + \
               th.pow(delta_xx/xx_size, self.p) + \
               th.pow(delta_xy/xy_size, self.p)
        return self.TV2Loss_weight*loss.sum()/batch_size

def RMSE(x,y):
    ''' Root-mean-squared error loss '''
    if isinstance(x, th.Tensor) and isinstance(y, th.Tensor):
        out = th.sqrt(th.nn.MSELoss()(x,y))
    elif isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
        out = np.sqrt(((x - y)**2).mean())
    else:
        raise Exception('What types do inputs have?')
        return
    return out

def compose_criterion(opt):
    '''
    Let's go through all loss parameters in ArgParser 
    and compose the LOSS function.
    
    opt - class ArgParser. For better understanding, look into arg_parser.py.
    '''

    def criterion(output, target):
        loss = 0
        if opt.L1:
            loss += th.nn.L1Loss()(output,target)
        if opt.L2:
            loss += th.nn.MSELoss()(output,target)
        if opt.TV1 > 0:
            loss += TVLoss(TVLoss_weight=opt.TV1, p=opt.TV1_p)(output)
        if opt.TV2:
            loss += TV2Loss(TV2Loss_weight=opt.TV2, p=opt.TV2_p)(output)
        return loss
    return criterion

