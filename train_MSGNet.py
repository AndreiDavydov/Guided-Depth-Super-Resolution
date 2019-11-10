import numpy as np
import torch as th

from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR

from torch.nn import MSELoss, L1Loss

import os

from training_procedure import training_procedure, load_model
from networks.MSGNet import MSGNet

from utils.dataset_loader import SintelCropped_hf, Sintel_FullTraining
from utils.arg_parser import parse_args_joint_train as parse_args, \
                                get_exp_name, log_criterion
from utils.losses import compose_criterion


'''
This is the executable (via the terminal) file, 
it runs all the training with model loading/saving, data processing, training procedure logging.
'''

# First of all, all parameters of training must be collected. 
# For clarifications, it is elaborated in arg_parser.py file.
opt = parse_args()

# Let's fix anything we can to make experiments reproducible. 
th.manual_seed(1234)
th.cuda.manual_seed(1234)

# It is assumed by default that all the training is run via GPU. 
# Unfortunately, to change training processor to CPU, all the code must be rewritten
# (In fact, only ".cuda()" phrases should be discarded).
gpu_id = opt.gpu_id # 0, by default
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_id)

def initialize_training(opt):
    '''
    Network, Optimizer (maybe, with Scheduler) and Criterion initialization. 
    '''
    model = MSGNet(m=opt.m, nnl_type=opt.nnl_type, 
                training_type  =opt.training_type,
                num_channels   =opt.num_channels,
                num_channels_y1=opt.num_channels_y1,
                num_channels_d1=opt.num_channels_d1,
                num_channelsT  =opt.num_channelsT,
                ).cuda()

    lr0 = float(opt.lr0)
    optimizer = Adam(model.parameters(), lr=lr0)
    milestones = list(map(int, opt.milestones.strip('[]').split(','))) \
                if opt.milestones != '' else []
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1) # it was not used at all.
    criterion = compose_criterion(opt)

    return model, lr0, optimizer, milestones, scheduler, criterion

def initialize_network(opt, model, optimizer, scheduler, save_path):
    '''
    Loads the model, prepares (creates or loads) losses arrays.
    Creates a folder for experiment if there is no one.

    "start_epoch" will load the model saved at this epoch. 
    '''
    start_epoch =  opt.start_epoch

    if not os.path.isdir(save_path):
        os.mkdir(save_path)
        train_losses = []
        val_losses = []
        np.save(save_path+'train_losses.npy', train_losses)
        np.save(save_path+'val_losses.npy', val_losses)
        np.save(save_path+'training_time.npy', 0)
        start_epoch = 1
    else:
        if start_epoch is None:
            train_losses = []
            val_losses = []
            np.save(save_path+'train_losses.npy', train_losses)
            np.save(save_path+'val_losses.npy', val_losses)
            start_epoch = 1
        else:
            last_saved_epoch = int(np.load(save_path+'last_saved_epoch.npy'))
            if start_epoch > last_saved_epoch:
                start_epoch = last_saved_epoch

            train_losses = list(np.load(save_path+'train_losses.npy'))
            val_losses   = list(np.load(save_path+'val_losses.npy'))
            
            train_div, train_mod = opt.num_images_train//opt.train_batchsize,\
                                   opt.num_images_train%opt.train_batchsize
            val_div, val_mod     = opt.num_images_val//opt.val_batchsize,\
                                   opt.num_images_val%opt.val_batchsize  
                                                      
            train_num_batches = train_div if train_mod == 0 else train_div + 1
            val_num_batches   = val_div   if val_mod == 0 else val_div + 1

            np.save(save_path+'train_losses.npy', 
                train_losses[:train_num_batches])
            np.save(save_path+'val_losses.npy',   
                val_losses[:val_num_batches])

            model, optimizer, scheduler = load_model(start_epoch,
                model, optimizer, scheduler, save_path)


def initialize_datasets(opt):
    '''
    Initializes data processing. Returns data loaders.

    To use the dataset, another from Sintel, new functions must be written.
    '''
    th.manual_seed(0)

    set_train = Sintel_FullTraining(mode='train', 
                                    num_images=opt.num_images_train,
                                    training_type=opt.training_type)

    train_loader = DataLoader(set_train, batch_size=opt.train_batchsize,
        shuffle=True, num_workers=5) 

    set_val = Sintel_FullTraining(mode='val', 
                                    num_images=opt.num_images_val,
                                    training_type='ordinary')

    val_loader = DataLoader(set_val, batch_size=opt.val_batchsize, 
        shuffle=False, num_workers=5)

    return train_loader, val_loader

#######################################################################
#######################################################################
#######################################################################

if __name__ == '__main__':
    model, lr0, optimizer, milestones, scheduler, criterion = \
                                initialize_training(opt)
    
    exp_name = get_exp_name(opt)

    save_path = './saved_models/MSGNet/'+exp_name+'/'
    start_epoch = 1 if opt.start_epoch is None else opt.start_epoch

    initialize_network(opt, model, optimizer, scheduler, save_path)

    train_div, train_mod = opt.num_images_train//opt.train_batchsize,\
                           opt.num_images_train%opt.train_batchsize
    val_div, val_mod     = opt.num_images_val//opt.val_batchsize,\
                           opt.num_images_val%opt.val_batchsize  
                                              
    train_num_batches = train_div if train_mod == 0 else train_div + 1
    val_num_batches   = val_div   if val_mod == 0 else val_div + 1

    np.save(save_path+'exp_params.npy', {
        'num_images_train':opt.num_images_train, 
        'num_images_val':opt.num_images_val,
        'train_batchsize':opt.train_batchsize,
        'val_batchsize':opt.val_batchsize,
        'train_num_batches':train_num_batches,
        'val_num_batches':val_num_batches,
        'save_path':save_path,
        'lr0':lr0, 'milestones':milestones,
        'num_epochs':opt.num_epochs,
        'save_each_epoch':opt.save_each_epoch,
        'm':opt.m,
        'num_channels'   :opt.num_channels,
        'num_channels_y1':opt.num_channels_y1,
        'num_channels_d1':opt.num_channels_d1,
        'num_channelsT'  :opt.num_channelsT,
        'training_type':opt.training_type,
        'nnl_type':opt.nnl_type,
        'pad_type':opt.pad_type,
        'criterion':log_criterion(opt)})

    train_loader, val_loader = initialize_datasets(opt)

    # Runs the training...
    training_procedure(start_epoch, opt.num_epochs, model, optimizer, scheduler, 
                        criterion, train_loader, val_loader, 
                        save_path, opt.save_each_epoch, opt.no_val)