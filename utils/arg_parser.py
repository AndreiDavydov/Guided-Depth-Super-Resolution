import argparse

'''
This module provides Parser description, as it can be executed from terminal.
'''

def parse_args_joint_train():
    parser = argparse.ArgumentParser(description='Training of MSGNet.')

    parser.add_argument('--m', type=int, default=1, help='# of scalings, scale == 2^m.')
    parser.add_argument('--train_batchsize', type=int, default=50, help='training batch size.')
    parser.add_argument('--val_batchsize', type=int, default=50, help='testing batch size.')
    parser.add_argument('--num_epochs', type=int, default=2000, help='number of epochs to train for.')
    parser.add_argument('--lr0', type=str, default='1e-3', help='learning rate in Adam optimizer')
    parser.add_argument('--gpu_id', type=int, default=0, help='which GPU to use?')
    parser.add_argument('--num_images_train', type=int, default=1000, help='# of images to train on.')
    parser.add_argument('--num_images_val', type=int, default=200, help='# of images to validate on.')
    parser.add_argument('--num_channels', type=int, default=32, help='# of channels in each inside block.')
    parser.add_argument('--num_channelsT', type=int, default=32, help='# of channels after each upscale layer.')
    parser.add_argument('--num_channels_y1', type=int, default=49, help='# of channels in the first block of Y.')
    parser.add_argument('--num_channels_d1', type=int, default=64, help='# of channels in the first block of D.')
    parser.add_argument('--start_epoch', type=int, default=None, help='index of epoch where to resume training from.')
    parser.add_argument('--save_each_epoch', type=int, default=50, help='With what period (in epochs) to save models')
    parser.add_argument('--exp_name', type=str, default=None, help='model short type name, additional folder division. ')
    parser.add_argument('--milestones', type=str, default='', help='List of milestone points, Ex: "[1,2,3,4,5]". ')
    parser.add_argument('--nnl_type', type=str, default='prelu', help='Either "prelu" or "relu" is used as non-linearity.')
    parser.add_argument('--pad_type', type=str, default='reflect', help='Padding type. Possible: "reflect"|"circular"|"constant"')
    parser.add_argument('--no_val', action='store_true', help='Do validation or not. (not by default)')

    parser.add_argument('--training_type', type=str, default='ordinary', help='Training type. Possible: "ordinary"|"multiloss"|"full_freq"')
    
    parser.add_argument('--L1', action='store_true', help='Include L1 in loss (Default: False)')
    parser.add_argument('--L2', action='store_true', help='Include L2 in loss (Default: False)')
    parser.add_argument('--TV1', type=float, default=0, help='weight for TV component in loss.')
    parser.add_argument('--TV2', type=float, default=0, help='weight for 2nd order TV component in loss.')
    parser.add_argument('--TV1_p', type=float, default=1, help='power of norm in TV loss (1st order).')
    parser.add_argument('--TV2_p', type=float, default=1, help='power of norm in TV loss (2nd order).')

    opt = parser.parse_args()
    return opt

def get_exp_name(opt):
    ''' 
    Structure of exp_name: 
        "m_nnl_ch_imgs_loss"
    Structure of loss:
        "Likelihood_Regularization_paramReg" 
    '''
    if opt.L1:
        loss_appendix = 'L1'
    elif opt.L2:
        loss_appendix = 'L2'
    else:
        loss_appendix = 'NO'
    if opt.TV1 > 0:
        loss_appendix += '+{:1.1E}*TV1'.format(opt.TV1)
    if opt.TV1 > 0 and opt.TV1_p > 0:
        loss_appendix += '_p{}'.format(int(opt.TV1_p))
    if opt.TV2 > 0:
        loss_appendix += '+{:1.1E}*TV2'.format(opt.TV2)
    if opt.TV2 > 0 and opt.TV2_p > 0:
        loss_appendix += '_p{}'.format(int(opt.TV2_p))

        # opt.nnl_type+\
        # '_ch'+str(opt.num_channels)+\
    if opt.exp_name is None:
        exp_name = 'm'+str(opt.m)+\
        '_imgs'+str(opt.num_images_train)+\
        '_lr0:'+str(opt.lr0)+\
        '_loss:'+loss_appendix+\
        '_'+opt.training_type
    else:
        exp_name = opt.exp_name
    return exp_name

def log_criterion(opt):
    criterion_params = {}
    criterion_params['L1'] = opt.L1
    criterion_params['L2'] = opt.L2
    criterion_params['TV1'] = {'weight':opt.TV1, 'p':opt.TV1_p}
    criterion_params['TV2'] = {'weight':opt.TV2, 'p':opt.TV2_p}
    return criterion_params