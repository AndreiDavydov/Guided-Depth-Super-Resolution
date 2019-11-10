import matplotlib.pyplot as plt
import torch as th
import numpy as np

from glob import glob
from tqdm import tqdm
from time import sleep
import os

from IPython.display import clear_output 
# it is needed only to refresh the losses plots by "flag_while", unnecessary.


def show(images, titles=None, cmaps=None, crops=None,
    figsize=None, fontsize=None, set_title=None):
    '''
    Makes the pleasant figure of images, optionally with custom titles, colormaps, crops. 

    If the number of images to show is more than one, then "images" is supposed to be a list.
    The titles must be either a string (1 image for input) or a list of strings.
    '''
    if not isinstance(images, list):
        images = [images]      

    images_recon = []
    for ind, image in enumerate(images):
        if len(image.shape) == 4:
            image = image[0]

        if not isinstance(image, np.ndarray): # assimed to be torch.Tensor  
            if image.is_cuda:
                image = image.cpu()
            image = image.detach().numpy()

        if image.shape[0] == 1:
            image = image[0] 

        if crops is not None:
        # assumed to be only from left-right corner
        # and only being only squared
            crop = crops[ind] if isinstance(crops, list) else crops
            image = image[..., :crop,:crop]
              
        images_recon.append(image)
    images = images_recon

    if fontsize is None:
        fontsize = 15

    num_images = len(images)

    if figsize is None:
        figsize = (8+num_images,8*num_images)
        if num_images == 3:
            figsize = (15,5)
        if num_images == 4:
            figsize = (20,5)
        
    fig, ax = plt.subplots(1, num_images, figsize=figsize)
    fig.patch.set_facecolor('white')

    if num_images == 1:
        ax = [ax]

    if cmaps is None:
        cmaps = ['depth']*num_images # Maybe, use 'gray' as default?

    elif isinstance(cmaps, str):
        cmaps = [cmaps]

    for i, (img, cmap) in enumerate(zip(images, cmaps)):
        if cmap == 'depth':
            cmap = 'jet'
        ax[i].imshow(img, cmap=cmap)
        ax[i].set_axis_off()

    # Now only titles have to be distributed.
    if titles is not None:

        if titles == 'shapes':
            titles = [str(img.shape) for img in images]

        elif isinstance(titles, str): # Title for one image
            ax[0].set_title(titles)

        if isinstance(titles, list):
            for i, title in enumerate(titles):
                ax[i].set_title(title, fontsize=fontsize)

            if len(titles) < num_images:
                for i in range(len(titles), num_images):
                    ax[i].set_title('', fontsize=fontsize)

    if set_title is not None:
        fig.suptitle(set_title, fontsize=fontsize)
    # fig.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()
    return


def show_losses(exp_name='EXP', flag_while=False, log=False, loglog=False,\
                path2folder='./saved_models/MSGNet/',\
                last_epochs=5, only_val=False):
    '''
    It is used for losses trends visulisation. 
    The usage example is provided in the model_test.ipynb.
    '''
    
    path2folder += exp_name+'/'

    fontsize = 20
    while True:
        fig, ax = plt.subplots(1, 2, figsize=(15,5))
        train = np.load(path2folder+'train_losses.npy')
        val   = np.load(path2folder+'val_losses.npy')
        num_epochs = np.load(path2folder+'last_saved_epoch.npy')

        train_x = np.linspace(0, num_epochs, num=len(train))
        val_x   = np.linspace(0, num_epochs, num=len(val))
        
        train_x_last = int(len(train)/num_epochs*(num_epochs-last_epochs-1))
        val_x_last   = int(len(val)  /num_epochs*(num_epochs-last_epochs-1))
        
        if loglog:
            ax[0].loglog(train_x, train)
            ax[0].loglog(val_x, val)

            if not only_val:
                ax[1].loglog(train_x[train_x_last:], train[train_x_last:])
            ax[1].loglog(val_x[val_x_last:], val[val_x_last:])

        if log:
            ax[0].semilogy(train_x, train)
            ax[0].semilogy(val_x, val)

            if not only_val:
                ax[1].semilogy(train_x[train_x_last:], train[train_x_last:])
            ax[1].semilogy(val_x[val_x_last:], val[val_x_last:])

        else:
            ax[0].plot(train_x, train)
            ax[0].plot(val_x, val)

            if not only_val:
                ax[1].plot(train_x[train_x_last:], train[train_x_last:])
            ax[1].plot(val_x[val_x_last:], val[val_x_last:])
            
        # show only last last_epochs epochs
        ax[1].set_xlim((np.maximum(num_epochs-last_epochs, 0), num_epochs))

        if not only_val and len(val) > 0:
            min_y = np.minimum(train[train_x_last:].min(), val[val_x_last:].min())*0.9
            max_y = np.maximum(train[train_x_last:].max(), val[val_x_last:].max())*1.1
            ax[1].set_ylim(min_y, max_y)

        if min(train) > 0:
            min_val = min(val) if len(val)>0 else 0
            ax[0].set_title('min train=  {:3.3e} | min val= {:3.3e}'\
                    .format(min(train), min_val), fontsize=fontsize)
            last_val = val[-1] if len(val)>0 else 0
            ax[1].set_title('last train= {:3.3e} | last val= {:3.3e}'\
                    .format(train[-1], last_val), fontsize=fontsize)
        fig.tight_layout()
        plt.show()

        params = np.load(path2folder+'exp_params.npy').item()
        print('num images train = ', params['num_images_train'])
        print('num images val = ', params['num_images_val'])
        print('train size = {}, val size = {}'\
                .format(params['train_batchsize'], params['val_batchsize']))
        
        if flag_while:
            sleep(2)
            clear_output() 
        else:
            break


def save_infer_imgs(exp_name, net, rgb, depth, 
            mode='gaus', sigma=1, do_filtering=False,
            path2net_folder='./saved_models/MSGNet/', crop=None):
    '''
    Runs given [rgb, depth] through each saved model state 
    and puts resultive images in the folder.
    '''

    path2folder = path2net_folder+exp_name+'/state_'
    state_paths = sorted(glob(path2folder+'*.pth'))
    state_names = [state_path[len(path2folder):-4] for state_path in state_paths]

    if not os.path.isdir(path2net_folder+exp_name+'/InferImgs_'+exp_name):
        os.mkdir(path2net_folder+exp_name+'/InferImgs_'+exp_name)

    for state_name, state_path in tqdm(zip(state_names, state_paths)):
        state = th.load(state_path, map_location=th.device('cpu'))
        net.load_state_dict(state['model_state_dict'])
        res = net.do_inference(rgb, depth, mode=mode, sigma=sigma, 
                                    do_filtering=do_filtering)
        res[res < depth.min()] = depth.min()
        res[res > depth.max()] = depth.max()
        if crop is not None:
            res = res[:crop, :crop]
        plt.imsave(path2net_folder+exp_name+'/InferImgs_'+exp_name+'/'+state_name, 
                    res, cmap='jet')
        