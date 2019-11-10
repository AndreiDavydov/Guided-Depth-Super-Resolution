from matplotlib.pyplot import imread
import numpy as np
from glob import glob
from os.path import join

import torch as th
from torch.utils.data import Dataset

import sys

'''
Currently this module holds Sintel dataset processor only. 
To use it for another dataset, specific classes must be accomplished. 
'''

# here must be a path to Sintel script 
# for "get_depth" function in the "sintel_io" module
path2SintelModule = './../DATASETS/Sintel/depth_training_20150305/sdk/python/'
sys.path.append(path2SintelModule)
import sintel_io

from utils.functional import get_training_sample, get_full_sample

class Sintel(Dataset):
    '''
    Allows to work with raw files, RGB and Depth maps. 
    '''
    def __init__(self, 
        path2images='./../DATASETS/Sintel/training_images/training/final/', 
        path2depths='./../DATASETS/Sintel/depth_training_20150305/training/depth/', 
        m=1, num_images=None, get_name=False, size=(400,400),\
        mode=None, do_transforms=True, crop_size=256, sample_mode='ordinary'):
        '''
        "mode" can be: "train" - returns three images, 
                       None - returns just numpy-like RGB and Depth images
        '''
        self.size=size
        self.m = m
        self.path2images = path2images
        self.path2depths = path2depths

        self.depth_max = 1e2

        self.image_paths = sorted(glob(path2images+'*/*.png'))
        self.depth_paths = sorted(glob(path2depths+'*/*.dpt'))
        self.get_name = get_name

        if not ((mode == 'train') or (mode == 'val')):
            raise Exception(
                'Incorrect mode type! Only "train" or "val" are available.')
        self.mode = mode
        self.sample_mode = sample_mode
        self.do_transforms = do_transforms
        self.crop_size = crop_size

        if num_images is not None:
            self.image_paths = self.image_paths[:num_images]
            self.depth_paths = self.depth_paths[:num_images]

    def __len__(self):
        return len(self.image_paths)

    def transforms(self, image, depth):

        assert self.crop_size < min(depth.shape), \
        'Inconsistent crop size, it is bigger than image dimensions!'
        y0 = np.random.randint(0,image.shape[0] - self.crop_size)
        x0 = np.random.randint(0,image.shape[1] - self.crop_size)

        image = image[y0:y0+self.crop_size, x0:x0+self.crop_size]
        depth = depth[y0:y0+self.crop_size, x0:x0+self.crop_size]
        return image, depth

    def __getitem__(self, idx):
        image = imread(self.image_paths[idx]).astype(np.float32)
        depth = sintel_io.depth_read(self.depth_paths[idx])

        image /= image.max()
        depth[depth>self.depth_max] = self.depth_max 

        file_name = self.image_paths[idx][len(self.path2images):-4] 
        file_name = file_name.replace('/', '_')

        if self.mode == 'train':
            if self.do_transforms:
                image, depth = self.transforms(image, depth)
            if self.sample_mode == 'ordinary':
                out = get_training_sample(image, depth, m=self.m)
                out = [th.FloatTensor(elem)[None,...] for elem in out]
            elif self.sample_mode == 'full':
                out = get_full_sample(image, depth)
        elif self.mode == 'val':
            if self.size is not None:
                image = image[:self.size[0],:self.size[1]]
                depth = depth[:self.size[0],:self.size[1]]
            out = [image, depth]

        if self.get_name:
            return out+[file_name]
        else:
            return out

class SintelCropped_hf(Dataset):
    '''
    Allows to construct necessary low/high-spectrum images.
    MSGNet training requires high-freq parts only. 
    '''
    def __init__(self, path2folder='./../DATASETS/Sintel/CUSTOM_TRAINING_m3/',\
        mode='train', num_images=None, get_name=False):

        self.m = int(path2folder[-2:][0])
        self.mode = mode
        self.path2folder = path2folder+mode+'/'
        self.path2D_hr_hf_est = self.path2folder+'D_hr_hf_est/'
        self.path2D_lr_hf     = self.path2folder+'D_lr_hf/'
        self.path2Y_hr_hf     = self.path2folder+'Y_hr_hf/'

        self.D_hr_hf_est_paths = sorted(glob(self.path2D_hr_hf_est+'*.pth'))
        self.D_lr_hf_paths     = sorted(glob(self.path2D_lr_hf    +'*.pth'))
        self.Y_hr_hf_paths     = sorted(glob(self.path2Y_hr_hf    +'*.pth'))

        self.get_name = get_name

        if num_images is not None:
            self.D_hr_hf_est_paths = self.D_hr_hf_est_paths[:num_images]
            self.D_lr_hf_paths     = self.D_lr_hf_paths    [:num_images]
            self.Y_hr_hf_paths     = self.Y_hr_hf_paths    [:num_images]

    def __len__(self):
        return len(self.D_hr_hf_est_paths)

    def __getitem__(self, idx):
        D_hr_hf_est = th.load(self.D_hr_hf_est_paths[idx])
        D_lr_hf     = th.load(self.D_lr_hf_paths[idx])
        Y_hr_hf     = th.load(self.Y_hr_hf_paths[idx])

        file_name = self.D_hr_hf_est_paths[idx][len(self.path2D_hr_hf_est):-4] 

        out = [D_hr_hf_est, D_lr_hf, Y_hr_hf]

        if self.get_name:
            return out+[file_name]
        else:
            return out


class Sintel_FullTraining(Dataset):
    '''
    Rewritten SintelCropped_hf. Now it allows to get any pack of spectra and scales 
    (based on "training_type" choice).
    '''
    def __init__(self, mode='train', num_images=None, 
        get_name=False, training_type='ordinary'):

        # here must be a path to folder with prepared dataset. Should be a parameter in this function.
        path2dataset='./../DATASETS/Sintel/all_scales_freqs_training_set/' 
        self.mode = mode
        self.path2mode_folder = path2dataset+mode+'/'
        self.paths2folders = sorted(glob(self.path2mode_folder+'*'))
        self.folders = [folder[len(self.path2mode_folder):] 
                            for folder in self.paths2folders]
        self.paths2files = {folder:sorted(glob(path2folder+'/*.pth')) 
                                for folder, path2folder in zip(self.folders, self.paths2folders)}

        self.get_name = get_name
        self.training_type = training_type 
        #Possible values: "full_set"|"ordinary"|"multiloss"|"full_freq"

        if self.training_type == 'ordinary': # as in paper, hf-to-hf training
            self.get_from_folders = ['Y_hf', 'Dx1_hf', 'Dx8_hf']
        elif self.training_type == 'multiloss': # hf-to-hfs training (at each scale of upsampling)
            self.get_from_folders = ['Y_hf', 'Dx1_hf', 'Dx2_hf', 'Dx4_hf', 'Dx8_hf']
        elif self.training_type == 'full_freq': # hf-to-(hf+lf) training, 
                                                # it was used to be able to clip network output by depth_max value.
            self.get_from_folders = ['Y_hf', 'Dx1_hf', 'Dx8_hf', 'Dx8_lf']
        elif self.training_type == 'full_set': # independent from training, just full pack of all possible images.
            self.get_from_folders = self.folders.copy()
        else:
            assert False, 'Incorrect type of training.'

        if num_images is not None:
            for folder in self.folders:
                self.paths2files[folder] = self.paths2files[folder][:num_images]

    def __len__(self):
        return len(self.paths2files['Y_hf'])

    def __getitem__(self, idx):
        out = {}

        for folder in self.get_from_folders:
            path2file = self.paths2files[folder][idx]
            out[folder] = th.load(path2file)

        file_name = path2file[len(self.path2mode_folder+'/'+folder):-4] 
        if self.get_name:
            return [out, file_name]
        else:
            return out