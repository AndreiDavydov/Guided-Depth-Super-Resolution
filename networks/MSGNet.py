import torch as th
from torch import nn
from utils.functional import get_validation_sample
import networks.layers as layers

class MSGNet(nn.Module): 
    '''
    Class for MSGNet network architecture. 
    Based on "https://github.com/twhui/MSG-Net" project.

    In a few words: the network learns to upsample 
    high frequencies of low-res depth because low frequencies
    are supposed to be same in high- and low- resolution versions.

    It leads to stable, fast training due to sparse nature of the data.

    We improve it with:
        custom paddings in convolutions ("reflect" by default);
        custom number of channels after each block;

    params:
    m (int) - the degree of upsampling (s = 2^m - upscaling factor)

    Note: kernels are supposed to be square and odd!
    '''
    def __init__(self, m=1, depth_max=1e2, **kwargs):
        super(MSGNet, self).__init__()

        self.m = m
        self.scale = 2**m

        self.nnl_type = kwargs['nnl_type'] \
            if 'nnl_type' in kwargs.keys() else 'prelu'
        self.pad_type = kwargs['pad_type'] \
            if 'pad_type' in kwargs.keys() else 'reflect'
        self.kernel_size = kwargs['kernel_size'] \
            if 'kernel_size' in kwargs.keys() else (5,5)
        self.num_channels = kwargs['num_channels'] \
            if 'num_channels' in kwargs.keys() else 32
        self.num_channels_y1 = kwargs['num_channels_y1'] \
            if 'num_channels_y1' in kwargs.keys() else 49 
        self.num_channels_d1 = kwargs['num_channels_d1'] \
            if 'num_channels_d1' in kwargs.keys() else 64
        self.num_channelsT = kwargs['num_channelsT'] \
            if 'num_channelsT' in kwargs.keys() else self.num_channels
        self.training_type = kwargs['training_type'] \
            if 'training_type' in kwargs.keys() else 'ordinary' 

        assert self.training_type == 'ordinary' or \
               self.training_type == 'full_freq' or \
               self.training_type == 'multiloss' or \
               self.training_type == 'full_set', 'Training type is incorrect!'
            # for better understanding what each training_type means
            # look into training_procedure.py, "compute_loss" function.

        if self.training_type == 'multiloss':
            self.each_scale_outs = []
            self.num_channelsT = 1
            # output at each scale must be 1-channel! 
            # (to compute loss with corresponding freq from dataset)

        self.depth_min = 0.        # such restrictions should be used
        self.depth_max = depth_max # explicitly for training stabilization.
                
        self.Yinit()
        self.Dinit()

    def Yinit(self): # RGB-image downsampling cascade initialization
        self.Yblocks = nn.ModuleList() 
        Y1_block = layers.nnl_conv_pad_block(1, self.num_channels_y1, 
            self.pad_type, kernel_size=(7,7), nnl_type=self.nnl_type) # 1st conv 
                                                                      # (look into original paper)

        next_block = layers.nnl_conv_pad_block(self.num_channels_y1, self.num_channels, 
            self.pad_type, kernel_size=self.kernel_size, nnl_type=self.nnl_type) 

        self.Yblocks.append(nn.Sequential(*[Y1_block, next_block]))

        for i in range(self.m-1):
            self.Yblocks.append(
                    layers.pool_block(self.num_channels, self.num_channels, 
                        self.pad_type, kernel_size=self.kernel_size, 
                        nnl_type=self.nnl_type)) 

    def Dinit(self): # depth-array upsampling cascade initialization
        self.Dblocks = nn.ModuleList()
        self.Dblocks.append(
                    layers.nnl_conv_pad_block(1, self.num_channels_d1, 
                    self.pad_type, kernel_size=self.kernel_size, 
                    nnl_type=self.nnl_type))

        last_num_channels = self.num_channels_d1
        for i in range(self.m):
            self.Dblocks.append(
                layers.nnl_convT_block(last_num_channels, 
                    self.num_channelsT, nnl_type=self.nnl_type))

            last_num_channels = self.num_channels

            #############################################################
            #                                                           #
            # here must be a fusion with Y-branch in "forward" function #
            #                                                           #
            #############################################################

            self.Dblocks.append(
                nn.Sequential(
                    *[layers.nnl_conv_pad_block(
                            self.num_channels+self.num_channelsT, 
                            self.num_channels, 
                            self.pad_type, kernel_size=self.kernel_size, 
                            nnl_type=self.nnl_type),
                      layers.nnl_conv_pad_block(
                            self.num_channels, self.num_channels, 
                            self.pad_type, kernel_size=self.kernel_size, 
                            nnl_type=self.nnl_type)]))

        # it is easier to keep indices of concatenation during initialization
        self.cat_indices = list(range(len(self.Dblocks)))[1:][::2] 


        # additional convolutions (last blocks)
        self.Dblocks.append(
            nn.Sequential(
                *[layers.nnl_conv_pad_block(
                        self.num_channels, self.num_channels, 
                        self.pad_type, kernel_size=self.kernel_size, 
                        nnl_type=self.nnl_type),
                  layers.nnl_conv_pad_block(
                        self.num_channels, 1, 
                        self.pad_type, kernel_size=self.kernel_size, 
                        nnl_type=None) # <- this is the last conv => no nnl
                        ])) 

    def Yforward(self, Y_hr_hf):
        '''forward function for Y-branch'''
        out = Y_hr_hf
        outs = [] # list of all outputs from Y-branch
        for block in self.Yblocks:
            out = block(out)
            outs.append(out)
        return outs # from finest to coarsest

    def Dforward(self, D_lr_hf, Youts):
        '''forward function for D-branch'''
        out = D_lr_hf
        self.each_scale_outs = []
        for i, block in enumerate(self.Dblocks):
            out = block(out)
            if i in self.cat_indices:
                if self.training_type == 'multiloss' and i != self.cat_indices[-1]: 
                    # One should check i for "not last i", 
                    # because last upscale layer goes to the network output.
                    self.each_scale_outs.append(out)
                out = th.cat((out, Youts.pop()), dim=1)
        return out

    def forward(self, Y_hr_hf, D_lr_hf):
        '''
        Shape of Y must be in "scale" times higher than D's shape. 
        This "forward" is used only in train mode.
        To do fair inference from original D_lr (low-res) and Y_hr (high-res) 
        use "do_inference" function.

        arguments:
        Y_hr_hf - high-freq part of high-res RGB (grayscale) image
                  Y_hr is supposed to be 1-channel, 
                  it can be a luminance channel in YCbCr color space or simple average of R,G and B.

        D_lr_hf - high-freq part of low-res depth image
        '''
        Youts = self.Yforward(Y_hr_hf) 
        out = self.Dforward(D_lr_hf, Youts)
        if th.isnan(out).sum() > 0:
            assert False, 'NaNs detected!'
        return out


    def do_inference(self, Y_hr, D, mode='gaus', sigma=1, do_filtering=False):
        '''
        Performs fair inference (during test) from Y_hr and D.
        Currently it works FOR 1 IMAGE ONLY!

        If sizes of D and Y_hr coincide, then test setting is implied,
        low-res version of depth will be prepared automatically.

        arguments:
        Y_hr - high-res grayscale image
        D - high-res (or low-res) depth image

        Both tensors are thought to be NUMPY-like!
        '''     
        Y_hr_hf, D_lr_hf, D_hr_lf_estimate = get_validation_sample(
                                            Y_hr, D, m=self.m, 
                                            mode=mode, sigma=sigma, 
                                            do_filtering=do_filtering)

        Y_hr_hf = th.FloatTensor(Y_hr_hf)[None, None, ...]
        D_lr_hf = th.FloatTensor(D_lr_hf)[None, None, ...]
        D_hr_hf_estimate = self.forward(Y_hr_hf, D_lr_hf).detach().numpy()[0,0]
        res = D_hr_hf_estimate + D_hr_lf_estimate
        return res   

    def __repr__(self): 
        print(self.__class__.__name__ + '(m = ' + str(self.m) + ')')
        print('\n Y-branch ({} block(s)):'.format(len(self.Yblocks)))
        for block in self.Yblocks:
            print('\t', block)

        print('\n D-branch ({} block(s)):'.format(len(self.Dblocks)))
        for block in self.Dblocks:
            print('\t', block)
        return ''





