import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

'''
This file holds all routine for inputs/outputs preparation for MSGNet.

All input tensors are supposed to be numpy-like HxW(x3).
'''

def get_gaus_filt(sigma_filt=1, size=(10,10)): 
    # this function is used for FFT gaussian filtering.
    if isinstance(sigma_filt, tuple):
        sigma_y, sigma_x = sigma_filt[0], sigma_filt[1]
    else:
        sigma_y, sigma_x = sigma_filt, sigma_filt

    xx,yy = np.meshgrid(np.linspace(-size[1]//2,size[1]//2, size[1]),
                        np.linspace(-size[0]//2,size[0]//2, size[0]))
    gaus_filt = np.exp( -(xx**2)/2/(sigma_x**2) -(yy**2)/2/(sigma_y**2) )
    return gaus_filt

'''
Here come three functions, as denoted, for "training", "validation" and "full".
They have same attributes: "m", "mode", "sigma", "do_filtering".

"m" - the degree of upsampling/downsampling (s = 2^m - upscaling factor).

"mode" can be either "gaus" (for gaussian filtering) or "fft" for 
gaussian filtering in Fourier domain. 

"sigma" parameter is used for filtering only in time domain, 
it is a standard deviation of gaussian filter.

"do_filtering" is a flag of how low-res depth should be taken.
If False, then low-res version will be taken via NearestNeighbor, without pre-filtering. 
If True,  then same filter was used for pre-filtering as "mode" prescribes.

For depth processing, mode == gaus, sigma == 1, filtering == False 
were chosen as most justified.
'''

def get_training_sample(Y_hr, D_hr, m=1, mode='gaus', sigma=1, do_filtering=False):
    '''
    For given Y (RGB image, high-res) and D (depth, high-res) this function returns
    Y_hr_hf, D_lr_hf, D_hr_hf_estimate only.

    In other words, it returns inputs and ground-truth output.
    '''

    # input for Y-branch of the network
    Y_hr_hf = get_Y_hr_hf(Y_hr, mode=mode, sigma=sigma) 
    
    D_lr = get_lr(D_hr, m=m, mode=mode, sigma=sigma, 
                            do_filtering=do_filtering)
    # input for D-branch of the network
    D_lr_hf = get_hf(D_lr, mode=mode, sigma=sigma) 
    
    # target for loss calculation
    D_hr_hf_estimate = get_hr_hf_estimate(D_lr, D_hr, mode=mode, sigma=sigma) 
    
    return [Y_hr_hf, D_lr_hf, D_hr_hf_estimate]

def get_validation_sample(Y_hr, D, m=1, mode='gaus', 
                                sigma=1, do_filtering=False):
    '''
    For given Y (RGB image, high-res) and D (depth) this function returns
    Y_hr_hf, D_lr_hf, D_hr_lf_estimate only.

    In other words, it returns inputs and necessary part of output for full-spectrum inference.
    '''
    Y_hr_hf = get_Y_hr_hf(Y_hr, mode=mode, sigma=sigma) 
    D_lr = get_lr(D, m=m, mode=mode, sigma=sigma, do_filtering=do_filtering)\
                            if Y_hr_hf.shape == D.shape else D
    D_lr_hf = get_hf(D_lr, mode=mode, sigma=sigma) 
    D_hr_lf_estimate = get_hr_lf_estimate(D_lr, Y_hr.shape[:2], \
                                                mode=mode, sigma=sigma)
    return [Y_hr_hf, D_lr_hf, D_hr_lf_estimate]

def get_full_sample(Y, D): 
    '''
    For given Y (RGB image, high-res) and D (depth) this function returns
    ALL spectrum parts for each scale as dictionaries.  

    Comparing with previous functions, unnecessary attributes have been discarded.
    '''
    m, mode, sigma, do_filtering = 3, 'gaus', 1, False
    interp_filt = 'bilinear'
    resample = Image.BILINEAR

    Y_hf = get_Y_hr_hf(Y, mode=mode, sigma=sigma) 
    # prepare full depth at each dimension
    dict_Dx = {8:D}
    for m in [1,2,3]:
        Dxm = get_lr(D, m=m, 
                     mode=mode, sigma=sigma, 
                     do_filtering=do_filtering)
        dict_Dx[int(2**(3-m))] = Dxm
    # prepare low-freq at each scale by interpolation of lowest Depth (Dx1)
    # prepare high-freq as (full-low_freq)
    Dx1_lf = get_lf(dict_Dx[1], mode=mode, sigma=sigma)
    Dx1_lf_size = Dx1_lf.shape
    Dx1_lf = Image.fromarray(Dx1_lf)
    dict_Dx_lf, dict_Dx_hf = {}, {}
    for scale in [1,2,4,8]:
        size = (Dx1_lf_size[0]*scale, Dx1_lf_size[1]*scale)[::-1]
        Dxm_lf = np.array(Dx1_lf.resize(size=size, resample=resample))
        dict_Dx_lf[scale] = Dxm_lf
        dict_Dx_hf[scale] = dict_Dx[scale] - Dxm_lf

    return [Y, Y_hf, dict_Dx_lf, dict_Dx_hf]


def get_lf(x, mode='gaus', sigma=1):
    ''' For given numpy 1-channel image provides its high-freq part'''
    if mode == 'gaus':
        x_lf = gaussian_filter(x, sigma=sigma)
    elif mode == 'fft':
        x_fft = np.fft.fftshift(np.fft.fft2(x))
        size = x_fft.shape
        sigma_filt = (size[0]/6, size[1]/6)
        gaus_filt_lf = get_gaus_filt(sigma_filt=sigma_filt, size=size)
        x_fft_lf = gaus_filt_lf*x_fft
        x_lf = np.abs(np.fft.ifft2(np.fft.ifftshift(x_fft_lf)))
    return x_lf

def get_hf(x, mode='gaus', sigma=1):
    ''' For given numpy 1-channel image provides its high-freq part'''
    x_lf = get_lf(x, mode=mode, sigma=sigma)
    x_hf = x - x_lf
    return x_hf

def get_lr(x_hr, m=1, mode='gaus', sigma=1, do_filtering=False): 
    ''' For given numpy 1-channel image provides its low-res version'''
    x_hr = x_hr.astype(np.float32)
    if do_filtering:
        x_hr = get_lf(x_hr, mode=mode, sigma=sigma)
    scale = int(2**m)
    x_lr = x_hr[scale//2::scale, scale//2::scale]
    return x_lr

def get_Y_hr_hf(Y_hr, mode='gaus', sigma=1):
    ''' For given Y_hr RGB image provides its high-res version, high-freq ONLY!'''
    Y_hr = Y_hr.astype(np.float32)
    
    if len(Y_hr.shape) == 3: # Checking whether Y is given as R-G-B (3 ch-s). 
                             # Tranform it to grayscale version.
        # Maybe, taking luminance in YCbCr colorspace would be better solution...
        Y_hr = (Y_hr[...,0] + Y_hr[...,1] + Y_hr[...,2])/3 
    Y_hr_hf = get_hf(Y_hr, mode=mode, sigma=sigma) 
    return Y_hr_hf

def get_hr_lf_estimate(x_lr, x_hr_size, mode='gaus', sigma=1, 
                                    interp_filt='bilinear'):
    ''' 
        Returns low-freq of high-res version of depth (for output summation of spectra).
        "x_hr_size" must be tuple/list of length 2, HxW!
    '''
    x_hr_size = x_hr_size[::-1] # for PIL.Image resize (WxH)

    x_lr_lf = get_lf(x_lr, mode=mode, sigma=sigma)
    x_hr_lf_estimate = Image.fromarray(x_lr_lf)

    if interp_filt == 'bilinear':
        resample = Image.BILINEAR # by default the simplest one is chosen
    elif interp_filt == 'bicubic':
        resample = Image.BICUBIC
    elif interp_filt == 'nearest':
        resample = Image.NEAREST
    elif interp_filt == 'lanczos':
        resample = Image.LANCZOS
    else:
        assert False, 'Interpolation type is incorrect.'

    x_hr_lf_estimate = x_hr_lf_estimate.resize(size=x_hr_size, \
                                            resample=resample) 
    x_hr_lf_estimate = np.array(x_hr_lf_estimate)
    return x_hr_lf_estimate 

def get_hr_hf_estimate(x_lr, x_hr, mode='gaus', sigma=1):
    ''' Returns high-freq of high-res version of depth (for loss calculation).'''
    x_hr_lf_estimate = get_hr_lf_estimate(x_lr, x_hr.shape[:2], 
                                        mode=mode, sigma=sigma)
    x_hr_hf_estimate = x_hr - x_hr_lf_estimate 
    return x_hr_hf_estimate
