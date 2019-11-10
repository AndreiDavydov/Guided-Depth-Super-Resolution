import torch.nn as nn

def get_pad(pad_type, padding, value=0):
    assert len(padding) == 4, 'padding is incorrect!'
    if pad_type == 'reflect':
        return nn.ReflectionPad2d(padding)
    elif pad_type == 'constant':
        return nn.ConstantPad2d(padding, value=value)
    elif pad_type == 'replicate':
        return nn.ReplicationPad2d(padding)
    else:
        assert False, 'pad_type is incorrect!'

def get_conv(in_ch, out_ch, kernel_size, padding):
    return nn.Conv2d(in_ch, out_ch, 
            kernel_size=kernel_size, padding=padding)

def get_nnl(nnl_type):
    if nnl_type == 'relu':
        return nn.ReLU() # Tests have shown ReLU doesn't work at all! 
                         # It zeroes all the outputs after one epoch.
    elif nnl_type == 'prelu':
        return nn.PReLU() # Recommended type.
    else:
        assert False, 'Incorrect non-linearity is provided!'

def nnl_conv_pad_block(in_ch, out_ch, pad_type, kernel_size, nnl_type=None):
    padding = (kernel_size[0]//2, kernel_size[0]//2, 
               kernel_size[0]//2, kernel_size[0]//2) # padding is supposed to be square and odd!
    pad = get_pad(pad_type=pad_type, padding=padding)
    conv = get_conv(in_ch, out_ch, kernel_size=kernel_size, padding=(0,0))
    blocks = [pad, conv]
    if nnl_type is not None:
        nnl = get_nnl(nnl_type)
        blocks.append(nnl)
    return nn.Sequential(*blocks)

def pool_block(in_ch, out_ch, pad_type, kernel_size, nnl_type):
    block = nnl_conv_pad_block(in_ch, out_ch, pad_type, kernel_size, nnl_type)
    pool = nn.MaxPool2d((2,2))
    return nn.Sequential(*[block, pool])

def nnl_convT_block(in_ch, out_ch, nnl_type=None):
    convT = nn.ConvTranspose2d(in_ch, out_ch, 
                    stride=2, kernel_size=(4,4), padding=(1,1))
    blocks = [convT]
    if nnl_type is not None:
        nnl = get_nnl(nnl_type)
        blocks.append(nnl)
    return nn.Sequential(*blocks)