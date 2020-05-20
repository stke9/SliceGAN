#### Welcome to SliceGAN ###
####### Steve Kench ########

from TrainTest import trainer
from Architect import Architect
import util

## Data Processing
Project_name = 'Sep_im128' #Creates directory with output images
image_type = 'threephase' # threephase, twophase or colour
data_type = 'array2D' # png, jpg, tif, array, array2D
data_path = ['Seg_sep.png'] # path to training data.
isotropic = True

## Network Architectures
imsize, channels = 64, 3

dl, gl = [imsize,32,16,8,4,1], [1,4,8,16,32, imsize]                  # disc & gen layer sizes
dk, gk = [4,4,4,4,4], [4,4,4,4,4]                                    # kernal sizes
ds, gs = [2,2,2,2,1], [1,2,2,2,2]                                    # strides
df, gf = [channels,64,128,256,256,1], [64,256,256,128,64, channels] # filter sizes for hidden layers
##Create Networks
netD, netG = Architect(dl, dk, ds, df, gl, gk ,gs, gf)

##Train
Training = False # Run with False to show an image during training

if Training:
    trainer(Project_name,image_type,data_type,data_path, netD, netG, isotropic)

##Save tif/ show full volume
else:
    img, raw = util.test_img(Project_name,image_type,netG(),show = False , lf = 1)