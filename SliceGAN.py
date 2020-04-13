#### Welcome to SliceGAN ###
####### Steve Kench ########

from Train import trainer
from Architect import Architect
import util

## Data Processing
Project_name = 'TEst' #Creates directory with output images
image_type = 'threephase' # threephase, twophase or colour
data_type = 'tif' # png, jpg, tif, array, 2Darray
data_path = ['Examples/img_stack_1000_high_reso_biphase.tif'] # path to training data.
isotropic = False

## Network Architectures
imsize, channels = 64, 3

dl, gl = [imsize,32,16,8,4,1], [1,4,8,16,32,imsize]                  # disc & gen layer sizes
dk, gk = [4,4,4,4,4], [4,4,4,4,4]                                    # kernal sizes
ds, gs = [2,2,2,2,1], [1,2,2,2,2]                                    # strides
df, gf = [channels,128,256,256,512,1], [64,512,256,128,64, channels] # filter sizes for hidden layers

##Create Networks
netD, netG = Architect(dl, dk, ds, df, gl, gk ,gs, gf)

##Train
Training = True # Run with False to show an image during training

if Training:
    trainer(Project_name,image_type,data_type,data_path, netD, netG, isotropic)

##Save tif/ show full volume
else:
    img = util.test_img(Project_name,image_type,netG(),show = True)