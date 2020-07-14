### Welcome to SliceGAN ###
####### Steve Kench #######

from SliceGAN_util import *

## make directory
Project_name = 'nf_exploration' #Creates directory with output images
Project_dir = 'NMC/'

## Data Processing
image_type = 'threephase' # threephase, twophase or colour
data_type = 'tif' # png, jpg, tif, array, array2D
data_path = ['Examples/NMC.tif'] # path to training data.
isotropic = False
Training = True # Run with False to show an image during training
Project_path = mkdr(Project_name, Project_dir, Training)

## Network Architectures
imsize, nz,  channels, sf = 64, 16, 3,1
dk, gk = [4,4,4,4,4], [4,4,4,4,4]                                    # kernal sizes
ds, gs = [2,2,2,2,2], [2,2,2,2,2]                                    # strides
df, gf = [channels,16,32,64,128,1], [nz,128,64,32,16,channels]  # filter sizes for hidden layers
dp, gp = [3,2,2,2,2],[2,2,2,2,3]
##Create Networks
netD, netG = Architect(Project_path, Training, dk, ds, df,dp, gk ,gs, gf, gp)
for ngf in [1,2,4]:
    Project_name = 'filter_explo_' + str(ngf) + '_active_layers'  # Creates directory with output images
    Project_path = mkdr(Project_name , Project_dir, Training)

    data = trainer(Project_path, image_type, data_type, data_path, netD, netG, isotropic, channels, imsize, nz, sf)
# if Training:
#     data = trainer(Project_path, image_type, data_type, data_path, netD, netG, isotropic, channels, imsize, nz)
#
# ##Save tif/ show full volume
# else:
#     img, raw, netG = test_img(Project_path, image_type, netG(), nz, show = False, lf = 4)