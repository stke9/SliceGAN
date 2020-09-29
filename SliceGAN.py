### Welcome to SliceGAN ###
####### Steve Kench #######

from SliceGAN_util import *

## make directory
Project_name = 'NMC_exemplar_64_64' #Define project name
Project_dir = 'NMC/' #Create/specify a project directory for output images

## Data Processing
image_type = 'threephase' # threephase, twophase or colour
data_type = 'png' # png, jpg, tif, array, array2D
data_path = ['Examples/Scott_NMC_pp.png'] # path to training data.
isotropic = False
Training = False # Run with False to show an image during training
Project_path = mkdr(Project_name, Project_dir, Training)

## Network Architectures
imsize, nz,  channels, sf = 64, 16, 3,3
lays = 5
dk, gk = [4]*lays, [4]*lays                                    # kernal sizes
ds, gs = [2]*lays, [2]*lays                                    # strides
df, gf = [channels,64,128,256,512,1], [nz,256,256,128,64,channels]  # filter sizes for hidden layers
dp, gp = [1,1,1,1,0],[2,2,2,2,0]
##Create Networks
netD, netG = Architect(Project_path, Training, dk, ds, df,dp, gk ,gs, gf, gp)

if Training:
    data = trainer(Project_path, image_type, data_type, data_path, netD, netG, isotropic, channels, imsize, nz, sf)
else:
    img, raw, netG = test_img(Project_path, image_type, netG(), nz, show=False, lf=4)