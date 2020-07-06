### Welcome to SliceGAN ###
####### Steve Kench #######
from TrainTest import trainer
from Architect import Architect
from Pckgs import util

## make directory
Project_name = 'sep_type2' #Creates directory with output images
Project_dir = 'Seperator/'

## Data Processing
image_type = 'twophase' # threephase, twophase or colour
data_type = 'tif' # png, jpg, tif, array, array2D
data_path = ['Examples/img_stack_1200_2500_1200_high_reso_biphase_V2.tif'] # path to training data.
isotropic = True
Training = False # Run with False to show an image during training
Project_path = util.mkdr(Project_name, Project_dir, Training)

## Network Architectures
imsize, nz,  channels = 128, 32, 2
dk, gk = [4,4,4,4,4,4], [4,4,4,4,4,4]                                    # kernal sizes
ds, gs = [2,2,2,2,2,2], [2,2,2,2,2,2]                                    # strides
df, gf = [channels,32,64,128,256,256,1], [nz,512,256,128,64,32, channels]  # filter sizes for hidden layers
dp, gp = [3,2,2,2,2,2],[2,2,2,2,2,3]
##Create Networks
netD, netG = Architect(Project_path, Training, dk, ds, df,dp, gk ,gs, gf, gp)

if Training:
    data = trainer(Project_path, image_type, data_type, data_path, netD, netG, isotropic, channels, imsize, nz)

##Save tif/ show full volume
else:
    img, raw, netG = util.test_img(Project_path, image_type, netG(), nz, show = False, lf = 4)