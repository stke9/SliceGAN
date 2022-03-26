### Welcome to SliceGAN ###
####### Steve Kench #######
'''
Use this file to define your settings for a training run, or
to generate a synthetic image using a trained generator.
'''

from slicegan import model, networks, util
import argparse
# Define project name
Project_name = 'NMC_exemplar_final'
# Specify project folder.
Project_dir = 'Trained_Generators/NMC'
# Run with False to show an image during or after training
parser = argparse.ArgumentParser()
parser.add_argument('training', type=int)
args = parser.parse_args()
Training = args.training
Project_path = util.mkdr(Project_name, Project_dir, Training)

## Data Processing
# Define image  type (colour, grayscale, three-phase or two-phase.
# n-phase materials must be segmented)
image_type = 'threephase'
# define data type (for colour/grayscale images, must be 'colour' / '
# greyscale. nphase can be, 'tif', 'png', 'jpg','array')
data_type = 'tif'
# Path to your data. One string for isotrpic, 3 for anisotropic
data_path = ['Examples/NMC.tif']

## Network Architectures
# Training image size, no. channels and scale factor vs raw data
img_size, img_channels, scale_factor = 64, 3,  1
# z vector depth
z_channels = 16
# Layers in G and D
lays = 6
# kernals for each layer
# dk, gk = [4]*lays, [4]*lays
# # strides
# ds, gs = [2]*lays, [2]*lays
# # no. filters
# df, gf = [img_channels,64,128,256,512,1], [z_channels,512,256,128,64,img_channels]
# # paddings
# dp, gp = [1,1,1,1,0],[2,2,2,2,3]

net_params = {

    "pth": util.mkdr(Project_name, Project_dir, Training),
    "Training": Training,
    "imtype": 'threephase',

    "dk" : [4]*lays,
    "gk" : [4]*lays,

    "ds": [2]*lays,
    "gs": [2]*lays,

    "df": [img_channels,64,128,256,512,1],
    "gf": [z_channels,512,256,128,64,img_channels],

    "dp": [1,1,1,1,0],
    "gp": [2,2,2,2,3],

    }

## Create Networks
netD, netG = networks.slicegan_nets(**net_params)

lz_calced = model.calc_lz(img_size, net_params["gk"], net_params["gs"], net_params["gs"])

# Train
if Training:
    train_params = {
        "pth": Project_path,
        "imtype": image_type,
        "datatype": data_type,
        "real_data": data_path,
        "Disc": netD,
        "Gen": netG,
        "nc": img_channels,
        "l": img_size,
        "nz": z_channels,
        "sf": scale_factor,
        "lz": lz_calced
    }

    model.train(**train_params)
else:
    test_params = {
        "pth": Project_path,
        "imtype": image_type,
        "Gen": netG(),
        "nz": z_channels,
        "lf": 4,
        "periodic": False
    }
    img, raw, netG = util.test_img(**test_params)
