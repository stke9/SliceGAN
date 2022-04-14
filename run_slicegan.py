### Welcome to SliceGAN ###
####### Steve Kench #######
'''
Use this file to define your settings for a training run, or
to generate a synthetic image using a trained generator.
'''
import os

from matplotlib import use
from slicegan import model, networks, util, Circularity
import argparse
# Define project name
# Project_name = 'Hyperparameter_tuning_binary'
Project_name = 'CNet_merged_master_final_10_epochs_preprocessed'

# Specify project folder.
Project_dir = 'Trained_Generators'
# Run with False to show an image during or after training
parser = argparse.ArgumentParser()

# 0 Eveluation

# Run with False to show an image during or after training
parser = argparse.ArgumentParser()

# 0 Evaluation
# 1 Training
parser.add_argument('training', type=int)

# 0 for no CircNet
# 1 for CircNet WITHOUT training
# 2 for CircNet WITH training
parser.add_argument("use_Circ", type=int)
parser.add_argument("noise_type", type=str, nargs="?", default="normal")

args = parser.parse_args()
Training = args.training
use_Circ = args.use_Circ

## Data Processing
# Define image  type (colour, grayscale, three-phase or two-phase.
# n-phase materials must be segmented)
image_type = 'twophase'
# define data type (for colour/grayscale images, must be 'colour' / '
# greyscale. nphase can be, 'tif', 'png', 'jpg','array')
data_type = 'tif'
# Path to your data. One string for isotrpic, 3 for anisotropic
data_path = ['Examples/3D_data_binary.tif']

## Network Architectures
# Training image size, no. channels and scale factor vs raw data
img_size, img_channels, scale_factor = 64, 2,  1
# z vector depth
z_channels = 16
# Layers in G and D
lays = 6

# Type of noise distribution
noise_type = "normal"


beta1_values = [0, .2, .5, .8, .9]
beta2_values = [.1, .3, .5, .7, .9]
test_list = [0, .5]
# index 0-4: beta1, 5-9 beta2
betas = [.7, .8, 1]
betas = [0]
for index, beta in enumerate(betas):
    beta1 = 0
    beta2 = 0

    # if index < 5:
    #     beta1 = beta
    #     beta2 = .9
    # if index < 2:
    #     beta1 = 0
    #     beta2 = beta
    #
    # else:
    beta1 = 0.0
    beta2 = 0.9

    Project_name_beta = Project_name + '_beta1_' + str(beta1) + '_beta2_' + str(beta2)
    project_path = util.mkdr(Project_dir, Project_name_beta, Training)
    print('index: ', index)

    net_params = {

        "pth": project_path,
        "Training": Training,
        "imtype": image_type,

        # kernals for each layer
        "dk" : [4]*lays,
        "gk" : [4]*lays,

        # strides
        "ds": [2]*lays,
        "gs": [2]*lays,

        # no. filters
        "df": [img_channels,64,128,256,512,1],
        "gf": [z_channels,512,256,128,64,img_channels],

        # paddings
        "dp": [1,1,1,1,0],
        "gp": [2,2,2,2,3],
        }

    ## Create Networks

    # Test efficacy of Blob Detector on Real Image

    # imm = util.testCircleDetector(data_path)
    # ts = Circularity.numCircles(imm)
    circleNet = None
    if use_Circ != 0:    ## Create and Train CircleNet
        ## Create Path
        print(project_path)

        # Circle_path = util.mkdr(Project_name, Circle_dir, W_dir)
        # blob_path = util.mkdr(Project_name, Circle_dir, img_dir)
        # Test efficacy of Blob Detector on Real Image

        # imm = util.testCircleDetector(data_path, blob_path)
        # ts = Circularity.numCircles(imm)

        ## Create and Train CircleNet

        circleNet = Circularity.init_circle_net(net_params["dk"], net_params["ds"], net_params["df"], net_params["dp"])
        circle_dir = 'TrainedCNet'
        circle_path = project_path + circle_dir + '/'

        # if reusing weights
        if use_Circ == 1:
            circleNet = Circularity.CircleWeights(circleNet, circle_path, False)
            # print('Uncomment CNet')
        # If training circlenet
        if use_Circ == 2:

            util.mkdr(project_path, circle_dir, 1)
            Circularity.trainCNet(data_type, data_path, img_size, scale_factor, circleNet, project_path)
            Circularity.CircleWeights(circleNet, circle_path, True)



    ## Create GAN
    netD, netG = networks.slicegan_nets(**net_params)
    # netD, netG = networks.slicegan_nets(Project_path, Training, image_type, dk, ds, df, dp, gk, gs, gf, gp)

    lz_calced = model.lz_img_size_converter(net_params["gk"], net_params["gs"], net_params["gp"], img_size)

    # Train
    if Training:
        train_params = {
            "pth": project_path,
            "imtype": image_type,
            "datatype": data_type,
            "real_data": data_path,
            "Disc": netD,
            "Gen": netG,
            "nc": img_channels,
            "l": img_size,
            "nz": z_channels,
            "sf": scale_factor,
            "lz": lz_calced,
            "num_epochs": 10,
            "CircNet": circleNet(),
            "use_Circ": use_Circ,
            "noise_type": noise_type,
            "sub_images": 32,
            "beta1": beta1,
            "beta2": beta2
        }

        model.train(**train_params)
    else:
        test_params = {
            "pth": project_path,
            "imtype": image_type,
            "netG": netG(),
            "nz": z_channels,
            "lf": 4,
            "periodic": False,
            "noise_type": noise_type
        }
        img, raw, netG = util.test_img(**test_params)
