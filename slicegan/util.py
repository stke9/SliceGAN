import os

import cv2
from torch import nn
from slicegan import util
import torch
from torch import autograd
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import sys
## Training Utils

def mkdr(proj_dir, proj, Training):
    """
    When training, creates a new project directory or overwrites an existing directory according to user input. When testing, returns the full project path
    :param proj: project name
    :param proj_dir: project directory
    :param Training: whether new training run or testing image
    :return: full project path
    """
    pth = proj_dir + '/' + proj
    if Training:
        try:
            os.mkdir(pth)
            return pth + '/'
        except FileExistsError:
            print('Directory', pth, 'already exists. Enter new project name or hit enter to overwrite')
            new = input()
            if new == '':
                return pth + '/'
            else:
                pth = mkdr(proj_dir, new, Training)
                return pth + '/'
        except FileNotFoundError:
            print('The specifified project directory ' + proj_dir + ' does not exist. Please change to a directory that does exist and again')
            sys.exit()
    else:
        return pth + '/'


def weights_init(m):
    """
    Initialises training weights
    :param m: Convolution to be intialised
    :return:
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def calc_gradient_penalty(netD, real_data, fake_data, batch_size, l, device, gp_lambda,nc):
    """
    calculate gradient penalty for a batch of real and fake data
    :param netD: Discriminator network
    :param real_data:
    :param fake_data:
    :param batch_size:
    :param l: image size
    :param device:
    :param gp_lambda: learning parameter for GP
    :param nc: channels
    :return: gradient penalty
    """
    #sample and reshape random numbers
    alpha = torch.rand(batch_size, 1, device = device)
    alpha = alpha.expand(batch_size, int(real_data.nelement() / batch_size)).contiguous()
    alpha = alpha.view(batch_size, nc, l, l)

    # create interpolate dataset
    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())
    interpolates.requires_grad_(True)

    #pass interpolates through netD
    disc_interpolates = netD(interpolates)  # passing through the network
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size(), device = device),
                              create_graph=True, only_inputs=True)[0]
    # extract the grads and calculate gp
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * gp_lambda
    return gradient_penalty


def calc_eta(steps, time, start, i, epoch, num_epochs):
    """
    Estimates the time remaining based on the elapsed time and epochs
    :param steps:
    :param time: current time
    :param start: start time
    :param i: iteration through this epoch
    :param epoch:
    :param num_epochs: totale no. of epochs
    """
    elap = time - start
    progress = epoch * steps + i + 1
    rem = num_epochs * steps - progress
    ETA = rem / progress * elap
    hrs = int(ETA / 3600)
    mins = int((ETA / 3600 % 1) * 60)
    print('[%d/%d][%d/%d]\tETA: %d hrs %d mins'
          % (epoch, num_epochs, i, steps,
             hrs, mins))

## Plotting Utils
def post_proc(img,imtype):
    """
    turns one hot image back into grayscale
    :param img: input image
    :param imtype: image type
    :return: plottable image in the same form as the training data
    """
    try:
        #make sure it's one the cpu and detached from grads for plotting purposes
        img = img.detach().cpu()
    except:
        pass
    # for n phase materials, seperate out the channels and take the max
    if imtype == 'twophase':
        # print(f"\nImage Shape before 2phase convert = {img.shape}\np1 = {img} \nImage p1 = {np.array(img[0][1]).shape}")
        img_pp = np.zeros(img.shape[2:])
        p1 = np.array(img[0][0])
        p2 = np.array(img[0][1])
        img_pp[(p1 < p2)] = 1  # background, yellow
        # print(img_pp.shape)
        return img_pp
    if imtype == 'threephase':
        img_pp = np.zeros(img.shape[2:])
        p1 = np.array(img[0][0])
        p2 = np.array(img[0][1])
        p3 = np.array(img[0][2])
        img_pp[(p1 > p2) & (p1 > p3)] = 0  # background, yellow
        img_pp[(p2 > p1) & (p2 > p3)] = 1  # spheres, green
        img_pp[(p3 > p2) & (p3 > p1)] = 2  # binder, purple
        return img_pp
    # colour and grayscale don't require post proc, just a shift
    if imtype == 'colour':
        return np.int_(255 * (np.swapaxes(img[0], 0, -1)))
    if imtype == 'grayscale':
        return 255*img[0][0]

def test_plotter(img,slcs,imtype,pth, circ=False):
    """
    creates a fig with 3*slc subplots showing example slices along the three axes
    :param img: raw input image
    :param slcs: number of slices to take in each dir
    :param imtype: image type
    :param pth: where to save plot
    """

    img = post_proc(img,imtype)

    if circ:
        # print('(if circl == true )saving png in test_plotter...')
        # fig, ax = plt.subplots(slcs)
        # # for j in range(slcs):
        # #     axs[j, 0].imshow(img[j, :, :])
        # plt.savefig(pth + '_slices.png')
        # plt.close()
        plt.imsave(pth +'_slices.png', img)
    else:

        fig, axs = plt.subplots(slcs, 3)
        if imtype == 'colour':
            for j in range(slcs):
                axs[j, 0].imshow(img[j, :, :, :], vmin = 0, vmax = 255)
                axs[j, 1].imshow(img[:, j, :, :],  vmin = 0, vmax = 255)
                axs[j, 2].imshow(img[:, :, j, :],  vmin = 0, vmax = 255)
        elif imtype == 'grayscale':
            for j in range(slcs):
                axs[j, 0].imshow(img[j, :, :], cmap = 'gray')
                axs[j, 1].imshow(img[:, j, :], cmap = 'gray')
                axs[j, 2].imshow(img[:, :, j], cmap = 'gray')
        else:
            for j in range(slcs):
                axs[j, 0].imshow(img[j, :, :])
                axs[j, 1].imshow(img[:, j, :])
                axs[j, 2].imshow(img[:, :, j])
        plt.savefig(pth + '_slices.png')
        plt.close()

def graph_plot(data,labels,pth,name):
    """
    simple plotter for all the different graphs
    :param data: a list of data arrays
    :param labels: a list of plot labels
    :param pth: where to save plots
    :param name: name of the plot figure
    :return:
    """

    for datum, lbl in zip(data, labels):
        plt.plot(datum, label=lbl)

    plt.legend()
    plt.savefig(pth + '_' + name)
    plt.close()


def test_img(pth, imtype, netG, nz = 64, lf = 4, periodic=False, noise_type=None):
    """
    saves a test volume for a trained or in progress of training generator
    :param pth: where to save image and also where to find the generator
    :param imtype: image type
    :param netG: Loaded generator class
    :param nz: latent z dimension
    :param lf: length factor
    :param show:
    :param periodic: list of periodicity in axis 1 through n
    :return:
    """
    netG.load_state_dict(torch.load(pth + '_Gen.pt', map_location=torch.device("cpu")))
    netG.eval()
    noise = torch.randn(1, nz, lf, lf, lf)
    if periodic:
        if periodic[0]:
            noise[:, :, :2] = noise[:, :, -2:]
        if periodic[1]:
            noise[:, :, :, :2] = noise[:, :, :, -2:]
        if periodic[2]:
            noise[:, :, :, :, :2] = noise[:, :, :, :, -2:]
    raw = netG(noise)
    print('Postprocessing')
    gb = post_proc(raw,imtype)
    if periodic:
        if periodic[0]:
            gb = gb[:-1]
        if periodic[1]:
            gb = gb[:,:-1]
        if periodic[2]:
            gb = gb[:,:,:-1]
    tif = np.int_(gb)
    tifffile.imwrite(pth + '.tif', tif)

    return tif, raw, netG

def testCircleDetector(pathy, p2):

    image = np.array(tifffile.imread(pathy[0]))
    print(f"Image Shape: {image.shape}")

    image = image[0,:,:]
    valys = np.unique(image)

    # y = np.random.randint(0, 915)
    # z = np.random.randint(0, 915)
    ims = 916
    data = np.empty([1, len(valys), ims, ims])

    for cnt, phs in enumerate(list(valys)):
        img1 = np.zeros([ims, ims])

        img1[image[:, :] == phs] = 1
        data[0, cnt, :, :] = img1[:, :]
    util.test_plotter(data, 1, "twophase", p2, True)

    imgf = cv2.imread(p2 + "_slices.png")

    return imgf


