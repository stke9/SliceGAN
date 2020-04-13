import numpy as np
import matplotlib.pyplot as plt
import os
import tifffile
import torch
import torch.nn as nn
import torch.autograd as autograd
## Training Utils
def mkdr(proj):
    try:
        os.mkdir(proj)
        return proj
    except:
        print('Directory', proj, 'already exists. Enter new project name or hit enter to overwrite')
        new = input()
        if new == '':
            return proj
        else:
            mkdr(new)
            return new

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def calc_gradient_penalty(netD, real_data, fake_data, batch_size, dim, device, gp_lambda,nc):
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(batch_size, int(real_data.nelement() / batch_size)).contiguous()
    alpha = alpha.view(batch_size, nc, dim, dim)
    alpha = alpha.to(device)

    fake_data2 = fake_data.view(batch_size, nc, dim, dim)
    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data2.detach())

    interpolates = interpolates.to(device)
    interpolates.requires_grad_(True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * gp_lambda
    return gradient_penalty

def calc_ETA(steps, time, start, i, epoch, num_epochs):
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
def PostProc(img,imtype):
    if imtype == 'colour':
        return 255*np.swapaxes(img[0], 0, -1)
    if imtype == 'twophase':
        sqrs = np.zeros(img.shape[2:])
        p1 = np.array(img[0][0])
        p2 = np.array(img[0][1])
        sqrs[(p1 < p2)] = 1  # background, yellow
        return sqrs
    if imtype == 'threephase':
        sqrs = np.zeros(img.shape[2:])
        p1 = np.array(img[0][0])
        p2 = np.array(img[0][1])
        p3 = np.array(img[0][2])
        sqrs[(p1 > p2) & (p1 > p3)] = 0  # background, yellow
        sqrs[(p2 > p1) & (p2 > p3)] = 1  # spheres, green
        sqrs[(p3 > p2) & (p3 > p1)] = 2  # binder, purple
        return sqrs

def Plotter(sqrs,slcs,imtype):
    fig, axs = plt.subplots(slcs, 3)
    if imtype == 'colour':
        for j in range(slcs):
            axs[j, 0].imshow(sqrs[j, :, :, :])
            axs[j, 1].imshow(sqrs[:, j, :, :])
            axs[j, 2].imshow(sqrs[:, :, j, :])
    else:
        for j in range(slcs):
            axs[j, 0].imshow(sqrs[j, :, :])
            axs[j, 1].imshow(sqrs[:, j, :])
            axs[j, 2].imshow(sqrs[:, :, j])

def show_img(img):
    for j in range(np.shape(img)[-2]):
        plt.imshow(img[j, :, :])
        plt.pause(0.1)
        plt.clf()
    plt.close('all')

def test_img(Proj, imtype, netG, nz = 64, lf = 1, show = False):
    netG.load_state_dict(torch.load(Proj + '/' + Proj + '_Gen.pt'))
    netG.eval()
    noise = torch.randn(1, nz, lf, lf, lf)
    img = netG(noise).detach()
    print('Postprocessing')
    gb = PostProc(img,imtype)
    tif = np.int_(gb)
    tifffile.imwrite(Proj + '.tif', tif)
    if show:
        show_img(tif)
    return tif



