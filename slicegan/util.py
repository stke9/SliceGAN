import os
from torch import nn
import torch
from torch import autograd
import numpy as np
import matplotlib.pyplot as plt
import tifffile
## Training Utils
def mkdr(proj,proj_dir,Training):
    pth = proj_dir + proj
    if Training:
        try:
            os.mkdir(pth)
            return pth + '/' + proj
        except:
            print('Directory', pth, 'already exists. Enter new project name or hit enter to overwrite')
            new = input()
            if new == '':
                return pth + '/' + proj
            else:
                pth = mkdr(new, proj_dir, Training)
                return pth
    else:
        return pth + '/' + proj


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def calc_gradient_penalty(netD, real_data, fake_data, batch_size, l, device, gp_lambda,nc):
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(batch_size, int(real_data.nelement() / batch_size)).contiguous()
    alpha = alpha.view(batch_size, nc, l, l)
    alpha = alpha.to(device)

    fake_data2 = fake_data.view(batch_size, nc, l, l)
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


def cond_calc_gradient_penalty(netD, real_data, fake_data, batch_size, l, device, gp_lambda, nc, labs, nlabels):
    alpha = torch.rand(batch_size, 1)
    # lab_alpha = alpha.expand(batch_size, nlabels*l*l).view(batch_size,nlabels,l,l).contiguous().to(device)
    alpha = alpha.expand(batch_size, int(real_data.nelement() / batch_size)).contiguous()
    alpha = alpha.view(batch_size, nc, l, l)
    alpha = alpha.to(device)

    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())
    interpolates = interpolates.to(device)
    interpolates.requires_grad_(True)
    # lbl_interpolates = lab_alpha * real_lbls + ((1 - lab_alpha) * fk_lbls)
    # lbl_interpolates = lbl_interpolates.to(device)
    # lbl_interpolates.requires_grad_(True)
    gradient_penalty = 0
    for input_t in [interpolates]:#, lbl_interpolates]:
        disc_interpolates = netD(interpolates, labs)
        gradients = autograd.grad(outputs=disc_interpolates, inputs=input_t,
                                  grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                  create_graph=True, only_inputs=True)[0]

        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty += ((gradients.norm(2, dim=1) - 1) ** 2).mean() * gp_lambda
    return gradient_penalty


def calc_eta(steps, time, start, i, epoch, num_epochs):
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
    try:
        img = img.detach().cpu()
    except:
        pass
    if imtype == 'colour':
        return np.int_(255*(np.swapaxes(img[0], 0, -1)))
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
    if imtype == 'grayscale':
        return 255*img[0][0]

def test_plotter(sqrs,slcs,imtype,pth):
    sqrs = post_proc(sqrs,imtype)
    fig, axs = plt.subplots(slcs, 3)
    if imtype == 'colour':
        for j in range(slcs):
            axs[j, 0].imshow(sqrs[j, :, :, :], vmin = 0, vmax = 255)
            axs[j, 1].imshow(sqrs[:, j, :, :],  vmin = 0, vmax = 255)
            axs[j, 2].imshow(sqrs[:, :, j, :],  vmin = 0, vmax = 255)
    elif imtype == 'grayscale':
        for j in range(slcs):
            axs[j, 0].imshow(sqrs[j, :, :], cmap = 'gray')
            axs[j, 1].imshow(sqrs[:, j, :], cmap = 'gray')
            axs[j, 2].imshow(sqrs[:, :, j], cmap = 'gray')
    else:
        for j in range(slcs):
            axs[j, 0].imshow(sqrs[j, :, :])
            axs[j, 1].imshow(sqrs[:, j, :])
            axs[j, 2].imshow(sqrs[:, :, j])
    plt.savefig(pth + '_slices.png')
    plt.close()

def graph_plot(data,labels,pth,name):
    for datum,lbl in zip(data,labels):
        plt.plot(datum, label = lbl)
    plt.legend()
    plt.savefig(pth + '_' + name)
    plt.close()


def test_img(pth, imtype, netG, nz = 64, lf = 4, show = False):
    netG.load_state_dict(torch.load(pth + '_Gen.pt'))
    netG.eval()
    noise = torch.randn(1, nz, lf, lf, lf)
    raw = netG(noise)
    print('Postprocessing')
    gb = post_proc(raw,imtype)
    tif = np.int_(gb)
    tifffile.imwrite(pth + '.tif', tif)

    return tif,raw, netG

