import numpy as np
import matplotlib.pyplot as plt
import os
import tifffile
import torch
import torch.nn as nn
import torch.autograd as autograd
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
    img = img.detach().cpu()
    if imtype == 'colour':
        return (128*(1+np.swapaxes(img[0], 0, -1))).astype('int')
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

def TestPlotter(sqrs,slcs,imtype,pth):
    sqrs = PostProc(sqrs,imtype)
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
    plt.savefig(pth + '_slices.png')
    plt.close()

def GraphPlot(data,labels,pth,name):
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
    gb = PostProc(raw,imtype)
    tif = np.int_(gb)
    tifffile.imwrite(pth + '.tif', tif)

    return tif,raw, netG

def angslcs(img,l, bs,dim):
    angslc = torch.zeros(l*bs,2,l,l).cuda()
    if dim:
        img = torch.flip(img, dims = [2])
    pos = 0

    for off in range(-int(l/4),int(l/4)):
        slc = torch.diagonal(img, dim1=2, dim2=3, offset=off)
        up = nn.Upsample(size=(l, int(slc.shape[-1]* 2**0.5)), mode='bilinear', align_corners=True)
        slc = up(slc)
        slc1 = slc[:, :, :, -l:]
        slc2 = slc[:, :, :, :l]

        slc1 = slc1.permute(0, 1, 3, 2)
        slc2 = slc2.permute(0, 1, 3, 2)
        angslc[pos*bs:(pos+1)*bs] = slc1
        pos+=1
        angslc[pos * bs:(pos + 1) * bs] = slc2
        pos+=1
    return angslc

def filt_var(netG, Proj,it,epoch):
    MSEs = []
    for lay in range(5):
        weights = netG.convs._modules[str(lay)].weight
        ch_in, ch_out = weights.shape[:2]
        MSE = torch.zeros([ch_out, ch_out, ch_in])
        for i in range(ch_out):
            for j in range(ch_out):
                MSE[i,j, :] =  torch.mean((weights[:,i].detach()-weights[:,j].detach())**2, dim = [1,2,3])
        MSE[MSE==0]=1
        MSEs.append(MSE.cpu().numpy())

    fig, axs = plt.subplots(1,5,sharey = True)
    for lay,(MSE,ax)  in enumerate(zip(MSEs,axs)):
        vals = MSE[MSE<1]
        l = vals.shape
        ax.title.set_text('Weights' + str(lay))
        ax.hist(vals, weights = np.ones(l)/l, bins = 30)
        ax.ticklabel_format(axis = 'x', style = 'sci', scilimits=(0,0))
    fig.text(0.5, 0.04, 'MSE between a pair of filters', ha='center')
    fig.text(0.04, 0.5, 'Fraction of all filter pair combinations with given MSE', va='center', rotation='vertical')
    plt.savefig(Proj + '/' + Proj + '_filtvar_ep' + str(epoch) + '_iter' + str(it) + '.png',bbox_inches='tight')
    plt.close()
    return