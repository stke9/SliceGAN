import torch
from torch import nn
import numpy as np
import cv2
import tifffile
import os
from matplotlib import pyplot as plt
from Train import trainer
from Architect import Architect
import util

## Data Processing
Project_name = 'NMC_filttest' #Creates directory with output images
image_type = 'threephase' # threephase, twophase or colour
data_type = 'tif' # png, jpg, tif, array, array2D
data_path = 'NMC/' # path to training data.
isotropic = False

## Network Architectures
ngpu = 1
imsize, nz,  channels = 64, 16, 3

dl, gl = [imsize,32,16,6,8,4,1], [1,4,6,8,16,32, imsize]                 # disc & gen layer sizes
dk, gk = [4,4,4,4,4], [4,4,4,4,4]                                    # kernal sizes
ds, gs = [2,2,2,2,2], [2,2,2,2,2]                                    # strides
df, gf = [channels,64,128,26,512,1], [nz,512,256,128,64, channels]  # filter sizes for hidden layers
dp, gp = [3,2,2,2,2],[2,2,2,2,3]
##Create Networks
netD, netG = Architect('NMC/NMC_filttest/NMC_filttest_',True, dk, ds, df,dp, gk ,gs, gf, gp)
netG = netG()
netD = netD()
device = torch.device("cuda:0" if(torch.cuda.is_available() and ngpu > 0) else "cpu")
print(device, " will be used.\n")


netG.load_state_dict(torch.load(data_path + Project_name + '/' + Project_name + '_Gen.pt'))
netG.eval()
netG = nn.DataParallel(netG)
netG.to(device)

netD = nn.DataParallel(netD)
netD.load_state_dict(torch.load(data_path + Project_name + '/' + Project_name + '_Disc.pt'))
netD.eval()
netD.to(device)

long = 20
imshape = int(32*(2+(long-4)))
blocksize = 32
blocks = int(imshape/blocksize)

noise = torch.randn(2,nz,20,4,4,requires_grad=True, device = device)
targ = torch.linspace(0.45,0.55, steps = blocks, device = device)
# crit = nn.MSELoss()
# targ = torch.tensor(10., device = device)
imglist=[]
for param in netG.parameters():
    param.requires_grad = False
for param in netD.parameters():
    param.requires_grad = False
steps = 20
D_disc_log = []
D_part_log = []
D_tot_log = []
labels = ['disc']#',part','tot'
for i in range(steps):
    D_disc=0
    img = netG(noise)
    masses = torch.zeros(blocks, device = device)
    for blck in range(blocks):
        st = blck*blocksize
        fin = st + blocksize
        masses[blck] = torch.mean(img[0, 1,st:fin])
    # for k in range(64):
    #     D_disc += netD(img[:,:,k,:,:]) #- gb_mass*4*i
    #     D_disc += netD(img[:, :, :, :, k])  # - gb_mass*4*i
    #     D_disc += netD(img[:, :, :, k, :])  # - gb_mass*4*i
    # D_disc = torch.mean(D_disc)/(64*3)
    D =  torch.mean((targ - masses)**2) #- D_disc
    # D_disc_log.append(D_disc)
    D_part_log.append(masses)
    # D_tot_log.append(D)
    D.backward()
    with torch.no_grad():
        noise -= noise.grad * 100 * blocksize
        noise.grad.zero_()
    imglist.append(util.PostProc(img.detach().cpu(), image_type))
    print(i)
    for loss in D_part_log:
        plt.plot(loss.cpu().detach().numpy())
    plt.savefig('opt_losses.png')
    plt.close('all')
    # if i%10==0:
    #     for k in range(64):
    #         plt.imshow(imglist[-1][k], cmap = 'gray')
    #         plt.pause(0.05)
    #         plt.clf()
    # plt.close('all')


fig, ax = plt.subplots(2)
im = imglist[-1]
ax[0].imshow(np.swapaxes(im[:,0], 0,1))
for c,loss in enumerate(D_part_log):
    ax[1].plot(loss.cpu().detach().numpy(), color = (1,c/steps,0))
for j in range(64):
    ax[0].imshow(np.swapaxes(im[:, j], 0, 1))
    plt.pause(0.5)
# noise_0 = torch.unsqueeze(noise[0],0)
# noise_1 = torch.unsqueeze(noise[1],0)
# img_mass = []
# noise_frac = []
# noises = []
# n = 5
# for i in range(n):
#     alpha = i/n
#     noisetemp = (noise_0*(1-alpha)) + noise_1*alpha
#     img = netG(noisetemp)
#     noise_frac.append(alpha)
#     img_mass.append(torch.mean(img[0,1]))
#     noises.append(noisetemp)
# plt.figure()
# plt.plot(noise_frac,img_mass)
# newnoise = torch.zeros(1,16,n*4,4,4)
# for j in range(n):
#     st = 4*j
#     fin = 4*(j+1)
#     newnoise[:,:,st:fin,:,:] = noises[j]
# im = util.PostProc(netG(newnoise).detach().cpu(), image_type)