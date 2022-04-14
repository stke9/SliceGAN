import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from slicegan import networks, util, model

def calc_lz_dims(min_x, min_y, min_z, gk, gs, gp):

    found_x, found_y, found_z = False, False, False
    for i in range(20):
        temp_img_size  = model.lz_img_size_converter(gk, gs, gp, lz = i, lz_to_im=True)
        
        if (not found_x) and (temp_img_size >= min_x):
            lz_x = i
            im_x = temp_img_size
            found_x = True
        
        if (not found_y) and (temp_img_size >= min_y):
            lz_y = i
            im_y = temp_img_size
            found_y = True
        
        if (not found_z) and (temp_img_size >= min_z):
            lz_z = i
            im_z = temp_img_size
            found_z = True

        if found_z and found_y and found_x:
            break
    
    return lz_x, lz_y, lz_z, im_x, im_y, im_z
    
def show_random_slices(volume, figpath, fig_title):
    N = volume.shape
    fig, ax = plt.subplots(ncols=3, figsize = (20, 7))
    
    for i, N_i in enumerate(N):
        r = np.random.randint(N_i)
        if i == 0:
            cross_sect = volume[r,:,:]
            title = "YZ-plane"
        elif i == 1:
            cross_sect = volume[:,r,:]
            title = "XZ-plane"
        elif i == 2:
            cross_sect = volume[:,:,r]
            title = "XY-plane"

        ax[i].imshow(cross_sect)
        ax[i].set_title(title)
        # ax[i].tick_params(
        #     axis='both',          # changes apply to the x-axis
        #     which='both',      # both major and minor ticks are affected
        #     bottom=False,      # ticks along the bottom edge are off
        #     top=False,         # ticks along the top edge are off
        #     labelbottom=False) # labels along the bottom edge are off
        
        # ax[i].tick_params(
        #     axis='y',          # changes apply to the x-axis
        #     which='both',      # both major and minor ticks are affected
        #     bottom=False,      # ticks along the bottom edge are off
        #     top=False,         # ticks along the top edge are off
        #     labelbottom=False) # labels along the bottom e
        
        ax[i].tick_params(
            axis='both',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False, top= False, left = False, right = False,      # ticks along the bottom edge are off
            labelbottom=False, labeltop= False, 
            labelleft = False, labelright = False) # labels along the bottom e
        # ax.imshow(cross_sect)
        # ax.set_title(title)
    fig.suptitle(fig_title)
    fig.savefig(figpath)

def test_binary_generator(proj_dir, proj_name, min_xyz, noise_type = "normal"):
    img_channels = 2
    lays = 6
    Training = False
    z_channels = 16
    min_x, min_y, min_z = min_xyz

    pth = util.mkdr(proj_name , proj_dir, Training)
    print(pth)
        
    net_params = {

        "pth": pth,
        "Training": Training,
        "imtype": 'twophase',

        "dk" : [4]*lays,
        "gk" : [4]*lays,

        "ds": [2]*lays,
        "gs": [2]*lays,

        "df": [img_channels,64,128,256,512,1],
        "gf": [z_channels,512,256,128,64,img_channels],

        "dp": [1,1,1,1,0],
        "gp": [2,2,2,2,3],

    }

    _, netG = networks.slicegan_nets(**net_params)
    netG = netG()
    netG.load_state_dict(torch.load(pth + '_Gen.pt', map_location=torch.device("cpu")))
    netG.eval()

    lz_x, lz_y, lz_z, im_x, im_y, im_z = calc_lz_dims(min_x, min_y, min_z, net_params["gk"], net_params["gs"], net_params["gp"])

    noise_dist = model.noise_distributions[noise_type]
    noise = noise_dist.sample((1, z_channels, lz_x, lz_y, lz_z))
    print(noise.size())
    # noise = torch.randn(1, z_channels, lz_x, lz_y, lz_z)

    # volume = netG(noise)[0,0,:,:,:].cpu().detach().numpy()
    volume = netG(noise).cpu().detach()
    bin_volume = torch.zeros_like(volume[0,0])
    bin_volume[volume[0,0]>volume[0,1]] = 1
    
    bin_volume = bin_volume.numpy()
    return bin_volume