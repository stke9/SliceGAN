import numpy as np
import matplotlib.pyplot as plt
import torch

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
    
def show_random_slices(volume):
    N = volume.shape

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

        fig, ax = plt.subplots()
        ax.imshow(cross_sect)
        ax.set_title(title)
        fig.savefig(title)

img_channels = 2
lays = 6
Training = False
Project_name = '3D_binary_exemplar_final'
Project_dir = 'Trained_Generators'
z_channels = 16

pth = util.mkdr(Project_dir, Project_name, Training)
    
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

netD, netG = networks.slicegan_nets(**net_params)
netG = netG()
netG.load_state_dict(torch.load(pth + '_Gen.pt', map_location=torch.device("cpu")))
netG.eval()

min_x, min_y, min_z = 64, 200, 200

lz_x, lz_y, lz_z, im_x, im_y, im_z = calc_lz_dims(min_x, min_y, min_z, net_params["gk"], net_params["gs"], net_params["gp"])

# noise_dim = 14
noise = torch.randn(1, z_channels, lz_x, lz_y, lz_z)

volume = netG(noise)[0,0,:,:,:].cpu().detach().numpy()
s = volume.shape
print(f"Generated volume of size ({s[0]} * {s[1]} * {s[2]})")
print(volume.shape)
show_random_slices(volume)