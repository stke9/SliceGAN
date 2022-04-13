from sympy import NotReversible
import tifffile
import numpy as np
import matplotlib.pyplot as plt
import torch
import time

from slicegan import networks, util, model

def overlapping_noise(lz, periods, z_channels):
    invalid_periods = False
    if len(periods) != 3:
        invalid_periods = True
    for n in periods:
        if type(n) != int or n < 1:
            invalid_periods = True
    if invalid_periods:
        raise ValueError("Invalid periods")

    nx, ny, nz = periods

    noise = torch.randn(nx,ny,nz, 1, z_channels, lz, lz, lz)

    for x_i in range(nx-1):
        noise[x_i+1, :, :, :, :, 0, :, :] = noise[x_i,:, :, :, :, -1, :, :].clone()
    
    for y_i in range(ny-1):
        noise[:, y_i+1,:,:, :, :, 0, :] = noise[:, y_i,:,:, :, :, -1, :].clone()
    
    for z_i in range(nz-1):
        noise[:, :, z_i + 1, :, :, :, :, 0] = noise[:, :, z_i + 1, :, :, :, :, -1].clone()

    return noise

def multi_fwd_pass(netG, noise_arrray, periods, img_size, inner_img_size):
    
    margin = img_size - inner_img_size
    V = lambda p: img_size + (p - 1)*inner_img_size
    
    Vx = V(periods[0])
    Vy = V(periods[1])
    Vz = V(periods[2])

    volume = torch.empty(size = (Vx, Vy, Vz))
    # volume = np.empty(shape = (128, 128, 128))

    def big_V_indices(i):
        if i == 0:
            i0 = 0
            i1 = img_size
        else:
            i0 = img_size + (i-1)*inner_img_size
            # print(i0)
            i1 = img_size + i*inner_img_size
            # print(i1)
        # return (i*img_size, (i+1)*img_size)
        # print(i0, i1)
        return (i0, i1)
    
    def sub_V_indices(i):
        if i == 0:
            i0 = 0
        else:
            i0 = margin
        i1 = img_size
        return (i0, i1)

    for x_i in range(periods[0]):
        for y_i in range(periods[1]):
            for z_i in range(periods[2]):

                sub_noise = noise_arrray[x_i, y_i, z_i, :, :, :, :, :]
                sub_volume = netG(sub_noise)[0,0]#.cpu().detach().numpy()

                x0_s, x1_s = sub_V_indices(x_i)
                y0_s, y1_s = sub_V_indices(y_i)
                z0_s, z1_s = sub_V_indices(z_i)
                

                sub_volume = sub_volume[x0_s:x1_s, y0_s:y1_s, z0_s:z1_s]
                
                # print(sub_volume.size())
                # print(sub_noise.size())

                print(sub_volume.size())
                print(sub_noise.size())
                
                x0_V, x1_V = big_V_indices(x_i)
                y0_V, y1_V = big_V_indices(y_i)
                z0_V, z1_V = big_V_indices(z_i)

                print(big_V_indices(z_i), "HEY")

                volume[x0_V:x1_V, y0_V :y1_V, z0_V :z1_V] = sub_volume
    
    return volume


# trained_model = 'Trained_Generators/3D_binary_exemplar_final/3D_binary_exemplar_final.tif'

start = time.time()

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

lz = 5

img_size = model.lz_img_size_converter(net_params["gk"], net_params["gs"], net_params["gp"], lz = lz, lz_to_im=True)
inner_img_size = model.lz_img_size_converter(net_params["gk"], net_params["gs"], net_params["gp"], lz = lz-1, lz_to_im=True)
margin = img_size - inner_img_size


print(img_size, inner_img_size)
# size_margin = img_size - inner_img_size

periods = (1,1,2)

noise_arrray = overlapping_noise(lz, periods, z_channels)

print(noise_arrray.size())

volume = multi_fwd_pass(netG, noise_arrray, periods, img_size, inner_img_size)

print(volume.size())
# print(volume.size())


print(time.time() - start)

# volume = volume.cpu().detach().numpy()


fig, ax = plt.subplots()

ax.imshow(volume[23,:,:].cpu().detach().numpy())

fig.savefig("period-test")

