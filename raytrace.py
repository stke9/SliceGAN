import numpy as np
from plotoptix import TkOptiX
from plotoptix.materials import make_material
from plotoptix.materials import m_flat
from plotoptix.utils import map_to_colors  # map variable to matplotlib color map
### Welcome to SliceGAN ###
####### Steve Kench #######
import tifffile
import colorcet as cc
import matplotlib.pyplot as plt

s = 1
sb=1
a = 1
tort = False


min_cut = 0
min_val = 20
max_cut = 70
cmap = cc.cm.CET_L16
img = tifffile.imread('Trained_Generators/Teo/Deg/Deg.tif')
print(img.shape, np.unique(img))
crop = 10
# img = img[crop:-crop, crop:-crop, crop:-crop]
inrt = 1
img[:inrt, :inrt, :inrt] = -1
n = img.shape[0]
phases = np.unique(img)
phases = list(phases[1:])
phases.pop(1)
print('phases', phases)
phase_locs = {}
for ph in phases:
    phase_locs[str(ph)] = np.array(np.where(img == ph)).T
nphase = len(phases)
cols = [[0.1,0.1,0.1],[1,0,0],[0,0,1]]
x, y, z = img.shape
# cr = img[:, :, :, 0].reshape(-1)
# mask = cr ==0
# cr = cr[mask]
#
# c_parts = np.zeros((cr.size, 3))
# c_parts[:, 0] = cr
# c_parts[:,-1] = cb

print('done')

optix = TkOptiX(start_now=False) # no need to open the window yet
optix.set_param(min_accumulation_step=4,     # set more accumulation frames
                max_accumulation_frames=2000, # to get rid of the noise
                light_shading="Hard")        # use "Hard" light shading for the best caustics and "Soft" for fast convergence

optix.set_uint("path_seg_range", 15, 30)

alpha = np.full((1, 1, 4), 0.3, dtype=np.float32)
optix.set_texture_2d("mask", (255*alpha).astype(np.uint8))
m_diffuse_3 = make_material("TransparentDiffuse", color_tex="mask")
optix.setup_material("diffuse_1", m_diffuse_3)
for i, k in enumerate(phase_locs.keys()):
    c = plt.cm.jet(i/(nphase-1))
    print(i, c)
    optix.set_data(k, pos=phase_locs[k], u=[sb, 0, 0], v=[0, sb, 0], w=[0, 0, sb],
               geom="Parallelepipeds", # cubes, actually default geometry
               mat="diffuse",          # opaque, mat, default
               c=cols[i])

optix.setup_camera("cam1",cam_type="Pinhole", eye=[-3557.2212 , -730.4132, -1483.8723 ], target=[n/2,n/2,n/2], up=[0,-1, 0], fov=20)
optix.set_background(10)
# optix.set_ambient(0)


optix.set_float("tonemap_exposure", 0.5)
optix.set_float("tonemap_gamma", 2.2)

optix.add_postproc("Gamma")      # apply gamma correction postprocessing stage, or
# optix.add_postproc("Denoiser")  # use AI denoiser (exposure and gamma are applied as well)

x = n/2
optix.setup_light("light1", pos=[x, x, -x*2], color=10*np.array([1.0, 1.0, 1.0]), radius=50)
optix.setup_light("light2", pos=[x, -x*2, x], color=10*np.array([1.0, 1.0, 1.0]), radius=50)
optix.setup_light("light3", pos=[-484.97705,   n,  n], color=15*np.array([1.0, 1.0, 1.0]), radius=100)
print('starting')
optix.start()
