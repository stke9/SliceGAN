import tifffile
import numpy as np
import matplotlib.pyplot as plt

p = 'TrainingData/3D_data_binary.tif'

p= "DataProcessing/3D_data_bin.tif"
a = np.array(tifffile.imread(p))

subslice = a[567, 44:100, 120:184]

fig, ax = plt.subplots()

ax.imshow(subslice)

fig.savefig("WEWEWE")
