import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from scipy import ndimage as ndi
from skimage import morphology, color
from skimage.filters.thresholding import threshold_otsu
from skimage.morphology import closing, square, opening
from skimage.segmentation import watershed, random_walker
from skimage.feature import peak_local_max

# Generate an initial image with two overlapping circles
from tifffile import tifffile

x, y = np.indices((80, 80))
x1, y1, x2, y2, x3, y3, x4, y4 = 8, 28, 34, 52, 55, 30, 20, 12
r1, r2, r3, r4 = 4, 20, 20, 6
mask_circle1 = (x - x1) ** 2 + (y - y1) ** 2 < r1 ** 2
mask_circle2 = (x - x2) ** 2 + (y - y2) ** 2 < r2 ** 2
mask_circle3 = (x - x3) ** 2 + (y - y3) ** 2 < r3 ** 2
mask_circle4 = (x - x4) ** 2 + (y - y4) ** 2 < r4 ** 2

image = np.logical_or(mask_circle2, mask_circle3)
image[mask_circle1] = 1
image[mask_circle4] = 1

file = os.path.join(os.getcwd(), '3D_data_binary.tif')
image = np.array(tifffile.imread(file, key=0)) > 0
threshold = closing(image > threshold_otsu(image), square(4))


cleaned = morphology.remove_small_objects(morphology.remove_small_holes(threshold, 12), 24)
distance = ndi.distance_transform_edt(cleaned)
coords = peak_local_max(distance, min_distance=1)


# coords = peak_local_max(distance, footprint=np.ones((3, 3)), labels=cleaned)
mask = np.zeros(distance.shape, dtype=bool)
mask[tuple(coords.T)] = True
markers, _ = ndi.label(mask)
labels = morphology.h_minima(watershed(-distance, markers, mask=cleaned, watershed_line=False), 4)
closed = closing(labels, square(3))


fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(30, 10))
ax = axes.ravel()

for a, data in zip(ax, [cleaned, -distance, color.label2rgb(closed, cleaned, alpha=0.4, bg_label=0)]):
    # Plot the blurred data for the full image
    a.imshow(data, cmap=plt.cm.gray)
    a.set_axis_off()

    # Create smaller axes to show more details
    z = a.inset_axes([50, 575, 300, 300], transform=a.transData)  # 50 to 350
    z.set_xlim((150, 250))
    z.set_ylim((200, 100))
    a.indicate_inset_zoom(z, ec="red", lw=8)
    mark_inset(a, z, loc1=1, loc2=2, fc="none", ec="red", lw=4)

    # Show data without ticks and set style for plot border
    z.imshow(data, origin="lower", cmap=plt.cm.gray)
    z.get_xaxis().set_ticks([])
    z.get_yaxis().set_ticks([])
    for edge in ['top', 'bottom', 'left', 'right']:
        z.spines[edge].set_linewidth(8)
        z.spines[edge].set_color('red')

fig.tight_layout()
plt.show()

# ax[0].imshow(cleaned, cmap=plt.cm.gray)
# ax[1].imshow(-distance, cmap=plt.cm.gray)
# ax[2].imshow(morphology.h_minima(labels, 2), cmap=plt.cm.gray)

# zoom = ax[0].inset_axes([0.5, 0.5, 0.47, 0.47])
# zoom.set_xlim((50, 250))
# zoom.set_ylim((50, 250))
# zoom.imshow(cleaned, origin="lower", cmap=plt.cm.gray)
