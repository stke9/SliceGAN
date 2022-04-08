import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from scipy import ndimage as ndi
from scipy.ndimage import sobel
from skimage import morphology, color
from skimage.filters.thresholding import threshold_otsu
from skimage.morphology import closing, square
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from tifffile import tifffile

# Configuration settings
class config:
  c = {
    "filename": "3D_data_binary.tif", # filename of image
    "thr_k_size": 4,                  # threshold kernel size
    "rm_holes_size": 12,              # threshold in pixels
    "rm_objects_size": 24,            # threshold in pixels
    "local_peak_max": 1,              # while finding coordinates
    "closing_size": 3,                # size of closing algorithm
    "h_minima": 3                     # depth of the h minima
  }
  @staticmethod
  def var(name): return config.c[name]


c = config() # set the config as global variable

def main():
    raw, image = openTiff(c.var('filename'))
    distance, coords = computeDistance(image)
    markers = computeMarkers(distance, coords)
    labels = hmin_watershed(image, distance, markers)
    generatePlot(raw, image, distance, labels)


def openTiff(filename):
    raw = np.array(tifffile.imread(os.path.join(os.getcwd(), filename), key=0)) > 0 # Open file and normalise it
    return raw, cleanImage(raw)

def cleanImage(image):
    raw = closing(image > threshold_otsu(image), square(c.var('thr_k_size')))  # Apply closing and threshold
    raw = morphology.remove_small_holes(raw, c.var('rm_holes_size'))           # Remove small holes
    raw = morphology.remove_small_objects(raw, c.var('rm_objects_size'))       # Remove small objects
    return raw #sobel(raw)                                                     # Apply sobel filter


def computeDistance(image):
    distance = ndi.distance_transform_edt(image)
    # coords = peak_local_max(distance, footprint=np.ones((3, 3)), labels=cleaned)
    return distance, peak_local_max(distance, min_distance=c.var('local_peak_max'))


def computeMarkers(distance, coords):
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    # if c.var('closing_size') > 0: markers = closing(markers, square(c.var('closing_size')))
    return markers


def hmin_watershed(image, distance, markers, lines=False):
    return morphology.h_minima(watershed(-distance, markers, mask=image, watershed_line=lines), c.var('h_minima'))


def generatePlot(raw, image, distance, labels, tight=True):
    closed = closing(labels, square(3))
    fig, axes = plt.subplots(ncols=4, nrows=1, figsize=(40, 10))
    ax = axes.ravel()

    for a, data in zip(ax, [raw, image, -distance, color.label2rgb(closed, image, alpha=0.4, bg_label=0)]):
        # Plot the data of the full image and remove axis
        a.imshow(data, cmap=plt.cm.gray)
        a.set_axis_off()

        # Create smaller axes to show more details
        z = a.inset_axes([50, 575, 300, 300], transform=a.transData)
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

    # Show the final plot with tight lay-out if requested
    if tight: fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()


# ax[0].imshow(cleaned, cmap=plt.cm.gray)
# ax[1].imshow(-distance, cmap=plt.cm.gray)
# ax[2].imshow(morphology.h_minima(labels, 2), cmap=plt.cm.gray)

# zoom = ax[0].inset_axes([0.5, 0.5, 0.47, 0.47])
# zoom.set_xlim((50, 250))
# zoom.set_ylim((50, 250))
# zoom.imshow(cleaned, origin="lower", cmap=plt.cm.gray)