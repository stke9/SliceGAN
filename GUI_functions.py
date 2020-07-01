from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt

def skew_image(img, angle, xsq, ysq):
    width, height = img.size
    xshift = np.tan(abs(angle)) * height
    new_width = width + int(xshift)
    if new_width < 0:
        return img
    img = img.transform((new_width, height), Image.AFFINE,
        (1, angle, -xshift if angle > 0 else 0, 0, 1, 0),Image.BICUBIC)
    img = img.resize(size = (int(img.size[0]/xsq), int(img.size[1]/ysq)))
    return img

def cubeView(imgs):

    im2 = skew_image(imgs[1], -0.45, 1, 1.7)
    im3 = skew_image(imgs[2], 0.63, 1, 2.1)
    im3 = im3.rotate(90,expand = 1)
    final_image = np.zeros((im2.size[1]+66,im2.size[0]+2, 4))
    final_image[-74:, -74:, :] = imgs[0]
    final_image[:im2.size[1], -im2.size[0]:] = im2
    im3arr = np.array(im3)
    plt.imshow(im3arr)
    final_image[-im3.size[1]:,  :im3.size[0]] +=im3arr[:final_image.shape[0]]
    return Image.fromarray(np.uint8(final_image))

imgs = [ImageOps.expand(Image.open('seprot.png').crop((0,0,64,64)), border = 5, fill = 'red')]*3
cube = cubeView(imgs)
cube.show()