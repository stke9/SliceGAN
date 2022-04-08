import numpy as np
import torch
import matplotlib.pyplot as plt
import tifffile
def batch(data,type,l, sf, sub_images = 32*900):
    """
    Generate a batch of images randomly sampled from a training microstructure
    :param data: data path
    :param type: data type
    :param l: image size
    :param sf: scale factor
    :return:
    """
    Testing = False
    if type == 'png' or type == 'jpg':
        datasetxyz = []
        for img in data:
            img = plt.imread(img)
            if len(img.shape)>2:
                img = img[:,:,0]
            img = img[::sf,::sf]
            x_max, y_max= img.shape[:]
            phases = np.unique(img)
            data = np.empty([sub_images, len(phases), l, l])
            for i in range(sub_images):
                x = np.random.randint(0, x_max - l)
                y = np.random.randint(0, y_max - l)
                # create one channel per phase for one hot encoding
                for cnt, phs in enumerate(phases):
                    img1 = np.zeros([l, l])
                    img1[img[x:x + l, y:y + l] == phs] = 1
                    data[i, cnt, :, :] = img1

            if Testing:
                for j in range(7):
                    plot_data = np.zeros((l,l))
                    for i in range(phases-1):
                        plot_data += i/phases*data[j, i, :, :]
                    plt.imshow(plot_data)
                    plt.pause(0.3)
                    plt.show()
                    plt.clf()
                plt.close()
            data = torch.FloatTensor(data)
            dataset = torch.utils.data.TensorDataset(data)
            datasetxyz.append(dataset)

    elif type=='tif':
        datasetxyz=[]
        img = np.array(tifffile.imread(data[0]))
        img = img[::sf,::sf,::sf]
        ## Create a data store and add random samples from the full image
        x_max, y_max, z_max = img.shape[:]
        print('training image shape: ', img.shape)
        vals = np.unique(img)
        for dim in range(3): # change back to 3
            data = np.empty([sub_images, len(vals), l, l])
            print('dataset ', dim)
            for i in range(sub_images):
                x = np.random.randint(0, x_max - l)
                y = np.random.randint(0, y_max - l)
                z = np.random.randint(0, z_max - l)
                # create one channel per phase for one hot encoding
                lay = np.random.randint(img.shape[dim]-1)
                for cnt,phs in enumerate(list(vals)):
                    img1 = np.zeros([l,l])
                    if dim==0:
                        img1[img[lay, y:y + l, z:z + l] == phs] = 1
                    elif dim==1:
                        img1[img[x:x + l,lay, z:z + l] == phs] = 1
                    else:
                        img1[img[x:x + l, y:y + l,lay] == phs] = 1
                    data[i, cnt, :, :] = img1[:,:]
                    # data[i, (cnt+1)%3, :, :] = img1[:,:]

            if Testing:
                for j in range(2):
                    plt.imshow(data[j, 0, :, :] + 2 * data[j, 1, :, :])
                    plt.pause(1)
                    plt.show()
                    plt.clf()
                plt.close()
            data = torch.FloatTensor(data)
            dataset = torch.utils.data.TensorDataset(data)
            datasetxyz.append(dataset)

    elif type=='colour':
        ## Create a data store and add random samples from the full image
        datasetxyz = []
        for img in data:
            img = plt.imread(img)
            img = img[::sf,::sf,:]
            data = np.empty([sub_images, 3, l, l])
            x_max, y_max = img.shape[:2]
            for i in range(sub_images):
                x = np.random.randint(0, x_max - l)
                y = np.random.randint(0, y_max - l)
                # create one channel per phase for one hot encoding
                data[i, 0, :, :] = img[x:x + l, y:y + l,0]
                data[i, 1, :, :] = img[x:x + l, y:y + l,1]
                data[i, 2, :, :] = img[x:x + l, y:y + l,2]
            print('converting')
            if Testing:
                datatest = np.swapaxes(data,1,3)
                datatest = np.swapaxes(datatest,1,2)
                for j in range(5):
                    plt.imshow(datatest[j, :, :, :])
                    plt.pause(0.5)
                    plt.show()
                    plt.clf()
                plt.close()
            data = torch.FloatTensor(data)
            dataset = torch.utils.data.TensorDataset(data)
            datasetxyz.append(dataset)
    elif type=='grayscale':
        datasetxyz = []
        for img in data:
            img = plt.imread(img)
            if len(img.shape) > 2:
                img = img[:, :, 0]
            img = img/img.max()
            img = img[::sf, ::sf]
            x_max, y_max = img.shape[:]
            data = np.empty([sub_images, 1, l, l])
            for i in range(sub_images):
                x = np.random.randint(1, x_max - l - 1)
                y = np.random.randint(1, y_max - l - 1)
                subim = img[x:x + l, y:y + l]
                data[i, 0, :, :] = subim
            if Testing:
                for j in range(7):
                    plt.imshow(data[j, 0, :, :])
                    plt.pause(0.3)
                    plt.show()
                    plt.clf()
                plt.close()
            data = torch.FloatTensor(data)
            dataset = torch.utils.data.TensorDataset(data)
            datasetxyz.append(dataset)
    return datasetxyz