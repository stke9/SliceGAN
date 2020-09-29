import numpy as np
import torch
import matplotlib.pyplot as plt
import tifffile
def Batch(img1,img2,img3,type,l, sf,TI):
    Testing=TI

    if type == 'png' or type == 'jpg':
        datasetxyz = []
        #img = np.load(img1)
        img = plt.imread(img1)
        if len(img.shape)>2:
            img = img[:,:,0]
        img = img[::sf,::sf]
        x_max, y_max= img.shape[:]
        phases = np.unique(img)

        data = np.empty([32 * 900, len(phases), l, l])
        for i in range(32 * 900):
            x = np.random.randint(1, x_max - l-1)
            y = np.random.randint(1, y_max - l-1)
            # create one channel per phase for one hot encoding
            for cnt, phs in enumerate(phases):
                img1 = np.zeros([l, l])
                img1[img[x:x + l, y:y + l] == phs] = 1
                data[i, cnt, :, :] = img1

        if Testing:
            for j in range(7):
                plt.imshow(data[j, 0, :, :]+2*data[j, 1, :, :])
                plt.pause(0.3)
                plt.show()
                plt.clf()
            plt.close()
        data = torch.FloatTensor(data)
        dataset = torch.utils.data.TensorDataset(data)
        for i in range(3):
            datasetxyz.append(dataset)
    elif type == 'array':
        datasetxyz = []
        img = np.load(img1)
        l = 64
        dim = 0
        x_max, y_max, z_max = img.shape[:]
        data = np.empty([32 * 900, 2, l, l])
        for i in range(32 * 900):
            if i % (32 * 300) == 0:
                dim += 1
                print(i, dim)
            x = np.random.randint(1, x_max - l-1)
            y = np.random.randint(1, y_max - l-1)
            z = np.random.randint(1, z_max - 2)
            # create one channel per phase for one hot encoding
            for cnt, phs in enumerate([0, 1]):
                img1 = np.zeros([l, l])
                if dim == 1:
                    img1[img[x:x + l, z, y:y + l] == phs] = 1
                else:
                    img1[img[x:x + l, y:y + l, z] == phs] = 1
                data[i, cnt, :, :] = img1

        if Testing:
            for j in range(5):
                for ln in range(2):
                    plt.imshow(data[j, ln, :, :])
                    plt.pause(0.5)
                    plt.show()
                    plt.clf()
                plt.imshow(data[j, 0, :, :])
                plt.pause(1)
                plt.show()
                plt.clf()
            plt.close()
        data = torch.FloatTensor(data)
        dataset = torch.utils.data.TensorDataset(data)
        for i in range(3):
            datasetxyz.append(dataset)
    elif type=='tif':
        datasetxyz=[]
        img = np.array(tifffile.imread(img1))
        img = img[::sf,::sf,::sf]
        ## Create a data store and add random samples from the full image
        x_max, y_max, z_max = img.shape[:]
        print(img.shape)
        vals = np.unique(img)
        print(len(vals))
        for dim in range(3):
            data = np.empty([32 * 900, len(vals), l, l])
            print(dim)
            for i in range(32*900):
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
        front = plt.imread(img1)/128 - 1
        top = plt.imread(img2)/128 - 1
        side = plt.imread(img3)/128 -1
        imgs = [front, top, side]
        ## Create a data store and add random samples from the full image
        datasetxyz = []
        l = 64
        for img in imgs:
            #img = img[::2,::2,:]
            ep_sz = 32 * 1000
            data = np.empty([ep_sz, 3, l, l])
            x_max, y_max = img.shape[:2]
            for i in range(ep_sz):
                x = np.random.randint(0, x_max - l)
                y = np.random.randint(0, y_max - l)
                # create one channel per phase for one hot encoding
                data[i, 0, :, :] = img[x:x + l, y:y + l,0]
                data[i, 1, :, :] = img[x:x + l, y:y + l,1]
                data[i, 2, :, :] = img[x:x + l, y:y + l,2]

            if Testing:
                datatest = np.swapaxes(data,1,3)
                for j in range(5):
                    plt.imshow(datatest[j, :, :, :])
                    plt.pause(0.5)
                    plt.show()
                    plt.clf()
                plt.close()
            data = torch.FloatTensor(data)
            dataset = torch.utils.data.TensorDataset(data)
            datasetxyz.append(dataset)
    return datasetxyz

def CBatch(imgs,lbls, type, l, sf, TI):
    nlabs = len(lbls[0])
    data = np.empty([32 * 900, 3, l, l])
    labelset = np.empty([32 * 900, nlabs, 1, 1])
    p = 0
    nimgs = len(imgs)
    print('number of training imgs: ', nimgs, ' number of labels: ', nlabs)
    for img,lbl in zip(imgs,lbls):
        img = np.load(img)
        print(img.shape)
        if len(img.shape) > 3:
            img = img[:, :, 0]
        img = img[::sf, ::sf]
        x_max, y_max, z_max = img.shape[:]
        phases = np.unique(img)
        for i in range((32//nimgs) * 900):
            for j,lb in enumerate(lbl):
                labelset[p,j] = lb+1
            x = np.random.randint(1, x_max - l - 1)
            y = np.random.randint(1, y_max - l - 1)
            z = np.random.randint(1,z_max)
            # create one channel per phase for one hot encoding
            # for cnt, phs in enumerate(phases):
            #     img1 = np.zeros([l, l])
            #     img1[img[x:x + l, y:y + l, z] == phs] = 1
            #     data[p, cnt, :, :] = img1
            p+=1
            if i%1800==0:
                plt.imshow(data[p-1,0] + 2*data[p-1,1])
                print(lbl)
                plt.pause(1)
                plt.close('all')
    data = torch.FloatTensor(data)
    labelset = torch.FloatTensor(labelset)
    dataset = torch.utils.data.TensorDataset(data,labelset)
    return [dataset]*3

