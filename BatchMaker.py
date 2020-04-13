import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage import io
def Batch(img1,img2,img3,type,TI):
    Testing=TI
    if type == 'array2D':
        datasetxyz = []
        img = np.load(img1)
        l = 64
        dim = 0
        x_max, y_max= img.shape[:]
        data = np.empty([32 * 900, 3, l, l])
        for i in range(32 * 900):
            if i % (32 * 300) == 0:
                dim += 1
            x = np.random.randint(1, x_max - l-1)
            y = np.random.randint(1, y_max - l-1)
            # create one channel per phase for one hot encoding
            for cnt, phs in enumerate([0, 1, 2]):
                img1 = np.zeros([l, l])
                if dim == 1:
                    img1[img[x:x + l, y:y + l] == phs] = 1
                else:
                    img1[img[x:x + l, y:y + l] == phs] = 1
                data[i, cnt, :, :] = img1

        if Testing:
            plt.ioff()
            for j in range(3):
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
        img = np.array(io.imread(img1))
        img = img[::4,::4,::4]
        img = np.swapaxes(img,2,1)
        fig, axs = plt.subplots(3,4)
        [axi.set_axis_off() for axi in axs.ravel()]
        check_ori(img,[0,0,1],fig,axs)
        ## Create a data store and add random samples from the full image
        l = 64
        dim = 0
        x_max, y_max, z_max = img.shape[:]
        print(img.shape)
        vals = np.unique(img)

        for dim in range(3):
            data = np.empty([32 * 900, 3, l, l])
            print(dim)
            for i in range(32*900):
                x = np.random.randint(0, x_max - l)
                y = np.random.randint(0, y_max - l)
                z = np.random.randint(0, z_max-1)
                # create one channel per phase for one hot encoding
                for cnt,phs in enumerate(list(vals)):
                    img1 = np.zeros([l,l])
                    if dim==1:
                        img1[img[x:x + l,z, y:y + l] == phs] = 1
                    elif dim==2:
                        img1[img[z, x:x + l, y:y + l] == phs] = 1
                    else:
                        img1[img[x:x + l, y:y + l, z] == phs] = 1
                    data[i, cnt, :, :] = np.swapaxes(img1[:,:],0,1)

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
        front = plt.imread(img1)/255
        top = plt.imread(img2)/255
        side = plt.imread(img3)/255
        imgs = [front, top, side]
        ## Create a data store and add random samples from the full image
        datasetxyz = []
        l = 64
        for img in imgs:
            img = img[::2,::2,:]
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

def check_ori(img,rots,fig,axs):
    ax_grps = [[(1,1),(1,3)],[(0,1),(2,1)],[(1,0),(1,2)]]
    slcs = [img[0,:,:], img[:,0,:],img[:,:,0]]
    print(rots)
    for i, (a,slc,rot) in enumerate(zip(ax_grps, slcs, rots)):
        if rot:
            slc = np.swapaxes(slc,0,1)
        axs[a[0]].imshow(slc)
        axs[a[1]].imshow(slc)
        plt.pause(0.1)
    chng = input()
    if chng == '':
        return rot
    else:
        rots[int(chng)] = abs(rots[int(chng)]-1)
        check_ori(img, rots,fig,axs)


