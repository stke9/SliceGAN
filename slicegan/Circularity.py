from run_slicegan import dk, ds, dp, df
from slicegan import preprocessing, util
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import time
import matplotlib
import cv2
import torch.nn.functional as F
import pickle
from cv2 import SimpleBlobDetector


class CircleNet(nn.Module, dk, ds, dp, df):
    def __init__(self):
        super(CircleNet, self).__init__()
        self.convs = nn.ModuleList()
        for lay, (k, s, p) in enumerate(zip(dk, ds, dp)):
            self.convs.append(nn.Conv2d(df[lay], df[lay + 1], k, s, p, bias=False))

    def forward(self, x):
        for conv in self.convs[:-1]:
            x = F.relu_(conv(x))
        x = self.convs[-1](x)
        return x


def trainCNet(datatype, realData, l, sf, CNet):
    """
        train the network to detect and count circles
        :param datatype: training data format e.g. tif, jpg ect
        :param real_data: path to training data
        :param nc: channels
        :param CNet:
        :param l: image size
        :param nz: latent vector size
        :param sf: scale factor for training data
        :return:
    """

    if len(realData) == 1:
        realData *= 3

    print('Loading Dataset...')
    dataset_xyz = preprocessing.batch(realData, datatype, l, sf)

    ## Constants for NNs
    # matplotlib.use('Agg')
    ngpu = 1
    numEpochs = 30

    # batch sizes
    batch_size = 8
    # optimiser params for G and D
    lrc = 0.0001
    # change values of beta1 between 0.1-0.9, beta2 0.9-0.99 and calc evals?
    Beta1 = 0.9  # Different value as the use case here is fairly standard and therefore would benefit from a non-zero initialization of Beta1
    Beta2 = 0.99
    circle_dim = 0
    cudnn.benchmark = True
    workers = 0

    ##Dataloaders for each orientation
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    print(device, " will be used.\n")

    # Data Loaded along the dimension where circles are to be observed and counted
    dataLoader = torch.utils.data.DataLoader(dataset_xyz[circle_dim], batch_size=batch_size, shuffle=True,
                                             num_workers=workers)

    # Define Network

    cNet = CNet().to(device)
    if ('cuda' in str(device)) and (ngpu > 1):
        cNet = nn.DataParallel(cNet, list(range(ngpu)))
    optC = optim.Adam(cNet.parameters(), lr=lrc, betas=(Beta1, Beta2))
    cNet.zero_grad()

    print("Starting CNet Training...")

    for e in range(numEpochs):

        realData = dataLoader.to(device)
        iterc = 0

        for R in realData:
            pred_OutR = cNet(R).view(-1)
            real_OutR = numCircles(R)

            predR, realR = int(pred_OutR), int(real_OutR)

            iterc += 1

            print(f"Epoch {e} : Slice {iterc} - NRC {realR} NPR {predR} Diff {predR - realR}")

            CLoss = pred_OutR - real_OutR

        CLoss.backward()
        optC.step()


def CircleWeights(cnet, WeightPath, SL=bool(True)):
    """
    :param cnet: circlenet model
    :param WeightPath: Path to save or load weights
    :param SL: flag parameter to determine whether weights need to be saved or loaded
    :return:
    """

    if SL:
        torch.save(cnet.state_dict(), WeightPath)
    else:
        cnet = CircleNet()
        cnet.load_state_dict(torch.load(WeightPath))
        return cnet


def numCircles(slice_i):
    params = cv2.SimpleBlobDetector_Params()

    # params.filterByArea = True
    # params.minArea = 100

    params.filterByCircularity = True
    params.minCircularity = 0.9

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(slice_i)

    return len(keypoints)


def CircularityLoss(imreal, imfake, CL_CNET):
    realcirc, fakecirc, diffcircL = []
    rlen, flen = 0
    D = 0

    for r in imreal:
        realcirc.append(CL_CNET(r))
        rlen += 1

    for f in imfake:
        fakecirc.append(CL_CNET(f))
        flen += 1

    if rlen != flen:
        print("\n The number of real and fake slices do not match")
        return 0

    for i, R, F in enumerate(zip(realcirc, fakecirc)):
        diffcirc = ((F - R) ** 2) if R > F else 0  # 0 can also be substituted by int((R-F)**2)
        diffcircL.append(diffcirc)

        print(f"Slice {i} has a difference of {diffcirc} circles between real and fake \n")

    for diff in diffcircL:
        D += diff

    return D/rlen
