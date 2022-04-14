from slicegan import preprocessing, util
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import numpy as np
from pandas import DataFrame as Df
import cv2
import torch.nn.functional as F


dk = [4] * 6
ds = [2] * 6
dp = [1, 1, 1, 1, 0]
df = [3, 64, 128, 256, 512, 1]


def init_circle_net(dk, ds, df, dp):
    class CircleNet(nn.Module):
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

    return CircleNet


def trainCNet(datatype, realData, l, sf, CNet, project_path):
    """
        train the network to detect and count circles
        :type project_path: object
        :param datatype: training data format e.g. tif, jpg ect
        :param realData: path to training data
        :param CNet:
        :param l: image size
        :param sf: scale factor for training data
        :return:
    """

    if len(realData) == 1:
        realData *= 3

    params = cv2.SimpleBlobDetector_Params()

    params.filterByArea = False
    params.filterByConvexity = False
    params.filterByInertia = False

    params.filterByCircularity = True
    params.minCircularity = 0.5

    print('Loading Circle Dataset...')
    dataset_xyz = preprocessing.batch(realData, datatype, l, sf)
    # print(type(dataset_xyz[0]))

    ## Constants for NNs
    # matplotlib.use('Agg')
    ngpu = 1
    numEpochs = 10

    # batch sizes
    batch_size = 1  # CHANGE BACK TO 8
    # optimiser params
    lrc = 0.0001
    Beta1 = 0.9  # Different value as the use case here is fairly standard and therefore would benefit from a non-zero initialization of Beta1
    Beta2 = 0.99
    circle_dim = 0
    cudnn.benchmark = True
    workers = 0
    debug_flag = False

    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    print(device, " will be used.\n")

    # Data Loaded along the dimension where circles are to be observed and counted
    dataLoader = torch.utils.data.DataLoader(dataset_xyz[circle_dim], batch_size=batch_size, shuffle=True,
                                             num_workers=workers)

    # Define Network

    c_net = CNet().to(device)
    if ('cuda' in str(device)) and (ngpu > 1):
        c_net = nn.DataParallel(c_net, list(range(ngpu)))
    optC = optim.Adam(c_net.parameters(), lr=lrc, betas=(Beta1, Beta2))

    print("Starting CNet Training...")

    for e in range(numEpochs):

        closs_list = []
        for index, data_loader_tensors in enumerate(dataLoader):

            data_loader_tensor = data_loader_tensors[0].to(device)
            pred_OutR = c_net(data_loader_tensor).view(-1)

            util.test_plotter(data_loader_tensor, 1, 'twophase', project_path, True)

            R_img = cv2.imread(project_path + "_slices.png")

            detector = cv2.SimpleBlobDetector_create(params)

            keypoints = detector.detect(R_img)
            real_OutR = len(keypoints)

            if debug_flag:
                print_debug(data_loader_tensor, data_loader_tensors, pred_OutR, real_OutR)

            predR, realR = int(pred_OutR), int(real_OutR)

            c_net.zero_grad()
            cLoss = (pred_OutR - real_OutR) ** 2  # if pred_OutR > real_OutR else 0
            closs_list.append(cLoss)

            if (index % 5000 == 0):
                print('cLoss: ', cLoss)
                print('Pred Out R: ', pred_OutR)
                print(f"Epoch {e} : Slice {index} - NRC {realR} NPR {predR} Diff {predR - realR}\n")

            cLoss.backward()
            optC.step()

    cnet_weight_path = project_path + '/circleNet_weights.pt'
    torch.save(c_net.state_dict(), cnet_weight_path)

    try:
        int_closs_list = [int(cLossElement) for cLossElement in closs_list]
        temp_df = Df(int_closs_list)
        temp_df.to_csv(project_path + '/Circle_Loss.csv', encoding='utf-8', index=False)

        util.graph_plot([closs_list], ['CircleNet Loss'], project_path, 'CLossGraph')
    except Exception as e:
        print(f"Error! {str(e)} \nChange syntax for saving sheet or plotting graph at the trainCNet method in Circularity.py.")


def print_debug(data_loader_tensor, data_loader_tensors, pred_OutR, real_OutR):
    print('Predicted out R', pred_OutR)
    print(f"Type: {type(data_loader_tensor)} Size: {data_loader_tensor.size()}")
    print(f"Size DataLoader Tensor:\n {len(data_loader_tensors)}")
    print('Pred_outR_type', type(pred_OutR))
    print('Pred_outR', pred_OutR)

    print(type(real_OutR))
    print(real_OutR)


def CircleWeights(cnet, WeightPath, SL=bool(True)):
    """
    :param cnet: circlenet model
    :param WeightPath: Path to save or load weights
    :param SL: flag parameter to determine whether weights need to be saved or loaded
    :return:
    """

    cnet_weight_path = WeightPath + '/circleNet_weights.pt'
    if SL:
        print(WeightPath)
        for param_tensor in cnet().state_dict():
            print(param_tensor, "\t", cnet().state_dict()[param_tensor].size())
        torch.save(cnet().state_dict(), cnet_weight_path)
    else:

        cnet().load_state_dict(torch.load(cnet_weight_path))
        # for param_tensor in cnet().state_dict():
        #     print(param_tensor, "\t", cnet().state_dict()[param_tensor])
        return cnet


def numCircles(slice_i, area_find=3, MinArea=0, MaxArea=100):
    """
    :param slice_i: slice in which number of circles is to be calculated
    :param area_find: 1-> find and return min and max area; 2->filter by area; 3-> no filter, no find;
    :param MinArea: only used for calling in area_find=2
    :param MaxArea: only used for calling in area_find=2
    :return:
    """
    params = cv2.SimpleBlobDetector_Params()

    params.filterByArea = False
    params.filterByConvexity = False
    params.filterByInertia = False

    params.filterByCircularity = True
    params.minCircularity = 0.5

    if area_find == 1:

        ## We want to find max and min area of circles in the slice to be used later

        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(slice_i)

        print(len(keypoints))

        kmin, kmax = 0, 10000

        return len(keypoints), kmin, kmax

    elif area_find == 2:

        params.filterByArea = True
        params.minArea = MinArea
        params.maxArea = MaxArea

        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(slice_i)

        return len(keypoints)

    else:

        detector = cv2.SimpleBlobDetector_create(params)

        keypoints = detector.detect(slice_i)

        if area_find == 4:
            print(f"Number of detected circles is: {len(keypoints)}\nPress any key on the plot to continue.\n")

            im_with_keypoints = cv2.drawKeypoints(slice_i, keypoints, np.array([]), (0, 0, 255),

                                                  cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            cv2.imshow("Keypoints", im_with_keypoints)

            cv2.waitKey(0)

    return len(keypoints)


def CircularityLoss(imreal, imfake, CL_CNET):
    realcirc, fakecirc, diffcircL = [], [], []
    rlen, flen = 0, 0
    D = 0

    params = cv2.SimpleBlobDetector_Params()

    params.filterByArea = False
    params.filterByConvexity = False
    params.filterByInertia = False

    params.filterByCircularity = True
    params.minCircularity = 0.5

    for r in imreal:
        realcirc.append(CL_CNET(r))
        rlen += 1

    for f in imfake:
        fakecirc.append(CL_CNET(f))
        detector = cv2.SimpleBlobDetector_create(params)
        kpoints = detector.detect(f)
        # gg = len(kpoints)
        # print(f"Slice {f} has a difference of {CL_CNET(f) - gg} \n")
        flen += 1

    if rlen != flen:
        print("\n The number of real and fake slices do not match")
        return 0

    for i, R, F in enumerate(zip(realcirc, fakecirc)):
        diffcirc = ((F - R) ** 2) if R > F else 0  # 0 can also be substituted by int((R-F)**2)
        diffcircL.append(diffcirc)

        #print(f"Slice {i} has a difference of {diffcirc} circles between real and fake \n")

    for diff in diffcircL:
        D += diff

    return D / rlen
