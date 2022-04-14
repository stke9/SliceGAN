import numpy as np
from slicegan import preprocessing, util, Circularity
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import time
import matplotlib
import cv2

# Noise distributions that can be used as seeds for the generator
# Feel free to add more stuff here !

noise_distributions = {
    "normal" : torch.distributions.normal.Normal(0,1),
    "laplace" : torch.distributions.laplace.Laplace(0,1),
    "uniform" : torch.distributions.uniform.Uniform(-1,1),
    "cauchy": torch.distributions.cauchy.Cauchy(0,1),
    "exponential": torch.distributions.exponential.Exponential(1)

}


def train(pth, imtype, datatype, real_data, Disc, Gen, nc, l, nz, sf, lz, num_epochs, CircNet, use_Circ = 0, noise_type = "normal", sub_images = 32*900, beta1=0, beta2=0.9):
    """
    train the generator
    :param pth: path to save all files, imgs and data
    :param imtype: image type e.g nphase, colour or gray
    :param datatype: training data format e.g. tif, jpg ect
    :param real_data: path to training data
    :param Disc:
    :param Gen:
    :param nc: channels
    :param l: image size
    :param nz: latent vector size
    :param sf: scale factor for training data
    :param CircNet: Trained CircleNet
    :param beta1: beta1 for Adam optimizer
    :param beta2: beta2 for Adam optimizer
    :return:
    """
    cnet_weight_path = pth + '/circleNet_weights.pt'

    print(pth)
    print('beta1: ', beta1)
    print('beta2', beta2)
        
    try:
        noise_distribution = noise_distributions[noise_type]
    except:
        raise ValueError("invalid noise distribution")

    if len(real_data) == 1:
        real_data *= 3
        isotropic = True
    else:
        isotropic = False

    # batch sizes
    # batch_size = 32
    batch_size = 16
    D_batch_size = 8
    img_per_batch = 900
    N_images = batch_size * img_per_batch

    print('Loading Dataset...')
    dataset_xyz = preprocessing.batch(real_data, datatype, l, sf, N_images)

    ## Constants for NNs
    matplotlib.use('Agg')
    ngpu = 1
    # num_epochs = 30


    # optimiser params for G and D
    
    lrg = 0.0001
    lrd = 0.0001
    # change values of beta1 between 0.1-0.9, beta2 0.9-0.99 and calc evals?
    Lambda = 10
    critic_iters = 5
    cudnn.benchmark = True
    workers = 0
    # lz = 4
    # The value of lz is now calculated and passed as a computed parameter

    circularity_loss = 0

    ##Dataloaders for each orientation
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    print(device, " will be used.\n")

    if use_Circ:
        cnet = CircNet.to(device)
        cnet.load_state_dict(torch.load(cnet_weight_path))

    # D trained using different data for x, y and z directions
    dataloaderx = torch.utils.data.DataLoader(dataset_xyz[0], batch_size=batch_size,
                                              shuffle=True, num_workers=workers)
    dataloadery = torch.utils.data.DataLoader(dataset_xyz[1], batch_size=batch_size,
                                              shuffle=True, num_workers=workers)
    dataloaderz = torch.utils.data.DataLoader(dataset_xyz[2], batch_size=batch_size,
                                              shuffle=True, num_workers=workers)

    # Create the Genetator network
    netG = Gen().to(device)
    if ('cuda' in str(device)) and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))
    optG = optim.Adam(netG.parameters(), lr=lrg, betas=(beta1, beta2))

    # Define 1 Discriminator and optimizer for each plane in each dimension
    netDs = []
    optDs = []
    for i in range(3):
        netD = Disc()
        netD = (nn.DataParallel(netD, list(range(ngpu)))).to(device)
        netDs.append(netD)
        optDs.append(optim.Adam(netDs[i].parameters(), lr=lrd, betas=(beta1, beta2)))

    disc_real_log = []
    disc_fake_log = []
    gp_log = []
    Wass_log = []

    if use_Circ != 0:
        circ_loss_log = []

    # gk, gs, gp = netG.params[4,5,7]

    print("Starting Training Loop...")
    # For each epoch
    print(dataloaderx, len(dataloaderx))
    print(dataloadery, len(dataloadery))
    print(dataloaderz, len(dataloaderz))
    start = time.time()
    for epoch in range(num_epochs):
        # sample data for each direction
        
        for i, (datax, datay, dataz) in enumerate(zip(dataloaderx, dataloadery, dataloaderz), 1):
            print(i)
            dataset = [datax, datay, dataz]
            ### Initialise
            ### Discriminator
            ## Generate fake image batch with G
            # noise = torch.randn(D_batch_size, nz, lz, lz, lz, device=device)
            noise = noise_distribution.sample((D_batch_size, nz, lz, lz, lz)).to(device)
            fake_data = netG(noise).detach()

            realcirc, fakecirc, diffcircL = [], [], []
            rlen, flen = 0, 0

            # for each dim (d1, d2 and d3 are used as permutations to make 3D volume into a batch of 2D images)
            for dim, (netD, optimizer, data, d1, d2, d3) in enumerate(
                    zip(netDs, optDs, dataset, [2, 3, 4], [3, 2, 2], [4, 4, 3])):
                if isotropic:
                    netD = netDs[0]
                    optimizer = optDs[0]
                netD.zero_grad()
                ##train on real images
                real_data = data[0].to(device)
                for r in real_data:
                    realcirc.append(cnet(r))
                    rlen += 1
                out_real = netD(real_data).view(-1).mean()
                ## train on fake images
                # perform permutation + reshape to turn volume into batch of 2D images to pass to D
                # Round n
                    # 0 2 1 3 4
                    # 0 3 1 2 4
                    # 0 4 1 2 3
                fake_data_perm = fake_data.permute(0, d1, 1, d2, d3).reshape(l * D_batch_size, nc, l, l)
                out_fake = netD(fake_data_perm).mean()
                gradient_penalty = util.calc_gradient_penalty(netD, real_data, fake_data_perm[:batch_size],
                                                              batch_size, l,
                                                              device, Lambda, nc)

                disc_cost = (out_fake - out_real) + gradient_penalty

                disc_cost.backward()
                optimizer.step()

            # logs for plotting
            disc_real_log.append(out_real.item())
            disc_fake_log.append(out_fake.item())
            Wass_log.append(out_real.item() - out_fake.item())
            gp_log.append(gradient_penalty.item())
            ### Generator Training
            if i % int(critic_iters) == 0:
                netG.zero_grad()
                errG = torch.zeros(1).to(device)
                # noise = torch.randn(batch_size, nz, lz, lz, lz, device=device)
                noise = noise_distribution.sample((batch_size, nz, lz, lz, lz)).to(device)

                fake = netG(noise)

                for dim, (netD, d1, d2, d3) in enumerate(
                        zip(netDs, [2, 3, 4], [3, 2, 2], [4, 4, 3])):
                    if isotropic:
                        # only need one D
                        netD = netDs[0]
                    # permute and reshape to feed to disc
                    fake_data_perm = fake.permute(0, d1, 1, d2, d3).reshape(l * batch_size, nc, l, l)
                    output = netD(fake_data_perm)
                    errG -= output.mean()
                    if use_Circ and dim == 0 and (CircNet is not None):
                        ## If dimension along x, calculate circularity_loss
                        # circularity_loss = Circularity.CircularityLoss(data, fake_data_perm, CircNet)

                        params = cv2.SimpleBlobDetector_Params()

                        params.filterByArea = False
                        params.filterByConvexity = False
                        params.filterByInertia = False

                        params.filterByCircularity = True
                        params.minCircularity = 0.5

                        for f in fake_data_perm:
                            fakecirc.append(cnet(f))
                            # detector = cv2.SimpleBlobDetector_create(params)
                            # kpoints = detector.detect(f)
                            # gg = len(kpoints)
                            # print(f"Slice {f} has a difference of {CircNet(f) - gg} \n")
                            flen += 1

                        # if rlen != flen:
                            # print("\n The number of real and fake slices do not match")

                        for itt, (R, F) in enumerate(zip(realcirc, fakecirc), 1):
                            diffcirc = ((F - R) ** 2) # 0 can also be substituted by int((R-F)**2)
                            diffcircL.append(diffcirc)

                            #print(f"Slice {itt} has a difference of {diffcirc} circles between real and fake \n")

                        D = torch.zeros(1).to(device)

                        for diff in diffcircL:
                            D += diff.view(-1)

                        circularity_loss = D / rlen

                        errG += circularity_loss
                        circ_loss_log.append(circularity_loss.item())
                        # print(f"Circularity Loss for iteration {i} is {circularity_loss}\n Type of cLoss: {type(circularity_loss)}\n")
                        # Calculate gradients for G
                errG.backward()
                optG.step()
            

            # Output training stats & show imgs
            if i % 25 == 0:
                # Put model into evaluation mode
                netG.eval()
                with torch.no_grad():
                    torch.save(netG.state_dict(), pth + '_Gen.pt')
                    torch.save(netD.state_dict(), pth + '_Disc.pt')
                    # noise = torch.randn(1, nz, lz, lz, lz, device=device)
                    noise = noise_distribution.sample((D_batch_size, nz, lz, lz, lz)).to(device)
                    img = netG(noise)
                    ###Print progress
                    ## calc ETA
                    steps = len(dataloaderx)
                    util.calc_eta(steps, time.time(), start, i, epoch, num_epochs)
                    ###save example slices
                    util.test_plotter(img, 5, imtype, pth)
                    # plotting graphs
                    util.graph_plot([disc_real_log, disc_fake_log], ['real', 'perp'], pth, 'LossGraph')
                    util.graph_plot([Wass_log], ['Wass Distance'], pth, 'WassGraph')
                    util.graph_plot([gp_log], ['Gradient Penalty'], pth, 'GpGraph')
                    # store logs
                    np.save(pth + '_disc_real_log.npy', disc_real_log)
                    np.save(pth + '_disc_fake_log.npy', disc_fake_log)
                    np.save(pth + '_wass_log.npy', Wass_log)
                    np.save(pth + '_gp_log.npy', gp_log)

                # Put model into training mode
                netG.train()


def check_conv_vals(k, s, p):
    if s < k and k % s == 0 and p >= k - s:
        return 1
    else:
        return 0


def lz_img_size_converter(gk, gs, gp, img_size = None, lz= None, lz_to_im = False):
    # im_size-> original image size
    # gk, gs, gp-> generator kernel size, stride, and padding
    if not lz_to_im:
        gk, gs, gp = gk[::-1], gs[::-1], gp[::-1]
        # because, we want to go in reverse to calculate lz from image size
    
    if lz_to_im:
        if lz == None:
            raise ValueError("No lz noise dimension given")
        
        for lay, (k, s, p) in enumerate(zip(gk, gs, gp)):

            if lay == 0:
                next_im = lz

            # \/ equation to calculate size in next layer given transpose convolution (reverse this to find lz)
            next_im = k + ((next_im - 1) * s) - 2 * p

            next_im = int(next_im)
         
        return next_im

    else:
        if img_size == None:
            raise ValueError("No imagesize given")
        
        for lay, (k, s, p) in enumerate(zip(gk, gs, gp)):

            ch = check_conv_vals(k, s, p)
            if ch == 0:
                raise ValueError("Values not compatible for uniform information density in the generator samples.")
            if lay == 0:
                next_l = img_size

            # \/ equation to calculate size in next layer given transpose convolution (reverse this to find lz)
            # next_l = k + ((l - 1) * s) - 2 * p
            # next_l - k + 2 * p = (l - 1) * s
            # l = ((next_l - k + 2 * p) / s) + 1
            next_l = ((next_l - k + 2 * p)/s) + 1
            next_l = int(next_l)

        return next_l