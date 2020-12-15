from slicegan import preprocessing, util
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import time
import matplotlib

def train(pth, imtype, datatype, real_data, Disc, Gen, nc, l, nz, sf):

    if len(real_data) == 1:
        real_data *= 3
        isotropic = True
    else:
        isotropic = False

    print('Loading Dataset...')
    dataset_xyz = preprocessing.batch(real_data, datatype, l, sf,TI=True)

    ## Constants for NNs
    matplotlib.use('Agg')
    ngpu = 1
    num_epochs = 30

    # batch sizes
    batch_size = 32
    D_batch_size = 8
    # optimiser params for G and D
    lrg = 0.0001
    lrd = 0.0001
    beta1 = 0
    beta2 = 0.9
    Lambda = 10
    critic_iters = 5
    cudnn.benchmark = True
    workers = 0
    lz = 4
    ##Dataloaders for each orientation
    device = torch.device("cuda:0" if(torch.cuda.is_available() and ngpu > 0) else "cpu")
    print(device, " will be used.\n")

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

    print("Starting Training Loop...")
    # For each epoch
    start = time.time()
    for epoch in range(num_epochs):
        # sample data for each direction
        for i, (datax, datay, dataz) in enumerate(zip(dataloaderx, dataloadery, dataloaderz), 1):
            dataset = [datax, datay, dataz]
            ### Initialise
            ### Discriminator
            ## Generate fake image batch with G
            noise = torch.randn(D_batch_size, nz, lz,lz,lz, device=device)
            fake_data = netG(noise).detach()
            # for each dim (d1, d2 and d3 are used as permutations to make 3D volume into a batch of 2D images)
            for dim, (netD, optimizer, data, d1, d2, d3) in enumerate(
                    zip(netDs, optDs, dataset, [2, 3, 4], [3, 2, 2], [4, 4, 3])):
                if isotropic:
                    netD = netDs[0]
                    optimizer = optDs[0]
                netD.zero_grad()
                ##train on real images
                real_data = data[0].to(device)
                out_real = netD(real_data).view(-1).mean()
                ## train on fake images
                # perform permutation + reshape to turn volume into batch of 2D images to pass to D
                fake_data_perm = fake_data.permute(0, d1, 1, d2, d3).reshape(l * D_batch_size, nc, l, l)
                out_fake = netD(fake_data_perm).mean()
                gradient_penalty = util.calc_gradient_penalty(netD, real_data, fake_data_perm[:batch_size],
                                                                      batch_size, l,
                                                                      device, Lambda, nc)
                disc_cost = out_fake - out_real + gradient_penalty
                disc_cost.backward()
                optimizer.step()
            #logs for plotting
            disc_real_log.append(out_real.item())
            disc_fake_log.append(out_fake.item())
            Wass_log.append(out_real.item() - out_fake.item())
            gp_log.append(gradient_penalty.item())
            ### Generator Training
            if i % int(critic_iters) == 0:
                netG.zero_grad()
                errG = 0
                st = time.time()
                noise = torch.randn(batch_size, nz, lz,lz,lz, device=device)
                fake = netG(noise)
                print('gen', time.time()-st)
                st = time.time()

                for dim, (netD, d1, d2, d3) in enumerate(
                        zip(netDs, [2, 3, 4], [3, 2, 2], [4, 4, 3])):
                    if isotropic:
                        #only need one D
                        netD = netDs[0]
                    # permute and reshape to feed to disc
                    fake_data_perm = fake.permute(0, d1, 1, d2, d3).reshape(l * batch_size, nc, l, l)
                    output = netD(fake_data_perm)
                    errG -= output.mean()
                    # Calculate gradients for G
                errG.backward()
                optG.step()
                print('back', time.time() - st)

            # Output training stats & show imgs
            if i % 25 == 0:
                torch.save(netG.state_dict(), pth + '_Gen.pt')
                torch.save(netD.state_dict(), pth + '_Disc.pt')
                noise = torch.randn(1, nz,lz,lz,lz, device=device)
                img = netG(noise)
                ###Print progress
                ## calc ETA
                steps = len(dataloaderx)
                util.calc_ETA(steps, time.time(), start, i, epoch, num_epochs)
                ###save example slices
                util.TestPlotter(img, 5, imtype, pth)
                # plotting graphs
                util.GraphPlot([disc_real_log, disc_fake_log], ['real', 'perp'], pth, 'LossGraph')
                util.GraphPlot([Wass_log], ['Wass Distance'], pth, 'WassGraph')
                util.GraphPlot([gp_log], ['Gradient Penalty'], pth, 'GpGraph')
