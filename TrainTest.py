import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
#import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt
import BatchMaker
import time
import util as util
import matplotlib

def trainer(pth,imtype,datatype,real_data, Disc, Gen, isotropic, nc, l, nz):
    if len(real_data) == 1: real_data*=3

    print('Loading Dataset...')
    datasetxyz = BatchMaker.Batch(real_data[0],real_data[1],real_data[2],datatype,l, TI = True)
    matplotlib.use('Agg')

    ## Constants for NNs
    ngpu=1
    G_batch_size=8
    D_batch_size = 4
    num_epochs=500
    lrg=0.0004
    lr =0.0001
    beta1 =0
    beta2 =0.9
    Lambda =10
    critic_iters = 5
    cudnn.benchmark = True
    workers = 0

    ##Dataloaders for each orientation
    device = torch.device("cuda:0" if(torch.cuda.is_available() and ngpu > 0) else "cpu")

    dataloaderx = torch.utils.data.DataLoader(datasetxyz[0], batch_size=l*D_batch_size,
                                             shuffle=True, num_workers=workers)


    # Create the Genetator network
    netG = Gen().to(device)
    netG.apply(util.weights_init)
    if('cuda' in str(device)) and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))
    optG = optim.Adam(netG.parameters(), lr=lrg, betas=(beta1, beta2))

    # Define 1 discriminator and optimizer for each plane in each dimension
    netDs = []
    optDs = []
    for dim in range(2):
        netD = Disc().apply(util.weights_init)
        if ('cuda' in str(device)) and (ngpu > 0):
            netD = (nn.DataParallel(netD, list(range(ngpu)))).to(device)
        optD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, beta2))
        netDs.append(netD)
        optDs.append(optD)
    disc_real_log=[]
    disc_fake_log=[]
    disc_ang_log=[[],[]]
    Wass_log = []
    Gen_log=[]
    gp_log=[]

    print('Starting Training Loop...')
    # For each epoch
    start = time.time()
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(dataloaderx,1):
            out_fake,gradient_penalty =0,0
            ### Discriminator
            ##Real pass
            real_data = data[0].to(device)
            out_real = netD(real_data).mean()
            ##Fake pass
            noise = torch.randn(D_batch_size, nz, 4, 4, 4, device=device)
            fake_data = netG(noise).detach()
            for dim, (netD, d1, d2) in enumerate(zip(netDs,[2,3],[3,2])):
                for p in netD.parameters():  # reset requires_grad
                    p.requires_grad_(True)
                netD.zero_grad()
                ##train on real images
                #train on fake images
                fake_data_p = fake_data.permute(0,d1,1,d2,4).reshape(l*D_batch_size,nc,l,l)
                fake_data_ang = util.angslcs(fake_data, l, D_batch_size, dim)
                disc_ang_log[dim].append(netD(fake_data_ang).mean().item())
                for fake in [fake_data_p,fake_data_ang]:
                    out_fake += netD(fake).mean()
                    gradient_penalty += util.calc_gradient_penalty(netD, real_data, fake, l * D_batch_size, l,
                                                                   device, Lambda, nc)

            wd = out_fake - out_real
            disc_cost = wd + gradient_penalty
            disc_cost.backward()
            for optD in optDs:
                optD.step()

            ### Update logs
            disc_real_log.append(out_real.item())
            disc_fake_log.append(out_fake.item())
            gp_log.append(gradient_penalty.item())

            ### Generator Training
            if i % int(critic_iters) == 1:
                netG.zero_grad()
                errG=0
                noise = torch.randn(G_batch_size, nz, 4,4,4, device=device)
                fake_data = netG(noise)
                # For each dimension
                for dim, (netD, d1, d2) in enumerate(zip(netDs, [2, 3], [3, 2])):
                    fake_data_p = fake_data.permute(0, d1, 1, d2, 4).reshape(l*G_batch_size, nc, l, l)
                    fake_data_ang = util.angslcs(fake_data, l, G_batch_size,dim)
                    errG -= netD(fake_data_p).mean()
                    errG -= netD(fake_data_ang).mean()
                errG.backward()
                optG.step()
                Wass_log.append(wd.item())
                Gen_log.append(errG.item())
            # Output training stats & show imgs
            if i % 25 == 0:
                ## calc ETA
                util.calc_ETA(len(dataloaderx), time.time(), start, i, epoch, num_epochs)
                #save nets
                torch.save(netG.state_dict(), pth + '_Gen.pt')
                torch.save(netD.state_dict(), pth + '_Disc.pt')
                #plot test images
                noise = torch.randn(1, nz, 4,4,4, device = device)
                img = netG(noise)
                util.TestPlotter(img,5,imtype,pth)
                #plotting graphs
                util.GraphPlot([disc_real_log,disc_fake_log, disc_ang_log[0],disc_ang_log[1]],['real','perp','ang1','ang2'], pth, 'LossGraph')
                util.GraphPlot([Wass_log, Gen_log], ['Wass Distance', 'Generator Loss'], pth, 'WassGraph')
                util.GraphPlot([gp_log], ['Gradient Penalty'], pth, 'GpGraph')

