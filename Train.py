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

def trainer(Proj,imtype,datatype,real_data, Disc, Gen, isotropic):
    if len(real_data) == 1:
        real_data*=3
    Proj = util.mkdr(Proj)
    pth = Proj + '/' + Proj

    datasetxyz = BatchMaker.Batch(real_data[0],real_data[1],real_data[2],datatype, TI = True)
    ## Constants for NNs
    ngpu=1
    batch_size=32
    l=64
    nc=3
    nz=64
    num_epochs=50
    lrg=0.0001
    lr =0.0002
    beta1 =0
    beta2 =0.9
    Lambda =10
    critic_iters = 5
    cudnn.benchmark = True
    workers = 0

    ##Dataloaders for each orientation
    device = torch.device("cuda:0" if(torch.cuda.is_available() and ngpu > 0) else "cpu")
    print(device, " will be used.\n")

    dataloaderx = torch.utils.data.DataLoader(datasetxyz[0], batch_size=batch_size,
                                             shuffle=True, num_workers=workers)
    dataloadery = torch.utils.data.DataLoader(datasetxyz[1], batch_size=batch_size,
                                             shuffle=True, num_workers=workers)
    dataloaderz = torch.utils.data.DataLoader(datasetxyz[2], batch_size=batch_size,
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
    for i in range(3):
        netD = Disc().apply(util.weights_init)
        if ('cuda' in str(device)) and (ngpu > 0):
            netD = (nn.DataParallel(netD, list(range(ngpu)))).to(device)
        netDs.append(netD)
        optDs.append(optim.Adam(netDs[i].parameters(), lr=lr, betas=(beta1, beta2)))

    disc_real=[]
    disc_fake=[]
    gp=[]

    print("Starting Training Loop...")
    # For each epoch
    start = time.time()
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, (datax, datay, dataz) in enumerate(zip(dataloaderx, dataloadery, dataloaderz),1):
            dataset = [datax,datay,dataz]
            ### Initialise
            DL_real = 0
            DL_fake = 0
            gp_log = 0
            ### Discriminator
            ## Generate fake image batch with G
            noise = torch.randn(batch_size, nz, 1, 1, 1, device=device)
            fake_data = netG(noise).detach()
            #For each dimension
            for dim, (netD, optimizer, data) in enumerate(zip(netDs, optDs, dataset)):
                if isotropic:
                    netD = netDs[0]
                    optimizer = optDs[0]
                for p in netD.parameters():  # reset requires_grad
                    p.requires_grad_(True)
                ##train on real images
                real_data = data[0].to(device)
                netD.zero_grad()
                # Forward pass real batch through D
                out_real = netD(real_data).view(-1).mean()
                DL_real += out_real.item()/3
                #train on fake images
                out_fake=0
                gradient_penalty = 0
                for lyr in range(l):
                    # Cycling through the planes of each orientation
                    if dim == 0:
                        fake_data_slice = fake_data[:, :, lyr, :, :]
                        output = netD(fake_data_slice).view(-1)
                    elif dim == 1:
                        fake_data_slice = fake_data[:, :, :, lyr, :]
                        output = netD(fake_data_slice).view(-1)
                    else:
                        fake_data_slice = fake_data[:, :, :, :, lyr]
                        output = netD(fake_data_slice).view(-1)
                    ## Update variables for this individual plane
                    # Calculate D's loss on the all-fake batch
                    out_fake += output.mean()/l
                    DL_fake += output.mean()/(3*l)
                    # Calculate the gradients for this batch
                    gradient_penalty += util.calc_gradient_penalty(netD, real_data, fake_data_slice, batch_size, l, device, Lambda,nc)/l
                disc_cost = out_fake - out_real + gradient_penalty
                disc_cost.backward()
                gp_log += gradient_penalty.item()
                optimizer.step()
            ### Generator Training
            if i % int(critic_iters) == 0:
                GL_Tot = 0  # Gen Loss (ideal 0)
                netG.zero_grad()
                errG=0
                noise = torch.randn(batch_size, nz, 1, 1, 1, device=device)
                noise.requires_grad_(True)
                fake = netG(noise)
                # For each dimension
                for dim, netD in enumerate(netDs):
                    if isotropic:
                        netD = netDs[0]
                    for p in netD.parameters():
                        p.requires_grad_(False)
                    #For each plane
                    for lyr in range(l):
                        # Pass through relevant discriminator
                        if dim==0:
                            output = netD(fake[:, :, lyr, :, :]).view(-1)
                        elif dim==1:
                            output = netD(fake[:, :, :, lyr, :]).view(-1)
                        else:
                            output = netD(fake[:, :, :, :, lyr]).view(-1)
                        #Calculate error for this plane
                        errG -= output.mean()
                        GL_Tot += output.mean()/(l*3)
                        # Calculate gradients for G
                errG.backward()
                optG.step()
                disc_real.append(DL_real)
                disc_fake.append(DL_fake)
            # Output training stats & show imgs
            if i % 25 == 0:
                gp.append(gp_log)
                torch.save(netG.state_dict(), pth + '_Gen.pt')
                torch.save(netD.state_dict(), pth + '_Disc.pt')
                noise = torch.randn(1, nz, 1, 1, 1, device = device)
                img = netG(noise).detach().cpu().numpy()
                ###Print progress
                ## calc ETA
                steps = len(dataloaderx)
                util.calc_ETA(steps,time.time(),start,i,epoch,num_epochs)
                ###save example slices
                sqrs = util.PostProc(img,imtype)
                util.Plotter(sqrs,5,imtype)
                plt.show()
                plt.savefig(pth +'slices.png')
                plt.close()
                #plotting graphs
                disc_real_arr = np.asarray(disc_real).reshape(int((i+(epoch*steps))/int(critic_iters)), -1).mean(axis=1)
                disc_fake_arr = np.asarray(disc_fake).reshape(int((i+(epoch*steps))/int(critic_iters)), -1).mean(axis=1)
                plt.figure()
                plt.plot(disc_real[::10])
                plt.plot(disc_fake[::10])
                plt.savefig(pth +'Lossgraph.png')
                plt.clf()
                plt.plot(disc_fake_arr-disc_real_arr)
                plt.savefig(pth + 'Wass_dist.png')
                plt.clf()
                plt.plot(gp)
                plt.savefig(pth + 'gp.png')
                plt.close()