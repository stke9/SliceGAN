import torch.nn as nn
import torch.nn.functional as F
import torch

# Define the Generator Network
class GeneratorAGLa(nn.Module):
    def __init__(self, nz, nc, ngf, ngpu):
        super().__init__()
        self.ngpu = ngpu
        # Input is the latent vector Z.
        self.tconv1 = nn.ConvTranspose3d(nz, ngf * 8,
                                         kernel_size=4, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm3d(ngf * 8)

        # Input Dimension: (ngf*8) x 4 x 4
        self.tconv2 = nn.ConvTranspose3d(ngf * 8, ngf * 4,
                                         4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm3d(ngf * 4)

        # Input Dimension: (ngf*4) x 8 x 8
        self.tconv3 = nn.ConvTranspose3d(ngf * 4, ngf * 2,
                                         4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm3d(ngf * 2)

        # Input Dimension: (ngf*2) x 16 x 16
        self.tconv4 = nn.ConvTranspose3d(ngf * 2, ngf,
                                         4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm3d(ngf)

        # Input Dimension: (ngf) * 32 * 32
        self.tconv5 = nn.ConvTranspose3d(ngf, nc,
                                         4, 2, 1, bias=False)
        # Output Dimension: (nc) x 64 x 64

    def forward(self, x):
        device = torch.device("cuda:0")
        x = F.relu(self.bn1(self.tconv1(x)))
        print('1', x.shape)
        cpad = 3
        b_size, layers, H, W, L = x.shape
        image = torch.zeros(b_size, layers, H + 2 * cpad, W + 2 * cpad, L + 2 * cpad).to(device)
        # fill x into middle
        image[:, :, cpad:H + cpad, cpad:W + cpad, cpad:L + cpad] = x
        # H padding
        image[:, :, :cpad, :, :] = image[:, :, H:H + cpad, :, :]
        image[:, :, H + cpad:H + 2 * cpad, :, :] = image[:, :, cpad:2 * cpad, :, :]
        # W padding
        image[:, :, :, :cpad, :] = image[:, :, :, W:W + cpad, :]
        image[:, :, :, W + cpad:W + 2 * cpad, :] = image[:, :, :, cpad:2 * cpad, :]
        # L padding
        image[:, :, :, :, :cpad] = image[:, :, :, :, L:L + cpad]
        image[:, :, :, :, L + cpad:L + 2 * cpad] = image[:, :, :, :, cpad:2 * cpad]
        x = F.relu(self.bn2(self.tconv2(image)))
        #        cpad = 3
        #        b_size, layers, H, W, L = x.shape
        #        image = torch.zeros(b_size, layers, H + 2*cpad, W + 2*cpad, L + 2*cpad)
        #        image[:,:,cpad:H+cpad, cpad:W+cpad, cpad:L+cpad] = x
        #        #H padding
        #        image[:, :, 0:cpad, :, :] = image[:, :, H:H+cpad, :, :]
        #        image[:, :, H+cpad:H+2*cpad, :, :] = image[:, :, cpad:2*cpad, :, :]
        #        #W padding
        #        image[:, :, :, 0:cpad, :] = image[:, :, :, W:W+cpad, :]
        #        image[:, :, :, W+cpad:W+2*cpad, :] = image[:, :, :, cpad:2*cpad, :]
        #        #L padding
        #        image[:, :, :, :, 0:cpad] = image[:, :, :, :, L:L+cpad]
        #        image[:, :, :, :, L+cpad:L+2*cpad] = image[:, :, :, :, cpad:2*cpad]
        print(x.shape)

        x = F.relu(self.bn3(self.tconv3(x)))
        #        cpad = 3
        #        b_size, layers, H, W, L = x.shape
        #        image = torch.zeros(b_size, layers, H + 2*cpad, W + 2*cpad, L + 2*cpad)
        #        image[:,:,cpad:H+cpad, cpad:W+cpad, cpad:L+cpad] = x
        #        #H padding
        #        image[:, :, 0:cpad, :, :] = image[:, :, H:H+cpad, :, :]
        #        image[:, :, H+cpad:H+2*cpad, :, :] = image[:, :, cpad:2*cpad, :, :]
        #        #W padding
        #        image[:, :, :, 0:cpad, :] = image[:, :, :, W:W+cpad, :]
        #        image[:, :, :, W+cpad:W+2*cpad, :] = image[:, :, :, cpad:2*cpad, :]
        #        #L padding
        #        image[:, :, :, :, 0:cpad] = image[:, :, :, :, L:L+cpad]
        #        image[:, :, :, :, L+cpad:L+2*cpad] = image[:, :, :, :, cpad:2*cpad]
        #
        print(x.shape)

        x = F.relu(self.bn4(self.tconv4(x)))
        #        cpad = 3
        #        b_size, layers, H, W, L = x.shape
        #        image = torch.zeros(b_size, layers, H + 2*cpad, W + 2*cpad, L + 2*cpad)
        #        image[:,:,cpad:H+cpad, cpad:W+cpad, cpad:L+cpad] = x
        #        #H padding
        #        image[:, :, 0:cpad, :, :] = image[:, :, H:H+cpad, :, :]
        #        image[:, :, H+cpad:H+2*cpad, :, :] = image[:, :, cpad:2*cpad, :, :]
        #        #W padding
        #        image[:, :, :, 0:cpad, :] = image[:, :, :, W:W+cpad, :]
        #        image[:, :, :, W+cpad:W+2*cpad, :] = image[:, :, :, cpad:2*cpad, :]
        #        #L padding
        #        image[:, :, :, :, 0:cpad] = image[:, :, :, :, L:L+cpad]
        #        image[:, :, :, :, L+cpad:L+2*cpad] = image[:, :, :, :, cpad:2*cpad]
        print(x.shape)

        x = F.softmax(self.tconv5(x), dim=1)
        print('5',x.shape)

        return x

class GeneratorAGLb(nn.Module):
    def __init__(self, nz, nc, ngf, ngpu):
        super().__init__()
        self.ngpu = ngpu
        # Input is the latent vector Z.
        self.tconv1 = nn.ConvTranspose3d(nz, ngf * 8,
                                         kernel_size=4, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm3d(ngf * 8)

        # Input Dimension: (ngf*8) x 4 x 4
        self.tconv2 = nn.ConvTranspose3d(ngf * 8, ngf * 4,
                                         4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm3d(ngf * 4)

        # Input Dimension: (ngf*4) x 8 x 8
        self.tconv3 = nn.ConvTranspose3d(ngf * 4, ngf * 2,
                                         4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm3d(ngf * 2)

        # Input Dimension: (ngf*2) x 16 x 16
        self.tconv4 = nn.ConvTranspose3d(ngf * 2, ngf,
                                         4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm3d(ngf)

        # Input Dimension: (ngf) * 32 * 32
        self.tconv5 = nn.ConvTranspose3d(ngf, nc,
                                         4, 2, 1, bias=False)
        # Output Dimension: (nc) x 64 x 64

    def forward(self, x, padding, addpad):
        x = F.relu(self.bn1(self.tconv1(x)))
        ##Create correct 3D volume

        cpad = 4
        b_size, layers, H, W, L = x.shape
        image = torch.zeros(b_size, layers, H + 2 * cpad, W + 2 * cpad, L + 2 * cpad)
        # fill x into middle
        image[:, :, cpad:H + cpad, cpad:W + cpad, cpad:L + cpad] = x
        # H padding
        image[:, :, :cpad, :, :] = image[:, :, H:H + cpad, :, :]
        image[:, :, H + cpad:H + 2 * cpad, :, :] = image[:, :, cpad:2 * cpad, :, :]
        # W padding
        image[:, :, :, :cpad, :] = image[:, :, :, W:W + cpad, :]
        image[:, :, :, W + cpad:W + 2 * cpad, :] = image[:, :, :, cpad:2 * cpad, :]
        # L padding
        image[:, :, :, :, :cpad] = image[:, :, :, :, L:L + cpad]
        image[:, :, :, :, L + cpad:L + 2 * cpad] = image[:, :, :, :, cpad:2 * cpad]
        ## Calculate padding
        padhf = image[:, :, cpad:2 * cpad, :, :]
        padhb = image[:, :, H:H + cpad, :, :]
        padwf = image[:, :, :, cpad:2 * cpad, :]
        padwb = image[:, :, :, W:W + cpad, :]
        padlf = image[:, :, :, :, L:L + cpad]
        padlb = image[:, :, :, :, cpad:2 * cpad]
        if addpad!=False:
            y = F.relu(self.bn1(self.tconv1( torch.FloatTensor(1, 100, 1, 1,1).normal_(0, 1))))
            image[:, :, cpad:H + cpad, cpad:W + cpad, cpad:L + cpad] = y
        # if addpad == 76:
        #     image = x
        # else:
        #     image[:, :, cpad:H + cpad, cpad:W + cpad, cpad:L + cpad] = x
        #     b_size, layers, H, W, L = x.shape
        #     image = torch.zeros(b_size, layers, H + cpad * 2, W + cpad * 2, L + cpad * 2)
        #     # fill x into middle
        #     # image[:, :, cpad:H + cpad, cpad:W + cpad, cpad:L + cpad] = x
        #     # # H padding
        #     # image[:, :, :cpad, :, :] = torch.clone(padding[1])
        #     # image[:, :, H + cpad:H + 2 * cpad, :, :] = torch.clone(padding[0])
        #     # # W padding
        #     # image[:, :, :, :cpad, :] = torch.clone(padding[3])
        #     # image[:, :, :, W + cpad:W + 2 * cpad, :] = torch.clone(padding[2])
        #     # # L padding
        #     # image[:, :, :, :, :cpad] = torch.clone(padding[5])
        #     # image[:, :, :, :, L + cpad:L + 2 * cpad] = torch.clone(padding[4])
        #     image[:, :, :cpad, :, :] = padhb
        #     image[:, :, H + cpad:H + 2 * cpad, :, :] = padhf
        #     # W padding
        #     image[:, :, :, :cpad, :] = padwb
        #     image[:, :, :, W + cpad:W + 2 * cpad, :] = padwf
        #     # L padding
        #     image[:, :, :, :, :cpad] = padlb
        #     image[:, :, :, :, L + cpad:L + 2 * cpad] = padlf
        padding = [padhf, padhb, padwf, padwb, padlf, padlb]





        x = F.relu(self.bn2(self.tconv2(image)))
        #        cpad = 3
        #        b_size, layers, H, W, L = x.shape
        #        image = torch.zeros(b_size, layers, H + 2*cpad, W + 2*cpad, L + 2*cpad)
        #        image[:,:,cpad:H+cpad, cpad:W+cpad, cpad:L+cpad] = x
        #        #H padding
        #        image[:, :, 0:cpad, :, :] = image[:, :, H:H+cpad, :, :]
        #        image[:, :, H+cpad:H+2*cpad, :, :] = image[:, :, cpad:2*cpad, :, :]
        #        #W padding
        #        image[:, :, :, 0:cpad, :] = image[:, :, :, W:W+cpad, :]
        #        image[:, :, :, W+cpad:W+2*cpad, :] = image[:, :, :, cpad:2*cpad, :]
        #        #L padding
        #        image[:, :, :, :, 0:cpad] = image[:, :, :, :, L:L+cpad]
        #        image[:, :, :, :, L+cpad:L+2*cpad] = image[:, :, :, :, cpad:2*cpad]

        x = F.relu(self.bn3(self.tconv3(x)))
        #        cpad = 3
        #        b_size, layers, H, W, L = x.shape
        #        image = torch.zeros(b_size, layers, H + 2*cpad, W + 2*cpad, L + 2*cpad)
        #        image[:,:,cpad:H+cpad, cpad:W+cpad, cpad:L+cpad] = x
        #        #H padding
        #        image[:, :, 0:cpad, :, :] = image[:, :, H:H+cpad, :, :]
        #        image[:, :, H+cpad:H+2*cpad, :, :] = image[:, :, cpad:2*cpad, :, :]
        #        #W padding
        #        image[:, :, :, 0:cpad, :] = image[:, :, :, W:W+cpad, :]
        #        image[:, :, :, W+cpad:W+2*cpad, :] = image[:, :, :, cpad:2*cpad, :]
        #        #L padding
        #        image[:, :, :, :, 0:cpad] = image[:, :, :, :, L:L+cpad]
        #        image[:, :, :, :, L+cpad:L+2*cpad] = image[:, :, :, :, cpad:2*cpad]
        #

        x = F.relu(self.bn4(self.tconv4(x)))
        #        cpad = 3
        #        b_size, layers, H, W, L = x.shape
        #        image = torch.zeros(b_size, layers, H + 2*cpad, W + 2*cpad, L + 2*cpad)
        #        image[:,:,cpad:H+cpad, cpad:W+cpad, cpad:L+cpad] = x
        #        #H padding
        #        image[:, :, 0:cpad, :, :] = image[:, :, H:H+cpad, :, :]
        #        image[:, :, H+cpad:H+2*cpad, :, :] = image[:, :, cpad:2*cpad, :, :]
        #        #W padding
        #        image[:, :, :, 0:cpad, :] = image[:, :, :, W:W+cpad, :]
        #        image[:, :, :, W+cpad:W+2*cpad, :] = image[:, :, :, cpad:2*cpad, :]
        #        #L padding
        #        image[:, :, :, :, 0:cpad] = image[:, :, :, :, L:L+cpad]
        #        image[:, :, :, :, L+cpad:L+2*cpad] = image[:, :, :, :, cpad:2*cpad]

        x = F.softmax(self.tconv5(x), dim=1)

        return x,padding

class Generator(nn.Module):
    def __init__(self,  nz, nc, ngf, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose3d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm3d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*4) x 16 x 16
            nn.ConvTranspose3d(ngf*8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ngf * 4),
            nn.ReLU(True),
            # state
            nn.ConvTranspose3d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*4) x 16 x 16
            nn.ConvTranspose3d( ngf * 2, nc, 4, 2, 1, bias=False),
            nn.Sigmoid()
            # state size. (nc) x 20 x 20
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, nc, ndf, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x l x l
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, 1, 4, 2, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

class Generatorsf(nn.Module):
    def __init__(self,  nz, nc, ngf, ngpu):
        super(Generatorsf, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose3d( nz, ngf * 8, 4, 2, 0, bias=False),
            nn.BatchNorm3d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*4) x 16 x 16
            nn.ConvTranspose3d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ngf * 4),
            nn.ReLU(True),
            # state
            nn.ConvTranspose3d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ngf * 2),
            nn.ReLU(True),
            #state size
            nn.ConvTranspose3d(ngf * 2, ngf, 6, 2, 2, bias=False),
            nn.BatchNorm3d(ngf),
            nn.ReLU(True),
            # state size. (ngf*4) x 16 x 16
            nn.ConvTranspose3d( ngf, nc, 4, 2, 1, bias=False),
            #nn.Sigmoid()
            nn.Softmax(dim=1)
            # state size. (nc) x 20 x 20
        )

    def forward(self, input):
        return self.main(input)

class Discriminatorsf(nn.Module):
    def __init__(self, nc, ndf, ngpu):
        super(Discriminatorsf, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x l x l
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 8, 1, 4, 2, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

class GeneratorWGAN(nn.Module):
    def __init__(self,  nz, nc, ngf, ngpu):
        super(GeneratorWGAN, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose3d( nz, ngf * 8, 4, 2, 0, bias=False),
            nn.BatchNorm3d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*4) x 16 x 16
            nn.ConvTranspose3d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ngf * 4),
            nn.ReLU(True),
            # state
            nn.ConvTranspose3d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ngf * 2),
            nn.ReLU(True),
            #state size
            nn.ConvTranspose3d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ngf),
            nn.ReLU(True),
            # state size. (ngf*4) x 16 x 16
            nn.ConvTranspose3d(ngf, nc, 4, 2, 1, bias=False),
            #nn.Sigmoid()
            nn.Sigmoid()

            # state size. (nc) x 20 x 20
        )

    def forward(self, input):
        return self.main(input)

class DiscriminatorWGAN(nn.Module):
    def __init__(self, nc, ndf, ngpu):
        super(DiscriminatorWGAN, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x l x l
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 8, 1, 4, 2, 0, bias=False),
        )

    def forward(self, input):
        return self.main(input)

class GeneratorWGAN2D(nn.Module):
    def __init__(self, nz, nc, ngf, ngpu):
        super(GeneratorWGAN2D, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 2, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*4) x 16 x 16
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size
            nn.ConvTranspose2d(ngf , nc, 4, 2, 1, bias=False),
            # nn.Sigmoid()
            nn.Tanh()

            # state size. (nc) x 20 x 20
        )

    def forward(self, input):
        return self.main(input)

class DiscriminatorWGAN2D(nn.Module):
    def __init__(self, nc, ndf, ngpu):
        super(DiscriminatorWGAN2D, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(

            # state size. (ndf) x 32 x 32
            nn.Conv2d(nc, ndf * 2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 8, ndf*16, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 16, 1, 4, 2, 0, bias=False),
        )

    def forward(self, input):
        return self.main(input)

class GeneratorWGAN2Dl(nn.Module):
    def __init__(self, nz, nc, ngf, ngpu):
        super(GeneratorWGAN2Dl, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 2, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*4) x 16 x 16
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf*4) x 16 x 16
            nn.ConvTranspose2d(ngf, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.Conv2d(ngf, ngf, 9, 1, 4, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # input is (nc) x l x l
            nn.Conv2d(ngf, ngf, 17, 1, 8, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.Conv2d(ngf, nc, 33, 1, 16, bias=False),
            nn.Softmax(1)
        )

    def forward(self, input):
        return self.main(input)

class DiscriminatorWGAN2Dl(nn.Module):
    def __init__(self, nc, ndf, ngpu):
        super(DiscriminatorWGAN2Dl, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x l x l
            nn.Conv2d(nc, ndf, 33, 1, 16, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # input is (nc) x l x l
            nn.Conv2d(ndf, ndf, 17, 1, 8, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # input is (nc) x l x l
            nn.Conv2d(ndf, ndf, 9, 1, 4, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # input is (nc) x l x l
            nn.Conv2d(ndf, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 8, 1, 4, 2, 0, bias=False),
        )

    def forward(self, input):
        return self.main(input)

class GeneratorWGAN3Dl(nn.Module):
    def __init__(self, nz, nc, ngf, ngpu):
        super(GeneratorWGAN3Dl, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose3d(nz, ngf * 8, 4, 2, 0, bias=False),
            nn.BatchNorm3d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*4) x 16 x 16
            nn.ConvTranspose3d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ngf * 4),
            nn.ReLU(True),
            # state
            nn.ConvTranspose3d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ngf * 2),
            nn.ReLU(True),
            # state size
            nn.ConvTranspose3d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm3d(ngf),
            nn.ReLU(True),
            # state size
            nn.ConvTranspose3d(ngf, nc, 5, 1, 2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

class DiscriminatorWGAN3Dl(nn.Module):
    def __init__(self, nc, ndf, ngpu):
        super(DiscriminatorWGAN3Dl, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # state size. (ndf) x 32 x 32
            nn.Conv2d(nc, ndf, 5, 1, 2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 8, 1, 4, 2, 0, bias=False),
        )

    def forward(self, input):
        return self.main(input)