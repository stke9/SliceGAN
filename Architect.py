import torch
import torch.nn as nn
import torch.nn.functional as F

def Architect(dl,dk,ds,ndfs, gl,gk,gs,ngfs):

    g_pad = []
    d_pad = []

    for i in range(len(gk)):
            pad = ((gl[i] - 1) * gs[i] + 1 + (gk[i] - 1) - gl[i+1]) / 2
            g_pad.append(int(pad))
    for i in range(len(dk)):
            pad = ((dl[i+1] - 1) * ds[i] + 1 + (dk[i]-1) - dl[i])/2
            d_pad.append(int(pad))

    #Make nets
    class GeneratorWGAN(nn.Module):
        def __init__(self):
            super(GeneratorWGAN, self).__init__()
            self.convs = nn.ModuleList()
            self.bns = nn.ModuleList()
            for lay, (k,s,p) in enumerate(zip(gk,gs,g_pad)):
                self.convs.append(nn.ConvTranspose3d(ngfs[lay], ngfs[lay+1], k, s, p, bias=False))
                self.bns.append(nn.BatchNorm3d(ngfs[lay+1]))

        def forward(self, x):
            for conv,bn in zip(self.convs[:-1],self.bns[:-1]):
                x = F.relu_(bn(conv(x)))
            out = torch.sigmoid(self.convs[-1](x))
            return out
    print (g_pad, d_pad)
    class DiscriminatorWGAN(nn.Module):
        def __init__(self):
            super(DiscriminatorWGAN, self).__init__()
            self.convs = nn.ModuleList()
            for lay, (k, s, p) in enumerate(zip(dk, ds, d_pad)):
                self.convs.append(nn.Conv2d(ndfs[lay], ndfs[lay + 1], k, s, p, bias=False))

        def forward(self, x):
            for conv in self.convs[:-1]:
                x = F.relu_(conv(x))
            x = self.convs[-1](x)
            return x
    assert sum(n < 0 for n in g_pad) == 0, 'Negative value in Generator Padding: ' + str(g_pad) + \
                                      'Reduce the difference in layer sizes or increase stride'
    assert sum(n < 0 for n in d_pad) == 0, 'Negative value in Discriminator Padding: ' + str(d_pad) +  \
                                      'Reduce the difference in layer sizes or increase stride'

    return DiscriminatorWGAN, GeneratorWGAN
