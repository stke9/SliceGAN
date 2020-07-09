import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
def Architect(pth, Training, dk,ds,df,dp,gk,gs,gf,gp):
    #save params
    params = [dk, ds, df, dp, gk, gs, gf, gp]
    if Training:
        with open(pth + '_params.data', 'wb') as filehandle:
            # store the data as binary data stream
            pickle.dump(params, filehandle)
    else:
        with open(pth + '_params.data', 'rb') as filehandle:
            # read the data as binary data stream
            dk, ds, df, dp, gk, gs, gf, gp  = pickle.load(filehandle)
    # else:
    #     with open(pth + Proj + '/' + Proj + '_parameters.txt', 'w') as f:
    #Make nets
    class GeneratorWGAN(nn.Module):
        def __init__(self):
            super(GeneratorWGAN, self).__init__()
            self.convs = nn.ModuleList()
            self.bns = nn.ModuleList()
            for lay, (k,s,p) in enumerate(zip(gk,gs,gp)):
                self.convs.append(nn.ConvTranspose3d(gf[lay], gf[lay+1], k, s, p, bias=False))
                self.bns.append(nn.BatchNorm3d(gf[lay+1]))

        def forward(self, x):
            for conv,bn in zip(self.convs[:-1],self.bns[:-1]):
                x = F.relu_(bn(conv(x)))
            # out = torch.sigmoid(self.convs[-1](x))
            out = torch.softmax(self.convs[-1](x),1)
            return out
    class DiscriminatorWGAN(nn.Module):
        def __init__(self):
            super(DiscriminatorWGAN, self).__init__()
            self.convs = nn.ModuleList()
            for lay, (k, s, p) in enumerate(zip(dk, ds, dp)):
                self.convs.append(nn.Conv2d(df[lay], df[lay + 1], k, s, p, bias=False))

        def forward(self, x):
            for conv in self.convs[:-1]:
                x = F.relu_(conv(x))
            x = self.convs[-1](x)
            return x
    print('Architect Complete...')


    return DiscriminatorWGAN, GeneratorWGAN

