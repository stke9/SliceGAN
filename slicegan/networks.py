import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
def slicegan_nets(pth, Training, imtype, dk,ds,df,dp,gk,gs,gf,gp):
    """
    Define a generator and Discriminator
    :param Training: If training, we save params, if not, we load params from previous.
    This keeps the parameters consistent for older models
    :return:
    """
    #save params
    params = [dk, ds, df, dp, gk, gs, gf, gp]
    # if fresh training, save params
    if Training:
        with open(pth + '_params.data', 'wb') as filehandle:
            # store the data as binary data stream
            pickle.dump(params, filehandle)
    # if loading model, load the associated params file
    else:
        with open(pth + '_params.data', 'rb') as filehandle:
            # read the data as binary data stream
            dk, ds, df, dp, gk, gs, gf, gp  = pickle.load(filehandle)


    # Make nets
    class Generator(nn.Module):
        def __init__(self):
            super(Generator, self).__init__()
            self.convs = nn.ModuleList()
            self.bns = nn.ModuleList()
            for lay, (k,s,p) in enumerate(zip(gk,gs,gp)):
                self.convs.append(nn.ConvTranspose3d(gf[lay], gf[lay+1], k, s, p, bias=False))
                self.bns.append(nn.BatchNorm3d(gf[lay+1]))

        def forward(self, x):
            for conv,bn in zip(self.convs[:-1],self.bns[:-1]):
                x = F.relu_(bn(conv(x)))
            #use tanh if colour or grayscale, otherwise softmax for one hot encoded
            if imtype in ['grayscale', 'colour']:
                out = 0.5*(torch.tanh(self.convs[-1](x))+1)
            else:
                out = torch.softmax(self.convs[-1](x),1)
            return out

    class Discriminator(nn.Module):
        def __init__(self):
            super(Discriminator, self).__init__()
            self.convs = nn.ModuleList()
            for lay, (k, s, p) in enumerate(zip(dk, ds, dp)):
                self.convs.append(nn.Conv2d(df[lay], df[lay + 1], k, s, p, bias=False))

        def forward(self, x):
            for conv in self.convs[:-1]:
                x = F.relu_(conv(x))
            x = self.convs[-1](x)
            return x

    return Discriminator, Generator

