import torch
import torch.nn as nn
import torch.nn.parallel
#import SNGAN.spectral_normalization as SpectralNorm
from SNGAN.sn_convolution_2d import SNConv2d


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv' or 'SNConv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
#DCGAN model, fully convolutional architecture
class _netG_1(nn.Module):
    def __init__(self, ngpu, nz, nc , ngf, n_extra_layers_g):
        super(_netG_1, self).__init__()
        self.ngpu = ngpu
        self.leak = 0.1
        #self.nz = nz
        #self.nc = nc
        #self.ngf = ngf
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=True),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            #nn.LeakyReLU(self.leak, inplace=True),
            # state size. (ndf) x 1 x 32
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=True),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            #nn.LeakyReLU(self.leak, inplace=True),
            # state size. (ndf) x 1 x 32
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=True),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            #nn.LeakyReLU(self.leak, inplace=True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=True),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            #nn.LeakyReLU(self.leak, inplace=True),
            # state size. (ngf) x 32 x 32
            #nn.ConvTranspose2d(ngf, nc, 3, 1, 1, bias=True),
            #nn.ConvTranspose2d(ngf, ngf, 3, 1, 1, bias=True), # extra layer to make G stronger
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=True),

            nn.Tanh()
            # state size. (nc) x 32 x 32
        )
    def forward(self, input):
        gpu_ids = None
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            gpu_ids = range(self.ngpu)
        return nn.parallel.data_parallel(self.main, input, gpu_ids), 0

    
class _netD_1(nn.Module):
    def __init__(self, ngpu, nz, nc, ndf):
        super(_netD_1, self).__init__()
        self.ngpu = ngpu
        self.leak = 0.2
        self.main = nn.Sequential(
            # input is (nc) x 32 x 32
            #SNConv2d()
#             SNConv2d(nc, ndf, 3, 1, 1, bias=True),
#             nn.LeakyReLU(0.1, inplace=True),
            SNConv2d(nc, ndf, 4, 2, 1, bias=True),
            nn.LeakyReLU(self.leak, inplace=True),
            # state size. (ndf) x 1 x 32

            SNConv2d(ndf, ndf * 2, 4, 2, 1, bias=True),
            #nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(self.leak, inplace=True),
            # state size. (ndf*2) x 16 x 16
 
            SNConv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=True),
            nn.LeakyReLU(self.leak, inplace=True),
            # state size. (ndf*8) x 4 x 4
            SNConv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=True),
            nn.LeakyReLU(self.leak, inplace=True),
            SNConv2d(ndf * 8, 1, 4, 1, 0, bias=True),
            nn.Sigmoid()
        )
        #self.snlinear = nn.Sequential(SNLinear(ndf * 4 * 4 * 4, 1),
        #                              nn.Sigmoid())


    def forward(self, input):
        gpu_ids = None
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            gpu_ids = range(self.ngpu)
        output = nn.parallel.data_parallel(self.main, input, gpu_ids)
        return output.view(-1, 1)



#SNGAN model, fully convolutional architecture
class _netG_2(nn.Module):
    def __init__(self, ngpu, nz, nc , ngf):
        super(_netG_2, self).__init__()
        self.ngpu = ngpu
        #self.nz = nz
        #self.nc = nc
        #self.ngf = ngf
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=True),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=True),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=True),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=True),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            #nn.ConvTranspose2d(ngf, nc, 3, 1, 1, bias=True),
            nn.ConvTranspose2d(ngf, ngf, 3, 1, 1, bias=True), # extra layer to make G stronger
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=True),

            nn.Tanh()
            # state size. (nc) x 32 x 32
        )
    def forward(self, input):
        gpu_ids = None
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            gpu_ids = range(self.ngpu)
        return nn.parallel.data_parallel(self.main, input, gpu_ids), 0

    
class _netD_2(nn.Module):
    def __init__(self, ngpu, nz, nc, ndf):
        super(_netD_2, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 32 x 32
            #SNConv2d()
            SNConv2d(nc, ndf, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            SNConv2d(ndf, ndf, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            # state size. (ndf) x 1 x 32
            SNConv2d(ndf, ndf * 2, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            SNConv2d(ndf* 2, ndf * 2, 4, 2, 1, bias=True),
            #nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.1, inplace=True),
            # state size. (ndf*2) x 16 x 16
            SNConv2d(ndf * 2, ndf * 4, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            SNConv2d(ndf * 4, ndf * 4, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            # state size. (ndf*8) x 4 x 4
            SNConv2d(ndf * 4, ndf * 8, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            SNConv2d(ndf * 8, ndf * 8, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            
            SNConv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        #self.snlinear = nn.Sequential(SNLinear(ndf * 4 * 4 * 4, 1),
        #                              nn.Sigmoid())


    def forward(self, input):
        gpu_ids = None
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            gpu_ids = range(self.ngpu)
        output = nn.parallel.data_parallel(self.main, input, gpu_ids)
        return output.view(-1, 1)


