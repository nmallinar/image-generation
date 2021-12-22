import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.nn.parallel
from math import log2, pow

def SpectralConvTranspose2d(use_spectral_norm, *args, **kwargs):
    if use_spectral_norm:
        return spectral_norm(nn.ConvTranspose2d(*args, **kwargs))
    else:
        return nn.ConvTranspose2d(*args, **kwargs)

def SpectralConv2d(use_spectral_norm, *args, **kwargs):
    if use_spectral_norm:
        return spectral_norm(nn.Conv2d(*args, **kwargs))
    else:
        return nn.Conv2d(*args, **kwargs)

class Generator(nn.Module):
    def __init__(self, nz, ngf, nc, ngpu, image_size=64, leaky_relu_slope=0.1, act_fn='relu',
                 use_spectral_norm=False):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.nz = nz
        self.ngf = ngf
        modules = []
        resolution = log2(image_size)
        assert resolution.is_integer()
        resolution = int(resolution)

        num_layers = resolution - 2
        curr_in_fmaps = nz
        for idx in range(num_layers-1, -1, -1):
            if idx == num_layers-1:
                modules.append(SpectralConvTranspose2d(use_spectral_norm, curr_in_fmaps, ngf * int(pow(2, idx)), 4, 1, 0, bias=False))
            else:
                modules.append(SpectralConvTranspose2d(use_spectral_norm, curr_in_fmaps, ngf * int(pow(2, idx)), 4, 2, 1, bias=False))

            modules.append(nn.BatchNorm2d(ngf * int(pow(2, idx))))

            if act_fn == 'leaky_relu':
                modules.append(nn.LeakyReLU(leaky_relu_slope, inplace=True))
            elif act_fn == 'selu':
                modules.append(nn.SELU(inplace=True))
            elif act_fn == 'relu':
                modules.append(nn.ReLU(inplace=True))
            else:
                raise ValueError(f'act_fn: {act_fn} is invalid.')

            curr_in_fmaps = ngf * int(pow(2, idx))

        modules += [
            SpectralConvTranspose2d(use_spectral_norm, curr_in_fmaps, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        ]

        self.main = nn.Sequential(*modules)

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

class Discriminator(nn.Module):
    def __init__(self, nc, ndf, ngpu, image_size=64, leaky_relu_slope=0.1, act_fn='leaky_relu',
                 use_spectral_norm=False):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.nc = nc
        self.ndf = ndf
        modules = []
        resolution = log2(image_size)
        assert resolution.is_integer()
        resolution = int(resolution)

        num_layers = resolution - 2
        curr_in_fmaps = nc
        for idx in range(0, num_layers):
            if idx == 0:
                modules.append(SpectralConv2d(use_spectral_norm, curr_in_fmaps, ndf*int(pow(2, idx)), 4, 2, 1, bias=False))
            else:
                modules += [
                    SpectralConv2d(use_spectral_norm, curr_in_fmaps, ndf*int(pow(2, idx)), 4, 2, 1, bias=False),
                    nn.BatchNorm2d(ndf * int(pow(2, idx)))
                ]

            if act_fn == 'leaky_relu':
                modules.append(nn.LeakyReLU(leaky_relu_slope, inplace=True))
            elif act_fn == 'selu':
                modules.append(nn.SELU(inplace=True))
            else:
                raise ValueError(f'act_fn: {act_fn} is invalid.')

            curr_in_fmaps = ndf*int(pow(2, idx))

        modules += [
            SpectralConv2d(use_spectral_norm, curr_in_fmaps, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        ]
        self.main = nn.Sequential(*modules)

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)

class Generator128(nn.Module):
    def __init__(self, nz, ngf, nc, ngpu, image_size=64):
        super(Generator128, self).__init__()
        self.ngpu = ngpu
        self.nz = nz
        self.ngf = ngf
        modules = []
        resolution = log2(image_size)
        assert isinstance(resolution, int)

        num_layers = resolution - 2
        curr_in_fmaps = nz
        for idx in range(num_layers-1, -1, -1):
            if idx == num_layers-1:
                modules += [
                    nn.ConvTranspose2d(curr_in_fmaps, ngf * (2^idx), 4, 1, 0, bias=False),
                    nn.BatchNorm2d(ngf * (2^idx)),
                    nn.LeakyReLU(True)
                ]
            else:
                modules += [
                    nn.ConvTranspose2d(curr_in_fmaps, ngf * (2^idx), 4, 2, 1, bias=False),
                    nn.BatchNorm2d(ngf * (2^idx)),
                    nn.LeakyReLU(True)
                ]
            curr_in_fmaps = ngf * (2^idx)

        modules += [
            nn.ConvTranspose2d(curr_in_fmaps, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        ]

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.LeakyReLU(True),
            # state size. (ngf*16) x 4 x 4
            nn.ConvTranspose2d(     ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(True),
            # state size. (ngf*8) x 8 x 8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(True),
            # state size. (ngf*4) x 16 x 16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(True),
            # state size. (ngf*2) x 32 x 32
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(True),
            # state size. (ngf) x 64 x 64
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 128 x 128
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

class Generator64(nn.Module):
    def __init__(self, nz, ngf, nc, ngpu):
        super(Generator64, self).__init__()
        self.ngpu = ngpu
        self.nz = nz
        self.ngf = ngf
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

class Discriminator128(nn.Module):
    def __init__(self, nc, ndf, ngpu):
        super(Discriminator128, self).__init__()
        self.ngpu = ngpu
        self.nc = nc
        self.ndf = ndf
        self.main = nn.Sequential(
            # input is (nc) x 128 x 128
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 64 x 64
            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 32 x 32
            nn.Conv2d(ndf*2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 16 x 16
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 8 x 8
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*16) x 4 x 4
            nn.Conv2d(ndf * 16, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)

class Discriminator64(nn.Module):
    def __init__(self, nc, ndf, ngpu):
        super(Discriminator64, self).__init__()
        self.ngpu = ngpu
        self.nc = nc
        self.ndf = ndf
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
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
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)
