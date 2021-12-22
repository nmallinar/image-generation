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

def get_upsample_layer(use_spectral_norm, use_convtranspose2d, *args, **kwargs):
    if use_convtranspose2d:
        return [SpectralConvTranspose2d(use_spectral_norm, *args, **kwargs)]
    else:
        return [
            nn.Upsample(scale_factor=2, mode='nearest'),
            SpectralConv2d(use_spectral_norm, *args, **kwargs)
        ]

class Generator(nn.Module):
    def __init__(self, nz, ngf, nc, ngpu, image_size=64, leaky_relu_slope=0.1, act_fn='relu',
                 use_spectral_norm=False, use_convtranspose2d=False):
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
                modules += get_upsample_layer(use_spectral_norm, use_convtranspose2d, curr_in_fmaps, ngf * int(pow(2, idx)), 4, 1, 0, bias=False)
            else:
                modules += get_upsample_layer(use_spectral_norm, use_convtranspose2d, curr_in_fmaps, ngf * int(pow(2, idx)), 4, 2, 1, bias=False)

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

        modules += get_upsample_layer(use_spectral_norm, use_convtranspose2d, curr_in_fmaps, nc, 4, 2, 1, bias=False)
        modules += [nn.Tanh()]

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
            elif act_fn == 'relu':
                modules.append(nn.ReLU(inplace=True))
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
