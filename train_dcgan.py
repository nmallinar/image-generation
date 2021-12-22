'''
Originally sourced from: https://github.com/pytorch/examples/tree/master/dcgan
'''

from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

from models.dcgan_base import Generator, Discriminator
from datasets import get_default_datasets

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | lsun | mnist |imagenet | folder | lfw | fake')
parser.add_argument('--dataroot', required=False, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--sampleBatchSize', type=int, default=64, help='image samples batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr_disc', type=float, default=0.00005, help='discriminiator learning rate, default=0.000005')
parser.add_argument('--lr_gen', type=float, default=0.0002, help='generator learning rate, default=0.00002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--dry-run', action='store_true', help='check a single training cycle works')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--model-save-freq', type=int, dest='model_save_freq', default=10, help='frequency in epochs to save models')
parser.add_argument('--image-save-freq', type=int, dest='image_save_freq', default=100, help='frequency in batches per epoch to save image samples')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--classes', default='bedroom', help='comma separated list of classes for the lsun data set')
parser.add_argument('--grad_accumulate', default=1, type=int, help='gradient accumulation factor in train steps')
parser.add_argument('--use_spectral_norm', default=False, action='store_true', help='use spectral normalization on Conv2d and ConvTranspose2d layers')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf, exist_ok=True)
    os.makedirs(os.path.join(opt.outf, 'models'), exist_ok=True)
    os.makedirs(os.path.join(opt.outf, 'samples'), exist_ok=True)
except OSError:
    pass

models_outdir = os.path.join(opt.outf, 'models')
samples_outdir = os.path.join(opt.outf, 'samples')

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

dataset, nc = get_default_datasets(opt.dataset, opt.dataroot, opt.imageSize, opt.classes)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

device = torch.device("cuda:0" if opt.cuda else "cpu")
ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


netG = Generator(nz, ngf, nc, ngpu, image_size=opt.imageSize,
                 leaky_relu_slope=0.2, act_fn='leaky_relu',
                 use_spectral_norm=opt.use_spectral_norm).to(device)
netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)


netD = Discriminator(nc, ndf, ngpu, image_size=opt.imageSize,
                     leaky_relu_slope=0.2, act_fn='leaky_relu',
                     use_spectral_norm=opt.use_spectral_norm).to(device)
netD.apply(weights_init)

if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

criterion = nn.BCELoss()

fixed_noise = torch.randn(opt.sampleBatchSize, nz, 1, 1, device=device)
real_label = 1
fake_label = 0

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr_disc, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr_gen, betas=(opt.beta1, 0.999))

if opt.dry_run:
    opt.niter = 1

for epoch in range(opt.niter):
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        netD.zero_grad()
        real_cpu = data[0].to(device)
        batch_size = real_cpu.size(0)
        label = torch.full((batch_size,), real_label,
                           dtype=real_cpu.dtype, device=device)

        output = netD(real_cpu)
        errD_real = criterion(output, label)

        if opt.grad_accumulate > 1:
            errD_real = errD_real / opt.grad_accumulate

        errD_real.backward()
        D_x = output.mean().item()

        # train with fake
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake = netG(noise)
        label.fill_(fake_label)
        output = netD(fake.detach())
        errD_fake = criterion(output, label)

        if opt.grad_accumulate > 1:
            errD_fake = errD_fake / opt.grad_accumulate

        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake

        if ((i+1) % opt.grad_accumulate == 0) or (i+1 == len(dataloader)) or opt.grad_accumulate == 1:
            optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        output = netD(fake)
        errG = criterion(output, label)

        if opt.grad_accumulate > 1:
            errG = errG / opt.grad_accumulate

        errG.backward()
        D_G_z2 = output.mean().item()

        if ((i+1) % opt.grad_accumulate == 0) or (i+1 == len(dataloader)) or opt.grad_accumulate == 1:
            optimizerG.step()

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, opt.niter, i, len(dataloader),
                 errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
        if i % 100 == 0:
            if opt.sampleBatchSize < opt.batchSize:
                vutils.save_image(real_cpu[:opt.sampleBatchSize],
                        '%s/real_samples.png' % samples_outdir,
                        normalize=True)
            else:
                vutils.save_image(real_cpu,
                        '%s/real_samples.png' % samples_outdir,
                        normalize=True)

            fake = netG(fixed_noise)
            vutils.save_image(fake.detach(),
                    '%s/fake_samples_epoch_%03d.png' % (samples_outdir, epoch),
                    normalize=True)

        if opt.dry_run:
            break

    # do checkpointing
    if (epoch+1) % opt.model_save_freq == 0:
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (models_outdir, epoch+1))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (models_outdir, epoch+1))

torch.save(netG.state_dict(), '%s/netG_final.pth' % (models_outdir))
torch.save(netD.state_dict(), '%s/netD_final.pth' % (models_outdir))
