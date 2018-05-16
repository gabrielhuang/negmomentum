import argparse
import os
import random
import json
import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

from utils import find_latest_file


# Enums
NONSATURATING = 'nonsaturating'
ZEROSUM = 'zerosum'
WASSERSTEIN = 'wasserstein'
NOPENALTY = 'nopenalty'
REALFAKE = 'realfake'

parser = argparse.ArgumentParser()

# Folder to export everything
parser.add_argument('logdir', help='folder to output images and model checkpoints')

# Dataset
parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw | fake')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)

# DCGAN
parser.add_argument('--iterations', type=int, default=100000, help='input batch size')
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--formulation', default=NONSATURATING, choices=[ZEROSUM, NONSATURATING, WASSERSTEIN], help='GAN formulation')
parser.add_argument('--gp', default=10., type=float, help='magnitude of gradient penalty')
parser.add_argument('--gp-type', default=NOPENALTY, choices=[NOPENALTY, REALFAKE], help='type of gradient penalty')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64, help='size of generator')
parser.add_argument('--ndf', type=int, default=64, help='size of discriminator')
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lrG', type=float, default=0.0002, help='generator learning rate, default=0.0002')
parser.add_argument('--lrD', type=float, default=0.0002, help='discriminator learning rate, default=0.0002')
parser.add_argument('--beta1G', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--beta1D', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--batchnormD', type=int, default=0, help='use batchnorm for discriminator')

# General
parser.add_argument('--cuda', type=int, default=1, help='enables cuda')
parser.add_argument('--checkpoint', default='', help='start from another checkpoint (use logdir normally)')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--sample-every', default=100, type=int, help='save image interval')
parser.add_argument('--checkpoint-every', default=100, type=int, help='checkpoint')
parser.add_argument('--log-every', default=1, type=int, help='print log interval')
parser.add_argument('--start-iteration', default=0, type=int, help='start counting iterations from that')


args = parser.parse_args()
print(args)


# Set experiment folders
args_filename = os.path.join(args.logdir, 'args.json')
run_dir = args.logdir
check_dir = os.path.join(run_dir, 'checkpoint')
sample_dir = os.path.join(run_dir, 'sample')

# By default, continue training
# Check if args.json exists
if os.path.exists(args_filename):
    print 'Attempting to resume training. (Delete {} to start over)'.format(run_dir)
    # Resuming training is incompatible with other checkpoint
    # than the last one in logdir
    assert args.checkpoint == '', 'Cannot load other checkpoint when resuming training.'
    # Attempt to find checkpoint in logdir
    args.checkpoint = run_dir
else:
    print 'No previous training found. Starting fresh.'
    # Otherwise, create experiment folders
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    if not os.path.exists(check_dir):
        os.makedirs(check_dir)
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
    # Write args to args.json
    with open(args_filename, 'wb') as fp:
        json.dump(vars(args), fp, indent=4)


# Create tensorboard logger
logger = SummaryWriter(run_dir)


if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
print("Random Seed: ", args.manualSeed)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

if args.dataset in ['imagenet', 'folder', 'lfw']:
    # folder dataset
    dataset = dset.ImageFolder(root=args.dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(args.imageSize),
                                   transforms.CenterCrop(args.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
elif args.dataset == 'lsun':
    dataset = dset.LSUN(root=args.dataroot, classes=['bedroom_train'],
                        transform=transforms.Compose([
                            transforms.Resize(args.imageSize),
                            transforms.CenterCrop(args.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
elif args.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=args.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Resize(args.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
elif args.dataset == 'fake':
    dataset = dset.FakeData(image_size=(3, args.imageSize, args.imageSize),
                            transform=transforms.ToTensor())
assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batchSize,
                                         shuffle=True, num_workers=int(args.workers))
def make_infinite(dataloader):
    while True:
        for data in dataloader:
            yield data
data_iter = make_infinite(dataloader)

device = torch.device("cuda:0" if args.cuda else "cpu")
ngpu = int(args.ngpu)
nz = int(args.nz)
ngf = int(args.ngf)
ndf = int(args.ndf)
nc = 3


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
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



class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
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


class DiscriminatorNoBatchnorm(Discriminator):

    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0),
            nn.Sigmoid()
        )


# Create models
netG = Generator(ngpu).to(device)
netG.apply(weights_init)
print(netG)
if args.batchnormD:
    netD = Discriminator(ngpu).to(device)
else:
    netD = DiscriminatorNoBatchnorm(ngpu).to(device)
netD.apply(weights_init)
print(netD)

# Create optimizers
optimizerD = optim.Adam(netD.parameters(), lr=args.lrD, betas=(args.beta1D, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=args.lrG, betas=(args.beta1G, 0.999))

# Checkpoint is directory -> Find last model or '' if does not exist
if os.path.isdir(args.checkpoint):
    latest_checkpoint = find_latest_file(check_dir)
    if latest_checkpoint:
        print 'Latest checkpoint found:', latest_checkpoint
        args.checkpoint = os.path.join(check_dir, latest_checkpoint)
    else:
        args.checkpoint = ''

# Start fresh
if args.checkpoint == '':
    print 'No checkpoint. Starting fresh'
    fixed_noise = torch.randn(args.batchSize, nz, 1, 1, device=device)
# Load file
elif os.path.isfile(args.checkpoint):
    print 'Attempting to load checkpoint', args.checkpoint
    checkpoint = torch.load(args.checkpoint)
    netD.load_state_dict(checkpoint['netD'])
    netG.load_state_dict(checkpoint['netG'])
    optimizerD.load_state_dict(checkpoint['optimizerD'])
    optimizerG.load_state_dict(checkpoint['optimizerG'])
    args.start_iteration = checkpoint['iteration']
    fixed_noise = checkpoint['fixed_noise']
    info = checkpoint['info']
    print '-> Success.'
else:
    raise ArgumentError('Bad checkpoint. Delete logdir folder to start over.')


def get_gradient_penalty(real, fake, gp_type):
    # compute optional gradient penalty
    if gp_type == NOPENALTY:
        penalty = torch.zeros((), device=device)
    elif gp_type == REALFAKE:
        # Recompute - maybe this is not necessary?
        real = real.detach().requires_grad_()
        fake = fake.detach().requires_grad_()
        D_real = netD(real)
        D_fake = netD(fake)

        # maybe we could save those two computations; not sure ...
        # need torch.autograd.grad because create_graph breaks with backward
        fake_gradients = torch.autograd.grad(
            outputs=D_fake, inputs=fake,
            grad_outputs=torch.ones_like(D_fake).to(device),
            create_graph=True, only_inputs=True)[0]
        real_gradients = torch.autograd.grad(
            outputs=D_real, inputs=real,
            grad_outputs=torch.ones_like(D_real).to(device),
            create_graph=True, only_inputs=True)[0]

        real_penalty = (real_gradients ** 2).sum() / float(len(real))
        fake_penalty = (fake_gradients ** 2).sum() / float(len(fake))
        penalty = real_penalty + fake_penalty

    return penalty


def track(name, value):
    info.setdefault(name, OrderedDict())
    info[name][iteration] = value

    just_loaded = (iteration == args.start_iteration and args.checkpoint != '')
    if iteration % args.log_every == 0 and not just_loaded:
        logger.add_scalar(name, value, iteration)
        print '{}: {:.4f}'.format(name, value)


criterion = nn.BCELoss()
info = {}
for iteration in xrange(args.start_iteration, args.iterations):

    # Sample real
    data, __ = data_iter.next()
    real = data.to(device)
    batch_size = len(real)
    real_label = torch.full((batch_size,), 1, device=device)
    fake_label = torch.full((batch_size,), 0, device=device)

    # Generate fake
    noise = torch.randn(batch_size, nz, 1, 1, device=device)
    fake = netG(noise)

    ############################
    # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
    ###########################

    # Classify
    D_real = netD(real)
    D_fake = netD(fake.detach())  # we don't want to backprop into netG

    # Get classification loss
    D_real_loss = criterion(D_real, real_label)
    D_fake_loss = criterion(D_fake, fake_label)
    D_classification = D_real_loss + D_fake_loss

    # Get penalty
    D_gradient_penalty = args.gp * get_gradient_penalty(real, fake, args.gp_type)

    # Total loss
    D_loss = D_classification + D_gradient_penalty

    # Backprop
    netD.zero_grad()
    D_loss.backward()
    optimizerD.step()

    ############################
    # (2) Update G network: maximize log(D(G(z)))
    ###########################

    # Classify again (fake is not detached this time)
    D_fake2 = netD(fake)

    # Get classification (total) loss
    G_classification = criterion(D_fake2, real_label)  # generator wants to be labeled as real
    G_loss = G_classification

    # Backprop through D and G
    netG.zero_grad()
    G_loss.backward()
    optimizerG.step()

    ##############################
    # Debug and logging
    ##############################
    if iteration % args.log_every == 0:
        print '\nIteration', iteration

    # Log training values
    track('loss/D_loss', D_loss.item())
    track('loss/D_classification', D_classification.item())
    track('loss/D_gradient_penalty', D_gradient_penalty.item())
    track('loss/G_loss', G_loss.item())
    track('stats/D_real', D_real.mean().item())
    track('stats/D_fake', D_fake.mean().item())
    track('stats/D_fake2', D_fake2.mean().item())

    just_loaded = (iteration == args.start_iteration and args.checkpoint != '')


    if iteration % args.sample_every == 0 and not just_loaded:
        vutils.save_image(real, os.path.join(sample_dir, 'real_{:09d}.png'.format(iteration)), normalize=True)
        fake = netG(fixed_noise)
        vutils.save_image(fake.detach(), os.path.join(sample_dir, 'fake_{:09d}.png'.format(iteration)), normalize=True)

        gallery_real = vutils.make_grid(real.data, normalize=True, range=(-1, 1))
        gallery_fake = vutils.make_grid(fake.data, normalize=True, range=(-1, 1))
        logger.add_image('real', gallery_real, iteration)
        logger.add_image('fake', gallery_fake, iteration)
        print 'Saving samples to tensorboard'

    if iteration % args.checkpoint_every == 0 and not just_loaded:

        checkpoint = {
            'netD': netD.state_dict(),
            'netG': netG.state_dict(),
            'optimizerD': optimizerD.state_dict(),
            'optimizerG': optimizerG.state_dict(),
            'iteration': iteration,
            'info': info,
            'fixed_noise': fixed_noise
        }
        checkpoint_path = os.path.join(check_dir, 'checkpoint_{}.pth'.format(iteration))
        print 'Saving checkpoint to', checkpoint_path
        torch.save(checkpoint, checkpoint_path)
