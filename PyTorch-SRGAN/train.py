#!/usr/bin/env python

import argparse
import os
import sys
import numpy as np

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
from torch.autograd import Variable

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from tensorboard_logger import configure, log_value

from models import Generator, Discriminator, FeatureExtractor
#from utils import Visualizer
import torchvision.utils as vutils
from PIL import Image,ImageFilter
from scipy.misc import imsave

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='anime-faces', help='cifar10 | cifar100 | folder')
parser.add_argument('--dataroot', type=str, default='../anime-faces', help='path to dataset')
parser.add_argument('--workers', type=int, default=12, help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
parser.add_argument('--imageSize', type=int, default=32, help='the low resolution image size')
parser.add_argument('--upSampling', type=int, default=2, help='low to high resolution scaling factor')
parser.add_argument('--nEpochs', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--generatorLR', type=float, default=0.0001, help='learning rate for generator')
parser.add_argument('--discriminatorLR', type=float, default=0.0001, help='learning rate for discriminator')
parser.add_argument('--cuda', default=True, help='enables cuda')
parser.add_argument('--nGPU', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--generatorWeights', type=str, default='', help="path to generator weights (to continue training)")
parser.add_argument('--discriminatorWeights', type=str, default='', help="path to discriminator weights (to continue training)")
parser.add_argument('--out', type=str, default='checkpoints', help='folder to output model checkpoints')

opt = parser.parse_args()
print(opt)
opt.cuda = True
try:
    os.makedirs(opt.out)
except OSError:
    pass


transform = transforms.Compose([#transforms.RandomCrop(opt.imageSize*opt.upSampling),
                                 transforms.Scale(opt.imageSize*opt.upSampling),
                                transforms.ToTensor()])

normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                std = [0.229, 0.224, 0.225])

scale = transforms.Compose([transforms.ToPILImage(),
                            transforms.Scale(opt.imageSize),
                            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                            transforms.ToTensor(),
                            transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                std = [0.229, 0.224, 0.225])
                            ])


dataset = datasets.ImageFolder(root=opt.dataroot, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

##########################################################################################
# load weights
##########################################################################################
generator = Generator(16, opt.upSampling)
discriminator = Discriminator()
# if opt.generatorWeights != '':
#     generator.load_state_dict(torch.load(opt.generatorWeights))
# print(generator)
# if opt.discriminatorWeights != '':
#     discriminator.load_state_dict(torch.load(opt.discriminatorWeights))
# print(discriminator)

##########################################################################################
# For the content loss
feature_extractor = FeatureExtractor(torchvision.models.vgg19(pretrained=True))
print(feature_extractor)
content_criterion = nn.MSELoss()
adversarial_criterion = nn.BCELoss()

ones_const = Variable(torch.ones(opt.batchSize, 1))

# if gpu is to be used
print("gpu:", opt.cuda)
if opt.cuda:
    generator.cuda()
    discriminator.cuda()
    feature_extractor.cuda()
    content_criterion.cuda()
    adversarial_criterion.cuda()
    ones_const = ones_const.cuda()

optim_generator = optim.Adam(generator.parameters(), lr=opt.generatorLR)
optim_discriminator = optim.Adam(discriminator.parameters(), lr=opt.discriminatorLR)

# configure('logs/' + opt.dataset + '-' + str(opt.batchSize) + '-' + str(opt.generatorLR) + '-' + str(opt.discriminatorLR), flush_secs=5)
#visualizer = Visualizer(image_size=opt.imageSize*opt.upSampling)##############################

low_res = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)

# Pre-train generator using raw MSE loss
print('Generator pre-training')
# for epoch in range(2):
#     mean_generator_content_loss = 0.0

#     for i, data in enumerate(dataloader):
#         # Generate data
#         high_res_real, _ = data
        
#         if len(high_res_real) < opt.batchSize:
#             print("high_res_real data not enough")
#             print("data: ", len(high_res_real), "batch_size: ", opt.batchSize)
#             break;
            
#         # Downsample images to low resolution
#         for j in range(opt.batchSize):
#             low_res[j] = scale(high_res_real[j])
#             high_res_real[j] = normalize(high_res_real[j])

#         # Generate real and fake inputs
#         if opt.cuda:
#             high_res_real = Variable(high_res_real.cuda())
#             high_res_fake = generator(Variable(low_res).cuda())
#         else:
#             high_res_real = Variable(high_res_real)
#             high_res_fake = generator(Variable(low_res))

#         ######### Train generator #########
#         generator.zero_grad()

#         generator_content_loss = content_criterion(high_res_fake, high_res_real)
#         mean_generator_content_loss += generator_content_loss.data[0]

#         generator_content_loss.backward()
#         optim_generator.step()

#         ######### Status and display #########
#         sys.stdout.write('\r[%d/%d][%d/%d] Generator_MSE_Loss: %.4f' % (epoch, 2, i, len(dataloader), generator_content_loss.data[0]))
#         if i % 100 == 0:

#             vutils.save_image(high_res_fake.cpu().data,
#                     '%s/fake_samples_epoch_%03d.jpg' % (opt.out, epoch))
#             vutils.save_image(low_res.cpu().data,
#                     '%s/low_res_samples_epoch_%03d.jpg' % (opt.out, epoch))

        
        

# # Do checkpointing
# torch.save(generator.state_dict(), '%s/generator_pretrain.pth' % opt.out)


generator.load_state_dict(torch.load('checkpoints/generator_pretrain.pth'))
# SRGAN training
optim_generator = optim.Adam(generator.parameters(), lr=opt.generatorLR*0.1)
optim_discriminator = optim.Adam(discriminator.parameters(), lr=opt.discriminatorLR*0.1)

print('SRGAN training')
for epoch in range(opt.nEpochs):
    mean_generator_content_loss = 0.0
    mean_generator_adversarial_loss = 0.0
    mean_generator_total_loss = 0.0
    mean_discriminator_loss = 0.0

    for i, data in enumerate(dataloader):
        # Generate data
        high_res_real, _ = data
        
        if len(high_res_real) < opt.batchSize:
#             print("high_res_real data not enough")
#             print("data: ", len(high_res_real), "batch_size: ", opt.batchSize)
            break;

        # Downsample images to low resolution
        for j in range(opt.batchSize):
            low_res[j] = scale(high_res_real[j])
            high_res_real[j] = normalize(high_res_real[j])

        # Generate real and fake inputs
        if opt.cuda:
            high_res_real = Variable(high_res_real.cuda())
            high_res_fake = generator(Variable(low_res).cuda())
            target_real = Variable(torch.rand(opt.batchSize,1)*0.5 + 0.7).cuda()
            target_fake = Variable(torch.rand(opt.batchSize,1)*0.3).cuda()
        else:
            high_res_real = Variable(high_res_real)
            high_res_fake = generator(Variable(low_res))
            target_real = Variable(torch.rand(opt.batchSize,1)*0.5 + 0.7)
            target_fake = Variable(torch.rand(opt.batchSize,1)*0.3)
        
        ######### Train discriminator #########
        discriminator.zero_grad()

        discriminator_loss = adversarial_criterion(discriminator(high_res_real), target_real) + \
                             adversarial_criterion(discriminator(Variable(high_res_fake.data)), target_fake)
        mean_discriminator_loss += discriminator_loss.data[0]
        
        discriminator_loss.backward()
        optim_discriminator.step()

        ######### Train generator #########
        generator.zero_grad()

        real_features = Variable(feature_extractor(high_res_real).data)
        fake_features = feature_extractor(high_res_fake)

        generator_content_loss = content_criterion(high_res_fake, high_res_real) + 0.006*content_criterion(fake_features, real_features)
        mean_generator_content_loss += generator_content_loss.data[0]
        generator_adversarial_loss = adversarial_criterion(discriminator(high_res_fake), ones_const)
        mean_generator_adversarial_loss += generator_adversarial_loss.data[0]

        generator_total_loss = generator_content_loss + 1e-3*generator_adversarial_loss
        mean_generator_total_loss += generator_total_loss.data[0]
        
        generator_total_loss.backward()
        optim_generator.step()   
        
        ######### Status and display #########
        sys.stdout.write('\r[%d/%d][%d/%d] Discriminator_Loss: %.4f Generator_Loss (Content/Advers/Total): %.4f/%.4f/%.4f\n' % (epoch, opt.nEpochs, i, len(dataloader),
        discriminator_loss.data[0], generator_content_loss.data[0], generator_adversarial_loss.data[0], generator_total_loss.data[0]))

        ######### update images saved #########
        if i % 100 == 0:
            # the first 64 samples from the mini-batch are saved.
            vutils.save_image(high_res_fake.cpu().data,
                    '%s/fake_samples_epoch_%03d.jpg' % (opt.out, epoch))
            vutils.save_image(low_res.cpu().data,
                    '%s/low_res_samples_epoch_%03d.jpg' % (opt.out, epoch))

    # Do checkpointing
    torch.save(generator.state_dict(), '%s/generator_final.pth' % opt.out)
    torch.save(discriminator.state_dict(), '%s/discriminator_final.pth' % opt.out)

# Avoid closing
# while True:
#     pass
