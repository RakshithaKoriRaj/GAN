#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 23:53:14 2020

@author: rakshithakoriraj
"""

#libraries
import time
#import random
import numpy as np
#from numpy import random
from matplotlib import pyplot as plt
#import torch.nn.functional as F
from tqdm import tqdm
import os
import cv2
import Generator
import Discriminator
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils


EPOCHS = 1
batch_size = 100
image_size = 64
workers = 2


# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# Decide on which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

def make_folder(path):
	try:
		os.mkdir(path)
	except FileNotFoundError:
		print("{} path can't be created".format(path))
	except FileExistsError:
		print("{} folder already exists".format(path))

realPath = os.path.join(os.path.abspath(os.getcwd()),'folderimages')

# Create the dataset
dataset = dset.ImageFolder(root=realPath,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)



genImages = os.path.abspath(os.path.join(os.getcwd(), "genImages"))
make_folder(genImages)
    
# Plot some training images
real_batch = next(iter(dataloader))
fig = plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0][0:64], padding=2, normalize=True),(1,2,0)))
plt.show()
fig.savefig('genImages/trainImages-ts{}.png'.format(int(time.time())))

        
#Discriminator
discriminator = Discriminator.DNet(ngpu).to(device)
# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    discriminator = nn.DataParallel(discriminator, list(range(ngpu)))
    
#-------------------------------------#
#Generator
#Generating images from noise
generator = Generator.GNet(ngpu).to(device)
# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    generator = nn.DataParallel(generator, list(range(ngpu)))
    
gimageList = []

fixedNoise = torch.randn(8, 100, 1, 1, device=device)

loss_fuction = nn.BCELoss()
gOptimizer = optim.Adam(generator.parameters(),lr=0.0002, betas=(0.5, 0.999))
dOptimizer = optim.Adam(discriminator.parameters(),lr=0.0002, betas=(0.5, 0.999))

errG = []
errD = []
#-------------------------------------#    
            
for epoch in range(EPOCHS):
    #for batch in range(0, len(dimageList), BATCH):
    for i, data in enumerate(dataloader, 0):
        dX = data[0].to(device)
        real_labels = torch.ones(len(dX), device=device)
        fake_labels = torch.zeros(len(dX), device=device)
        
        #Discriminator training
        discriminator.zero_grad()
        outputd = discriminator(dX).view(-1)
        realloss = loss_fuction(outputd, real_labels)
        realloss.backward() #retain_graph=True
        D_x = outputd.mean().item()
        
        
        #Generator training   
        noise = torch.randn(len(dX), 100, 1, 1)
        gImages = generator(noise)   
        #gX = torch.Tensor(gImages).view(-1,3,64,64)
        
        outputg = discriminator(gImages.detach()).view(-1)
        fakeloss = loss_fuction(outputg, fake_labels)
        fakeloss.backward()
        G_x = outputg.mean().item()
        
        
        totalLoss = (realloss + fakeloss)
        dOptimizer.step()
        
        
        generator.zero_grad()        
        outputf = discriminator(gImages).view(-1)
        los = loss_fuction(outputf, real_labels)
        los.backward()
        gOptimizer.step()
        G_x2 = outputf.mean().item()
        
        
        #saving losses to plot grahs
        errD.append(totalLoss)
        errG.append(los)
        if i%100 == 0:
            print("batch:{},epoch:{},totalLoss{},los{}".format(i,epoch,totalLoss,los))
            
        if i%1000 == 0:
            #print("totalLoss{},los{}".format(totalLoss,los))
            with torch.no_grad():
                genImages = generator(fixedNoise)
                genImages = genImages 
                #_ = viewSamples(genImages, 1, 5) #, batch, epoch
                fig = plt.figure(figsize=(10,10))
                plt.axis("off")
                plt.title("Generated Images")
                plt.imshow(np.transpose(vutils.make_grid(genImages, nrow= 2, padding=2, normalize=True).cpu()))
                plt.show()
                fig.savefig('genImages/genImage-ts{}batch{}epoch{}.png'.format(int(time.time()),i,epoch))
       


with torch.no_grad():
    genImages = generator(fixedNoise)
    genImages = genImages
    fig = plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Final Images")
    plt.imshow(np.transpose(vutils.make_grid(genImages,  nrow= 2, padding=2, normalize=True).cpu()))
    fig.savefig('genImages/finalImage-ts{}.png'.format(int(time.time())))
    
    
    
    
fig = plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(errG,label="G")
plt.plot(errD,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()
fig.savefig('genImages/graph-ts{}.png'.format(int(time.time())))