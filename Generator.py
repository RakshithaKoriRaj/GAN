#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 23:51:14 2020

@author: rakshithakoriraj
"""

import torch.nn as nn


class GNet(nn.Module):
    def __init__(self, ngpu):
        super().__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channels=100, out_channels=64, kernel_size=4, 
                               stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, 
                               stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, 
                               stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=4, 
                               stride=2, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(in_channels=8, out_channels=3, kernel_size=4, 
                               stride=2, padding=1, bias=False), #number 3 represents channels = 3
            nn.Tanh(),
            
        )      
        
    def forward(self, input):
        return self.main(input)
