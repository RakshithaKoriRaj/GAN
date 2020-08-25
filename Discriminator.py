#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 23:51:14 2020

@author: rakshithakoriraj
"""

import torch.nn as nn

class DNet(nn.Module):
    def __init__(self, ngpu):
        super(DNet, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is 3 x 64 x 64
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, 
                      padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, 
                      stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, 
                      stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, 
                      stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, 
                      padding=0, bias=False),
            nn.Sigmoid()
        )


    def forward(self, input):
        x = self.main(input)
        return x