# -*- coding: utf-8 -*-
"""
@author: AsWali
Implementation of the Eye Semantic Segmentation with A Lightweight Model architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class BottleNeck(nn.Module):
    def __init__(self, in_filters, out_filters, t, c, s):
        super(BottleNeck, self).__init__()
        
        if s:
            self.conv1=nn.Conv2d(in_filters, out_filters, kernel_size=1)
            self.relu1 = nn.ReLU6()

            # depthwise
            self.spatial2=nn.Conv2d(in_filters, in_filters, kernel_size=3, groups=in_filters)
            self.depth2=nn.Conv2d(in_filters, out_filters, kernel_size=1)
            
            self.conv2=lambda x: self.depth2(self.spatial2(x))
            self.relu2 = nn.ReLU6()

            # normal
            self.conv3=nn.Conv2d(in_filters, out_filters, kernel_size=1)
            self.linear3 = nn.Linear()
            
        else:
            self.conv1=nn.Conv2d(in_filters, out_filters, kernel_size=1)
            self.relu1 = nn.ReLU6()

            # depthwise
            self.spatial2=nn.Conv2d(in_filters, in_filters, kernel_size=3, groups=in_filters, stride=2)
            self.depth2=nn.Conv2d(in_filters, out_filters, kernel_size=1)
            
            self.conv2=lambda x: self.depth2(self.spatial2(x))
            self.relu2 = nn.ReLU6()

            # normal
            self.conv3=nn.Conv2d(in_filters, out_filters, kernel_size=1)
            self.linear3 = nn.Linear()


    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.linear3(x)
        
        return x
