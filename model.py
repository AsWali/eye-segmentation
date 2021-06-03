# -*- coding: utf-8 -*-
"""
@author: AsWali
Implementation of the Eye Semantic Segmentation with A Lightweight Model architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

class BottleNeck(nn.Module):
    # t = expansion factor
    def __init__(self, in_filters, out_filters, t, s):
        super(BottleNeck, self).__init__()
        self.s = s
        self.in_filters = in_filters
        self.out_filters = out_filters
        t_fitlers = in_filters * t
        if (s == 1):
            self.conv1=nn.Conv2d(in_filters, t_fitlers, kernel_size=1)
            self.bn1 = nn.BatchNorm2d(t_fitlers)
            self.relu1 = nn.ReLU6()

            # depthwise
            self.spatial2=nn.Conv2d(t_fitlers, t_fitlers, kernel_size=3, padding=1, groups=t_fitlers)
            self.depth2=nn.Conv2d(t_fitlers, t_fitlers, kernel_size=1)
            self.bn2 = nn.BatchNorm2d(t_fitlers)
            self.relu2 = nn.ReLU6()
            self.conv2=lambda x: self.depth2(self.spatial2(x))

            # normal
            self.conv3=nn.Conv2d(t_fitlers, out_filters, kernel_size=1)
            self.bn3 = nn.BatchNorm2d(out_filters)
            
        else:
            self.conv1=nn.Conv2d(in_filters, t_fitlers, kernel_size=1)
            self.bn1 = nn.BatchNorm2d(t_fitlers)
            self.relu1 = nn.ReLU6()

            # depthwise
            self.spatial2=nn.Conv2d(t_fitlers, t_fitlers, kernel_size=3, padding=1, groups=t_fitlers, stride=2)
            self.depth2=nn.Conv2d(t_fitlers, t_fitlers, kernel_size=1)
            
            self.conv2=lambda x: self.depth2(self.spatial2(x))
            self.relu2 = nn.ReLU6()
            self.bn2 = nn.BatchNorm2d(t_fitlers)

            # normal
            self.conv3=nn.Conv2d(t_fitlers, out_filters, kernel_size=1)
            self.bn3 = nn.BatchNorm2d(out_filters)


    def forward(self, x):
        if(self.s == 1):
            orig_x = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        if(self.s == 1 and self.in_filters == self.out_filters):
            x += orig_x
        
        return x


class Enc(nn.Module):
    def __init__(self):
        super(Enc, self).__init__()
        
        #using grayscale showing shape
        #torch.Size([1, 1, 320, 200])
        self.enc1=nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        #torch.Size([1, 32, 320, 200])
        self.enc2=BottleNeck(32, 16, 1, 2)
        self.enc3=BottleNeck(16, 16, 1, 1)
        self.enc4=BottleNeck(16, 24, 6, 2)
        self.enc5=BottleNeck(24, 24, 6, 1)
        self.enc6=BottleNeck(24, 24, 6, 1)
        self.enc7=BottleNeck(24, 32, 6, 2)
        self.enc8=BottleNeck(32, 32, 6, 1)
        self.enc9=BottleNeck(32, 32, 6, 1)
        self.enc10=BottleNeck(32, 32, 6, 1)
        self.enc11=nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
    def forward(self, x):
        x=self.enc1(x)
        x=self.enc2(x)
        x=self.enc3(x)
        x=self.enc4(x)
        x=self.enc5(x)
        x=self.enc6(x)
        x=self.enc7(x)
        x=self.enc8(x)
        x=self.enc9(x)
        x=self.enc10(x)
        x=self.enc11(x)
            
        return x

# https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py
# use reduction=4
class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class Dec(nn.Module):
    def __init__(self):
        super(Dec, self).__init__()
        self.conv1=nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()

        self.bu1 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners = False)

        ## route main
        self.conv2=nn.Conv2d(64, 4, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(4)
        self.relu2 = nn.ReLU()

        self.conv3=nn.Conv2d(4, 4, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(4)
        self.relu3 = nn.ReLU()
        
        self.conv4=nn.Conv2d(4, 4, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(4)
        self.relu4 = nn.ReLU()
        ## mergepoint 1
        self.bu2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners = False)

        ## route 2
        self.conv5=nn.Conv2d(64, 4, kernel_size=1)
        self.bn5 = nn.BatchNorm2d(4)
        self.relu5 = nn.ReLU()
        self.bu3 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners = False)
        
        ## mergepoint 2
        self.se1 = SELayer(4)
        
        ## merge route
        self.sm1 = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.bu1(x)

        route_1_x = x
        route_2_x = x
        ## route 1
        route_1_x = self.conv2(route_1_x)
        route_1_x = self.bn2(route_1_x)
        route_1_x = self.relu2(route_1_x)

        route_1_x = self.conv3(route_1_x)
        route_1_x = self.bn3(route_1_x)
        route_1_x = self.relu3(route_1_x)

        route_1_x = self.conv4(route_1_x)
        route_1_x = self.bn4(route_1_x)
        route_1_x = self.relu4(route_1_x)

        route_1_x = self.bu2(route_1_x)

        ## route 2
        route_2_x = self.conv5(route_2_x)
        route_2_x = self.bn5(route_2_x)
        route_2_x = self.relu5(route_2_x)

        route_2_x = self.bu3(route_2_x)
        route_2_x = self.se1(route_2_x)

        ## merge
        x = self.sm1(route_1_x + route_2_x)
        return x



transform = transforms.Compose([transforms.Resize((200, 320)),
                            transforms.ToTensor(), transforms.Grayscale(num_output_channels=1)])
dataset = datasets.ImageFolder("data", transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)
images, labels = next(iter(dataloader))
d = images
#d = torch.rand(1, 1, 320, 200)
denc=Enc()
ddec=Dec()
e = denc(d)
f = ddec(e)
imgplot = plt.imshow(d[0,0,:,:])
plt.show()
imgplot = plt.imshow(torch.argmax(f, dim = 1).detach()[0].numpy())
plt.show()
print(f.shape)