import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from dataset import MyCustomDataset
from torchvision import datasets, transforms
import torchcontrib

import VRNet
import matplotlib.pyplot as plt

torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)

def main():
    model = VRNet.VRNet().cuda()
    model.load_state_dict(torch.load("f_model"))
    print(model)

    transform = transforms.Compose([transforms.Resize((320,200)),
                                transforms.ToTensor()])
    dataset = MyCustomDataset("data/validation", transform)

    # Train 1 image set batch size=1 and set shuffle to False
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    
    images = next(iter(dataloader))

    dec = model(images["image"].cuda(), returns='dec')
    imgplot = plt.imshow(images["image"][0,0,:,:].cpu())
    plt.show()
    imgplot = plt.imshow(torch.argmax(dec, dim = 1).cpu().detach()[0].numpy())
    plt.show()

if __name__ == '__main__':
    main()
