import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from dataset import MyCustomDataset
from torchvision import transforms
from post_process import filter

import VRNet
import matplotlib.pyplot as plt

# torch.manual_seed(79796125883900)
# torch.cuda.manual_seed(79796125883900)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def mean_iou(pred, label, num_classes=4):
    accum = False
    iou = None
    for idx in range(num_classes):
        out1 = (pred == idx)
        out2 = (label == idx)

        intersect = torch.sum(out1 & out2, dim=(1, 2)).type(torch.FloatTensor)
        union = torch.sum(out1 | out2, dim=(1, 2)).type(torch.FloatTensor)
        if accum:
            iou = iou + torch.div(intersect, union + 1e-16)
        else:
            iou = torch.div(intersect, union + 1e-16)
            accum = True
    m_iou = torch.mean(iou) / num_classes
    return m_iou

def main():
    model = VRNet.VRNet().cuda()
    model.load_state_dict(torch.load("f_model_3"))
    T = count_parameters(model)
    S = np.divide(np.multiply(T, 4), np.multiply(1024, 1024))

    transform = transforms.Compose([transforms.Resize((320,200)),
                                transforms.ToTensor()])
    dataset = MyCustomDataset("data/validation", transform)

    # Train 1 image set batch size=1 and set shuffle to False
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    
    iou_list = []
    for idx, images in enumerate(dataloader):
        print(idx)
        dec = model(images["image"].cuda(), returns='dec')
        filtered = filter(torch.argmax(dec, dim = 1))
        iou_list.append(mean_iou(torch.from_numpy(filtered), images["mask"]))

    print(np.mean(iou_list))

    dec = model(images["image"].cuda(), returns='dec')
    imgplot = plt.imshow(images["image"][0,0,:,:].cpu())
    plt.show()
    imgplot = plt.imshow(torch.argmax(dec, dim = 1).cpu().detach()[0].numpy())
    plt.show()
    fitlered = filter(torch.argmax(dec, dim = 1))
    print(mean_iou(torch.from_numpy(fitlered), images["mask"]))
    imgplot = plt.imshow(fitlered)
    plt.show()

if __name__ == '__main__':
    main()
