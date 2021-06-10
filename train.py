
import torch
import torch.nn as nn
import torchcontrib
import torch.nn.functional as F
import numpy as np
from dataset import MyCustomDataset
from post_process import filter
from torchvision import datasets, transforms
import VRNet
import matplotlib.pyplot as plt

torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)


def generalised_dice_loss_ce(output, target, device, n_classes=4, type_weight='simple', add_crossentropy=False):
    n_pixel = target.numel()
    _, counts = torch.unique(target, return_counts=True)
    cls_weight = torch.div(n_pixel, n_classes * counts.type(torch.FloatTensor)).to(device)
    if type_weight == 'square':
        cls_weight = torch.pow(cls_weight, 2.0)
    
    if add_crossentropy:
        loss_entropy = F.nll_loss(torch.log(output), target, weight=cls_weight)

    if len(target.size()) == 3:
        # Convert to one hot encoding
        encoded_target = F.one_hot(target.to(torch.int64), num_classes=n_classes)
        encoded_target = encoded_target.permute(0, 3, 1, 2).to(torch.float)
    else:
        encoded_target = target.clone().to(torch.float)
    # print(output.size(), encoded_target.size(), target.size(), len)
    assert output.size() == encoded_target.size()

    intersect = torch.sum(torch.mul(encoded_target, output), dim=(2, 3))
    union = torch.sum(output, dim=(2, 3)) + torch.sum(encoded_target, dim=(2, 3))
    union[union < 1] = 1

    gdl_numerator = torch.sum(torch.mul(cls_weight, intersect), dim=1)
    gdl_denominator = torch.sum(torch.mul(cls_weight, union), dim=1)
    generalised_dice_score = torch.sub(1.0, 2 * gdl_numerator / gdl_denominator)

    if add_crossentropy:
        loss = 0.5 * torch.mean(generalised_dice_score) + 0.5 * loss_entropy
    else:
        loss = torch.mean(generalised_dice_score)

    return loss

def train_op(model, optimizer, input, target):
    dec = model(input, returns='dec')
    loss=generalised_dice_loss_ce(dec,target, 'cuda')
    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

    return (model, loss)

def main():
    # Check if CUDA is available
    CUDA = torch.cuda.is_available()
    
    # Squeeze k
    img_size = (320, 200)
    vrnet = VRNet.VRNet()
    if(CUDA):
        vrnet = vrnet.cuda()
    learning_rate = 0.001
    optimizer = torch.optim.Adam(vrnet.parameters(), lr=learning_rate)
    # https://pytorch.org/blog/stochastic-weight-averaging-in-pytorch/
    # 51 epoch. need to caclulate how many steps that is
    optimizer = torchcontrib.optim.SWA(optimizer, swa_start=51, swa_freq=1, swa_lr=0.0005)
    transform = transforms.Compose([transforms.Resize(img_size),
                                transforms.ToTensor()])
    dataset = MyCustomDataset("data1", transform)

    # Train 1 image set batch size=1 and set shuffle to False
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    # Run for every epoch
    for epoch in range(1500):

        if (epoch > 27 and epoch <= 46):
            optimizer.param_groups[0]['lr'] -=  optimizer.param_groups[0]['lr'] * 0.0001
        if(epoch >= 47): 
            optimizer.param_groups[0]['lr'] =  0.0005

        # Print out every epoch:
        print("Epoch = " + str(epoch))
        print(optimizer.param_groups[0]['lr'])

        for (idx, batch) in enumerate(dataloader):
            # Train 1 image idx > 1
            if(idx > 1): break

            # Train vrnet with CUDA if available
           
            if CUDA:
                vrnet, loss = train_op(vrnet, optimizer, batch["image"].cuda(), batch["mask"].cuda())
            else:
                vrnet, loss = train_op(vrnet, optimizer, batch["image"], batch["mask"])
            print(loss.detach())

    images = next(iter(dataloader))
    
    optimizer.swap_swa_sgd()
    # Run vrnet with cuda if enabled
    if CUDA:
        images["image"] = images["image"].cuda()

    dec = vrnet(images["image"], returns='dec')
    imgplot = plt.imshow(images["image"][0,0,:,:].cpu())
    plt.show()
    imgplot = plt.imshow(torch.argmax(dec, dim = 1).cpu().detach()[0].numpy())
    plt.show()
    torch.save(vrnet.state_dict(), "model.pth")
    print("Done")

if __name__ == '__main__':
    main()
