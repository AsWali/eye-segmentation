
import torch
import torch.nn as nn
import torchcontrib
import torch.nn.functional as F

from torchvision import datasets, transforms
import VRNet
import matplotlib.pyplot as plt

def generalised_dice_loss_ce(output, target, device, n_classes=4, type_weight='simple', add_crossentropy=False):
    n_pixel = target.numel()
    _, counts = torch.unique(target, return_counts=True)
    cls_weight = torch.div(n_pixel, n_classes * counts.type(torch.FloatTensor)).to(device)

    if type_weight == 'square':
        cls_weight = torch.pow(cls_weight, 2.0)

    if add_crossentropy:
        loss_entropy = F.nll_loss(torch.log(output), target, weight=cls_weight)

    encoded_target = make_one_hot(target.to(torch.int64),4)
    #encoded_target = target.clone().to(torch.float)
    # print(output.size(), encoded_target.size(), target.size(), len)
    assert output.size() == encoded_target.size()

    intersect = torch.sum(torch.mul(encoded_target, output), dim=(2, 3))
    union = torch.sum(output, dim=(2, 3)) + torch.sum(encoded_target, dim=(2, 3))
    union[union < 1] = 1
    print(cls_weight.shape)
    print(union.shape)

    gdl_numerator = torch.sum(torch.mul(cls_weight, intersect), dim=1)
    gdl_denominator = torch.sum(torch.mul(cls_weight, union), dim=1)
    generalised_dice_score = torch.sub(1.0, 2 * gdl_numerator / gdl_denominator)

    if add_crossentropy:
        loss = 0.5 * torch.mean(generalised_dice_score) + 0.5 * loss_entropy
    else:
        loss = torch.mean(generalised_dice_score)

    return loss


def make_one_hot(labels, num_classes):
    '''
    Converts an integer label torch.autograd.Variable to a one-hot Variable.

    Parameters
    ----------
    labels : torch.autograd.Variable of torch.cuda.LongTensor
        N x 1 x H x W, where N is batch size.
        Each value is an integer representing correct classification.
    Returns
    -------
    target : torch.autograd.Variable of torch.cuda.FloatTensor
        N x C x H x W, where C is class number. One-hot encoded.
    '''
    one_hot = torch.cuda.FloatTensor(labels.size(0), num_classes, labels.size(2), labels.size(3)).zero_()
    target = one_hot.scatter_(1, labels.data, 1) 
    return target

def train_op(model, optimizer, input, target):
    dec = model(input, returns='dec')
    #l = torch.argmax(dec, dim = 1, keepdim=True)
    loss=generalised_dice_loss_ce(dec,target, 'cuda')
    loss.requires_grad=True
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
    optimizer_swa = torchcontrib.optim.SWA(optimizer, swa_start=51, swa_freq=5, swa_lr=learning_rate)
    transform = transforms.Compose([transforms.Resize(img_size),
                                transforms.ToTensor(),  transforms.Grayscale(num_output_channels=1)])

    dataset = datasets.ImageFolder("data/train", transform=transform)

    # Train 1 image set batch size=1 and set shuffle to False
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    
    dataset_gt = datasets.ImageFolder("data/gt", transform=transforms.Compose([transforms.Resize((640, 400)),transforms.ToTensor()]))
    dataloader_gt = torch.utils.data.DataLoader(dataset_gt, batch_size=1, shuffle=False)

    # Run for every epoch
    for epoch in range(1):


        # if (epoch > 27 and epoch <= 46):
        #     optimizer.param_groups[0]['lr'] -=  0.0001
        #     optimizer_swa.param_groups[0]['lr'] -=  0.0001
        # if(epoch == 47):
        #     optimizer.param_groups[0]['lr'] -=  0.0005
        #     optimizer_swa.param_groups[0]['lr'] -=  0.0005


        # Print out every epoch:
        print("Epoch = " + str(epoch))

        for (idx, batch) in enumerate(dataloader):
            # Train 1 image idx > 1
            # if(idx > 1): break

            # Train Wnet with CUDA if available
            if CUDA:
                batch[0] = batch[0].cuda()
            
            images, labels = next(iter(dataloader_gt))
            vrnet, loss = train_op(vrnet, optimizer_swa, batch[0], images.cuda())

    images, labels = next(iter(dataloader))

    # Run wnet with cuda if enabled
    if CUDA:
        images = images.cuda()

    dec = vrnet(images, returns='dec')
    imgplot = plt.imshow(images[0,0,:,:].cpu())
    plt.show()
    imgplot = plt.imshow(torch.argmax(dec, dim = 1).cpu().detach()[0].numpy())
    plt.show()
    torch.save(vrnet.state_dict(), "model")
    print("Done")

if __name__ == '__main__':
    main()
