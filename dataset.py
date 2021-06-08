import glob
from PIL import Image
import numpy as np
from torch.utils.data.dataset import Dataset
from torchvision import transforms

class MyCustomDataset(Dataset):
    def __init__(self, root_path, transform=None):
        """ Initialization """
        self.list_png = sorted(glob.glob(root_path + '/images/*.png'))
        self.transform = transform

    def __getitem__(self, index):
        img_path = self.list_png[index]
        npy_path = img_path.replace('images', 'labels').replace('png', 'npy')

        img = Image.open(img_path)
        if img.mode == 'RGB':
            img = img.convert('L')

        if 'test' in img_path:
            npy = np.array([])
        else:
            npy = np.load(npy_path, allow_pickle=False)

        img_name = img_path.replace('\\', '/').split('/')[-1][:-4]
        sample = {'image': img, 'mask': npy, 'name': img_name}
        if self.transform:
            sample["image"] = self.transform(img)

        return sample

    def __len__(self):
        """ Denotes the toal number of samples """
        return len(self.list_png)
        