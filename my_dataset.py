import os

import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset


class DriveDataset(Dataset):
    def __init__(self, root: str, train: bool, transforms=None):
        super(DriveDataset, self).__init__()
        self.flag = "train" if train else "test"
        data_root = os.path.join(root, self.flag)
        assert os.path.exists(data_root), f"path '{data_root}' does not exists."
        self.transforms = transforms
        img_names = [i for i in os.listdir(os.path.join(data_root, "images"))]
        self.img_list = [os.path.join(data_root, "images", i) for i in img_names]
        self.masks = [os.path.join(data_root, "masks", i)
                      for i in img_names]
        # check files
        for i in self.masks:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"file {i} does not exists.")


    def __getitem__(self, item):
        image = self.img_list[item]
        mask = self.masks[item]
        '''load the data '''
        image = Image.open(image).convert('RGB')
        label = Image.open(mask).convert('L')
        label = np.array(label) // 255
        label = Image.fromarray(label)

        if self.transforms is not None:
            image, label = self.transforms(image, label)

        return image, label

    def __len__(self):
        return len(self.img_list)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs