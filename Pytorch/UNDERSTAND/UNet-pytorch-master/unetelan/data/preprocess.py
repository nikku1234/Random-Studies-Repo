import glob
import os
import re

import numpy.random as npr
import torch as th
from torch.utils import data
from torch.utils.data.dataloader import default_collate
from torchvision.transforms import *
from tqdm import tqdm

np.random.seed(0)


class HRFLoader:
    test = ['07_dr', '05_g', '08_h', '14_h', '01_g', '15_dr']

    def __init__(self, path):
        self.path = path

        candidates = os.listdir(path)
        candidates = [name for name in candidates if name.endswith('.jpg')]
        candidates = {name[:-4] for name in candidates}
        candidates = {name for name in candidates if not name.endswith("_mask")}
        candidates = {name for name in candidates if name not in self.test}

        self.train = list(sorted(list(candidates)))

    def _load(self, lst):
        images = [Image.open(os.path.join(self.path, f'{name}.jpg')) for name in lst]
        masks  = [Image.open(os.path.join(self.path, f'{name}.tif')) for name in lst]

        return images, masks

    def get_train(self):
        return self._load(self.train)

    def get_test(self):
        return self._load(self.test)


class DRVLoader:
    def __init__(self, path):
        self.path = path
        self.train_path = os.path.join(path, "training")
        self.test_path  = os.path.join(path, "test")

    def _load(self, path):
        images_path = os.path.join(path, 'images')
        masks_path  = os.path.join(path, '1st_manual')

        images = [Image.open(os.path.join(images_path, f'{name}')) for name in sorted(os.listdir(images_path))]
        masks  = [Image.open(os.path.join(masks_path,  f'{name}')) for name in sorted(os.listdir(masks_path))]

        return images, masks

    def get_train(self):
        return self._load(self.train_path)

    def get_test(self):
        return self._load(self.test_path)


class Rotate:
    def __init__(self, n, max_angle):
        self.n = n
        self.max_angle = max_angle

    def __call__(self, images, masks):
        ret_images = []
        ret_masks = []

        for img, msk in tqdm(zip(images, masks), desc="{:<10}".format("Rotate"), total=len(images)):
            ret_images += [img]
            ret_masks += [msk]

            for _ in range(self.n):
                alpha = npr.rand() * self.max_angle

                ret_images += [img.rotate(alpha, Image.BICUBIC)]
                ret_masks += [msk.rotate(alpha, Image.BICUBIC)]

        return ret_images, ret_masks


class RandomCrop:
    def __init__(self, n, size, th_img=50, th_msk=50):
        self.size = size
        self.n = n
        self.th_img = th_img * 255
        self.th_msk = th_msk * 255

    def crop(self, img, msk):
        w, h = img.size
        nnz_msk = 0
        nnz_img = 0

        max_left = w - self.size
        max_upper = h - self.size

        img_crop, msk_crop = None, None
        while nnz_img < self.th_img or nnz_msk < self.th_msk:
            left = npr.randint(low=0, high=max_left, size=1)
            upper = npr.randint(low=0, high=max_upper, size=1)

            box = (left, upper, left + self.size, upper + self.size)  # left, upper, right, and lower
            box = tuple(map(int, box))

            img_crop, msk_crop = img.crop(box).copy(), msk.crop(box).copy()

            nnz_img = np.sum(img_crop)
            nnz_msk = np.sum(msk_crop)

        return img_crop, msk_crop

    def __call__(self, images, masks):
        ret_images = []
        ret_masks = []

        for img, msk in tqdm(zip(images, masks), desc="{:<10}".format("Crop"), total=len(images)):
            for _ in range(self.n):
                img_crop, msk_crop = self.crop(img, msk)

                ret_images.append(img_crop)
                ret_masks.append(msk_crop)

        return ret_images, ret_masks


class RegularCrop:
    def __init__(self, size, step):
        self.size = size
        self.step = step

    def _pad(self, img, tup):
        img = np.asarray(img)
        img = np.pad(img, tup, 'constant', constant_values=0)
        img = Image.fromarray(img)

        return img

    def pad_Image(self, img):
        n_dim = len(np.asarray(img).shape)

        tup = [(self.step, self.step), (self.step, self.step)]
        if n_dim > 2:
            tup += [(0, 0)]

        return self._pad(img, tup)

    def pad_Crop(self, img, pad_right, pad_down):
        n_dim = len(np.asarray(img).shape)

        tup = [(0, pad_down), (0, pad_right)]
        if n_dim > 2:
            tup += [(0, 0)]

        return self._pad(img, tup)

    def __call__(self, images, masks):
        ret_images = []
        ret_masks = []

        for img, msk in tqdm(zip(images, masks), desc="{:<10}".format("RegCrop"), total=len(images)):
            img  = self.pad_Image(img)
            msk  = self.pad_Image(msk)
            w, h = img.size

            upper = 0
            while upper < h:

                left = 0
                while left < w:
                    right = left  + self.size
                    down  = upper + self.size

                    pad_right = max(0, right - w)
                    pad_down  = max(0, down - h)

                    right = min(right, w)
                    down  = min(down, h)

                    box = (left, upper, right, down)
                    box = tuple(map(int, box))

                    img_crop = self.pad_Crop(img=img.crop(box).copy(), pad_right=pad_right, pad_down=pad_down)
                    msk_crop = self.pad_Crop(img=msk.crop(box).copy(), pad_right=pad_right, pad_down=pad_down)

                    ret_images.append(img_crop)
                    ret_masks.append(msk_crop)

                    left += self.step
                upper += self.step

        return ret_images, ret_masks


def DriveCrop():
    def f(images, masks):
        for i, (img, msk) in tqdm(enumerate(zip(images, masks)), desc="DriveCrop", total=len(images)):
            img = img.crop((0, 10, 565, 584-9)).copy()
            msk = msk.crop((0, 10, 565, 584-9)).copy()

            images[i] = img
            masks[i]  = msk

        return images, masks
    return f


def RGBConverter():
    def f(images, masks):
        for i, img in enumerate(images):
            images[i] = img.convert(mode='L')

        return images, masks

    return f


def ToNumpy():
    def f(images, masks):
        for i, (img, msk) in tqdm(enumerate(zip(images, masks)), desc="{:<10}".format("Numpy"), total=len(images)):
            img = np.array(img, dtype=np.float32)
            img /= 255.

            msk = np.array(msk, dtype=np.float32)
            msk /= 255.

            images[i] = img
            masks[i] = msk

        return images, masks

    return f


def FeatureWiseStd(with_mean=True, with_std=True):
    def f(images, masks):
        for i, img in enumerate(tqdm(images, desc="{:<10}".format("Std"))):
            mu    = np.mean(img, axis=2, keepdims=True)
            sigma = np.std(img, axis=2, keepdims=True) + 1e-6

            if with_mean:
                img = img - mu
            if with_std:
                img = img / (sigma+1e-6)

            images[i] = img

        return images, masks
    return f


def ChannelWiseStd(with_mean=True, with_std=True):
    def f(images, masks):
        for i, img in enumerate(tqdm(images, desc="{:<10}".format("Std"))):
            mu    = np.mean(img, axis=(0, 1), keepdims=True)
            sigma = np.std(img, axis=(0, 1), keepdims=True) + 1e-6

            if with_mean:
                img = img - mu
            if with_std:
                img = img / (sigma+1e-6)

            images[i] = img

        return images, masks
    return f


def Standardizer(with_mean=True, with_std=True):
    def f(images, masks):
        mean = np.zeros(images[0].shape, dtype=np.float64)
        std = np.zeros(images[0].shape, dtype=np.float64)

        for img in images:
            mean += img
        mean /= len(images)

        for img in images:
            std += (img - mean) ** 2
        std /= len(images)
        std = np.sqrt(std)

        if with_mean:
            for i, img in enumerate(images):
                img = (img)
                img -= mean
                images[i] = (img)

        if with_std:
            for i, img in enumerate(images):
                img = (img)
                img /= std + 1e-6
                images[i] = (img)

        return images, masks

    return f


class ListDataset(th.utils.data.Dataset):
    def __init__(self, images, masks):
        assert len(images) == len(masks)

        self.images = images
        self.masks = masks

    def __getitem__(self, index):
        return self.images[index], self.masks[index]

    def __len__(self):
        return len(self.images)


def make_collate(divide_masks=False):
    def collate_fn(batch):
        batch = [(img, msk.type(th.FloatTensor)) for img, msk in batch]
        images, masks = default_collate(batch)

        if divide_masks:
            masks /= 255.

        return images, masks
    return collate_fn
