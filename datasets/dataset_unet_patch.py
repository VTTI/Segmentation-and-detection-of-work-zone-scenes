import glob
import os
import random
import re

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torchvision import transforms

from utils.utils import parse_xml, unique_name, merge_and_split


def get_wzdf(path, zone=1):  # workzone dataframe
    path = os.path.join(path, '**', '*.xml')
    data = list()
    for xmlfile in glob.iglob(path, recursive=True):
        print(f"Processing: {xmlfile}...")
        image_root = os.path.join(re.findall("(.+)/", xmlfile)[0], "Images")
        xml_data = parse_xml(xmlfile)
        if xml_data is not None:
            for item in xml_data.items():
                temp = list()
                image_path = os.path.join(image_root, item[0])
                temp.append(image_path)  # image path
                name = unique_name(image_path)  # unique name
                temp.append(name)
                temp.append(zone)  # work-zone or non-work zone
                temp.append(item[1])  # list of points containing work zone polygon annotations

                if re.search("Day", xmlfile):  # time of day i.e. Day/Night
                    temp.append(1)
                elif re.search("Night", xmlfile):
                    temp.append(0)
                else:
                    continue
                data.append(temp)
    df = pd.DataFrame(data, columns=['path', 'name', 'workzone', 'points', 'tod'])
    return df


def get_mask_weight(mask):
    mask_w = cv2.erode(mask, kernel=np.ones((8, 8), np.uint8), iterations=1)
    mask_w = mask - mask_w
    return mask_w


def normalize(image):
    return TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def add_noise(image, std=0.2, mean=0.):
    return image + torch.randn(image.size()) * std + mean


def flip_and_jitter(image, mask, mask_weight, j=0):
    jitter = transforms.ColorJitter(brightness=j, contrast=j, saturation=j, hue=j)
    if random.random() > 0.5:
        image = TF.hflip(image)
        image = jitter(image)
        mask = TF.hflip(mask)
        mask_weight = TF.hflip(mask_weight)
    return image, mask, mask_weight


def get_mask(polygons, image_shape):
    width, height = image_shape
    mask = Image.new('L', (width, height))  # workzone mask
    if not polygons:
        return mask
    else:
        for polygon in polygons:
            polygon_points = [tuple(map(int, point)) for point in polygon]
            ImageDraw.Draw(mask).polygon(polygon_points, outline="white", fill="white")
        return mask


class DatasetUnetPatch(Dataset):

    def __init__(self, df, crop_shape, split):
        self.df = df
        self.crop_shape = crop_shape
        self.split = split

    def __len__(self):
        return len(self.df)

    def random_crop(self, image, mask):
        crop_hw = self.crop_shape
        h, w, c = image.shape
        assert w >= 244, f"crop size {crop_hw} should be < image size. Got image size {(h, w)}"
        x = random.randint(0, w - crop_hw[1] - 1)
        y = random.randint(0, h - crop_hw[0] - 1)
        image = image[y:y + crop_hw[0], x:x + crop_hw[1], :]
        mask = mask[y:y + crop_hw[0], x:x + crop_hw[1]]
        return image, mask

    def get_patches(self, image, ismask=False):

        kh, kw = self.crop_shape  # kernel size
        dh, dw = self.crop_shape  # stride

        image = F.pad(image, (
            image.shape[2] % kw // 2, image.shape[2] % kw // 2, image.shape[1] % kh // 2, image.shape[1] % kh // 2))

        patches = image.unfold(2, kh, dh).unfold(3, kw, dw)

        if ismask:
            patches = patches.permute(2, 3, 0, 1, 4, 5).contiguous().view(-1, 1, kh, kw)
        else:
            patches = patches.permute(2, 3, 0, 1, 4, 5).contiguous().view(-1, 3, kh, kw)

        return patches

    def preprocess(self, image, mask):
        # image to numpy
        image = np.array(image)
        mask = np.array(mask)

        if self.split == "test":
            # transform to tensor
            image, mask = TF.to_tensor(image).unsqueeze(0), TF.to_tensor(mask).unsqueeze(0)
            # normalize
            image = normalize(image)
            image = self.get_patches(image)
            mask = self.get_patches(mask, ismask=True)
            return image, mask, -1

        elif self.split == "train":  # add additional transformations on the image
            image, mask = self.random_crop(image, mask)
            mask_weight = get_mask_weight(mask)
            # transform to tensor
            image, mask, mask_weight = TF.to_tensor(image), TF.to_tensor(mask), TF.to_tensor(mask_weight)
            # flipping + color jitters
            image, mask, mask_weight = flip_and_jitter(image, mask, mask_weight)
            # normalize
            image = normalize(image)
            # add noise
            if random.random() > 0.5:
                image = add_noise(image)
            return image, mask, mask_weight

    def __getitem__(self, item):
        image = Image.open(self.df.iloc[item, 0])
        image_shape = image.size
        label = self.df.iloc[item, 2]
        tod = self.df.iloc[item, 4]
        points = list(filter(None, self.df.iloc[item, 3]))  # filtering out None values
        mask = get_mask(points, image_shape)

        # Image pre-processing
        image, mask, mask_weight = self.preprocess(image, mask)
        data = {"image": image, "label": label, "tod": tod, "mask": mask, "mask_weight": mask_weight}
        return data


def dataset_unet_patch(workzone, out_dir="output", crop_shape=None):
    """**************** Work Zone ****************"""
    work_zone_df = get_wzdf(path=workzone, zone=1)

    all_data_df, train_df, test_df, val_df = merge_and_split(work_zone_df, out_dir, save=True)

    # sanity check
    # train_test_df = test_df["Name"].isin(train_df["Name"])
    # train_test_df.to_csv("train_test_df.csv")
    # train_val_df = val_df["Name"].isin(train_df["Name"])
    # train_val_df.to_csv("train_val_df.csv")

    """Creating dataset"""
    train_set = DatasetUnetPatch(df=train_df, crop_shape=crop_shape, split="train")
    val_set = DatasetUnetPatch(df=val_df, crop_shape=crop_shape, split="test")
    test_set = DatasetUnetPatch(df=test_df, crop_shape=crop_shape, split="test")
    return train_set, test_set, val_set
