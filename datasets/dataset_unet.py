import glob
import os
import random
import re

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torchvision import transforms

from utils.utils import parse_xml, unique_name, merge_and_split


def get_wzdf(path, zone=1):  # workzone dataframe
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


class DatasetUnet(Dataset):

    def __init__(self, df, resize_shape, split):
        self.df = df
        self.resize_shape = resize_shape
        self.split = split

    def __len__(self):
        return len(self.df)

    @staticmethod
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

    def resize(self, image, mask, mask_weight=None):
        image = TF.resize(image, self.resize_shape)
        mask = TF.resize(mask, self.resize_shape, interpolation=Image.NEAREST)
        if mask_weight is not None:
            mask_weight = TF.resize(mask_weight, self.resize_shape, interpolation=Image.NEAREST)
            return image, mask, mask_weight
        else:
            return image, mask

    @staticmethod
    def get_mask_weight(mask):
        mask_w = cv2.erode(mask, kernel=np.ones((8, 8), np.uint8), iterations=1)
        mask_w = mask - mask_w
        return mask_w

    @staticmethod
    def flip_and_jitter(image, mask, mask_weight, j=0.05):
        jitter = transforms.ColorJitter(brightness=j, contrast=j, saturation=j, hue=j)
        if random.random() > 0.5:
            image = TF.hflip(image)
            image = jitter(image)

            mask = TF.hflip(mask)
            mask_weight = TF.hflip(mask_weight)
        return image, mask, mask_weight

    @staticmethod
    def normalize(image):
        return TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    @staticmethod
    def add_noise(image, std=0.2, mean=0.):
        return image + torch.randn(image.size()) * std + mean

    def preprocess(self, image, mask):

        # image to numpy
        image = np.array(image)
        mask = np.array(mask)

        if self.split == "test":
            # transform to tensor
            image, mask = TF.to_tensor(image), TF.to_tensor(mask)
            # resize
            image, mask = self.resize(image, mask)
            # normalize
            image = self.normalize(image)
            return image, mask, -1

        elif self.split == "train":  # add additional transformations on the image
            # mask weight
            mask_weight = self.get_mask_weight(mask)
            # transform to tensor
            image, mask, mask_weight = TF.to_tensor(image), TF.to_tensor(mask), TF.to_tensor(mask_weight)
            # resize
            image, mask, mask_weight = self.resize(image, mask, mask_weight)
            # flipping + color jitters
            image, mask, mask_weight = self.flip_and_jitter(image, mask, mask_weight)
            # normalize
            image = self.normalize(image)
            # add noise
            if random.random() > 0.5:
                image = self.add_noise(image)
            return image, mask, mask_weight

    def __getitem__(self, item):
        image = Image.open(self.df.iloc[item, 0])
        image_shape = image.size
        label = self.df.iloc[item, 2]
        tod = self.df.iloc[item, 4]
        points = list(filter(None, self.df.iloc[item, 3]))  # filtering out None values
        mask = self.get_mask(points, image_shape)

        # Image pre-processing
        image, mask, mask_weight = self.preprocess(image, mask)
        data = {"image": image, "label": label, "tod": tod, "mask": mask, "mask_weight": mask_weight}
        return data


def dataset_unet(workzone, out_dir="output", resize_shape=(224, 224)):
    """**************** Work Zone ****************"""
    path = os.path.join(workzone, '**', '*.xml')
    work_zone_df = get_wzdf(path, zone=1)

    all_data_df, train_df, test_df, val_df = merge_and_split(work_zone_df, out_dir, save=True)

    # sanity check
    # train_test_df = test_df["name"].isin(train_df["name"])
    # train_test_df.to_csv("train_test_df.csv")
    # train_val_df = val_df["name"].isin(train_df["name"])
    # train_val_df.to_csv("train_val_df.csv")

    """Creating dataset"""
    train_set = DatasetUnet(df=train_df, resize_shape=resize_shape, split="train")
    val_set = DatasetUnet(df=val_df, resize_shape=resize_shape, split="test")
    test_set = DatasetUnet(df=test_df, resize_shape=resize_shape, split="test")

    return train_set, test_set, val_set
