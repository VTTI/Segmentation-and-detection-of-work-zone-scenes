import collections
import os
import random
import re
import sys
import xml.etree.ElementTree as ET
from ast import literal_eval

import pandas as pd
import torch
import yaml


def get_class_imbalance_weights(train_df):
    """Class imbalance mitigation"""
    count_work_zone = train_df.groupby('workzone').size().to_list()  # managing class imbalance
    count_tod = train_df.groupby('tod').size().to_list()  # managing class imbalance
    weights_work_zone = torch.FloatTensor(
        [max(count_work_zone) / count_work_zone[0], max(count_work_zone) / count_work_zone[1]])
    weighs_tod = torch.FloatTensor([max(count_tod) / count_tod[0], max(count_tod) / count_tod[1]])
    return weights_work_zone, weighs_tod


def get_points(polygon):
    points = polygon.attrib["points"].split(';')
    points_list = list()
    for point in points:
        points_list.append(literal_eval(point))
    return points_list, len(points_list)


def parse_xml(xmlfile):
    data = collections.defaultdict(list)
    # create element tree object
    try:
        tree = ET.parse(xmlfile)
        # get root element
        root = tree.getroot()
        for image in root.findall("image"):
            temp = list()
            image_name = image.attrib["name"]
            for polygon in image.findall("polygon"):
                label = polygon.attrib["label"]
                points, _ = get_points(polygon)
                # search for workzone related objects
                if (label == "workzone") and (polygon.find("attribute").text == "cone"):
                    temp.append(points)
                elif (label == "workzone") and (polygon.find("attribute").text == "drum"):
                    temp.append(points)
                elif (label == "workzone") and (polygon.find("attribute").text == "barricade"):
                    temp.append(points)
                elif (label == "workzone") and (polygon.find("attribute").text == "board"):
                    temp.append(points)
                elif (label == "workzone") and (polygon.find("attribute").text == "trailer"):
                    temp.append(points)
                elif (label == "workzone") and (polygon.find("attribute").text == "other"):
                    temp.append(points)
                elif (label == "workzone") and (polygon.find("attribute").text == "unknown"):
                    temp.append(points)
                elif label == "machine":
                    temp.append(points)
                elif label == "construction_sign":
                    temp.append(points)
                elif (label == "barrier") and (polygon.find("attribute").text == "workzone"):
                    temp.append(points)
                elif (label == "person") and (polygon.find("attribute").text == "construction_worker"):
                    temp.append(points)
            data[image_name] = temp
        print(len(data))
        return data
    except Exception as e:
        print(e, xmlfile)
        return None


def unique_name(pth):
    name = re.findall("_([0-9a-z]+)_", pth.split('/')[-1])[0]  # vehicle name
    if name[:5] in ("WZ_N_", "BB_D_", "BB_N_", "OO_D_", "OO_N_"):
        return name[5:]
    elif name[:4] in ("I_D_", "I_N_"):
        return name[4:]
    else:
        return name


def split(len_dataset, p_train=0.80, p_test=0.10, p_val=0.10):
    len_train = int(len_dataset * p_train)
    len_test = int(len_dataset * p_test)
    len_val = int(len_dataset * p_val)

    if len_dataset == len_train + len_test + len_val:
        return len_train, len_test, len_val
    else:
        difference = len_dataset - (len_train + len_test + len_val)
        return len_train, len_test + difference, len_val


def merge_and_split(work_zone_df, out_dir="output", save=True):
    all_data_df = work_zone_df
    unique_values = list(all_data_df['name'].unique())
    random.Random(42).shuffle(unique_values)
    train_split, test_split, val_split = split(len(unique_values))

    train_df = all_data_df.loc[all_data_df['name'].isin(unique_values[0:train_split])]
    test_df = all_data_df.loc[all_data_df['name'].isin(unique_values[train_split:train_split + test_split])]
    val_df = all_data_df.loc[all_data_df['name'].isin(unique_values[train_split + test_split:])]

    if save:
        sys.stdout.write("Saving csv files...\n")
        os.makedirs(os.path.join(out_dir, 'csv'), exist_ok=True)
        all_data_df.to_csv(os.path.join(out_dir, 'csv', 'all_data.csv'))
        train_df.to_csv(os.path.join(out_dir, 'csv', 'model.csv'))
        test_df.to_csv(os.path.join(out_dir, 'csv', 'test.csv'))
        val_df.to_csv(os.path.join(out_dir, 'csv', 'val.csv'))

    return all_data_df, train_df, test_df, val_df


def merge_and_split_baseline(work_zone_df, non_work_zone_df, out_dir="output", save=True):
    """Merging both datsets"""
    all_data_df = pd.concat([work_zone_df, non_work_zone_df], ignore_index=True, sort=False)
    unique_values = list(all_data_df['name'].unique())
    random.Random(42).shuffle(unique_values)
    train_split, test_split, val_split = split(len(unique_values))

    train_df = all_data_df.loc[all_data_df['name'].isin(unique_values[0:train_split])]
    test_df = all_data_df.loc[all_data_df['name'].isin(unique_values[train_split:train_split + test_split])]
    val_df = all_data_df.loc[all_data_df['name'].isin(unique_values[train_split + test_split:])]

    if save:
        os.makedirs(os.path.join(out_dir, 'csv'), exist_ok=True)
        all_data_df.to_csv(os.path.join(out_dir, 'csv', 'all_data.csv'))
        train_df.to_csv(os.path.join(out_dir, 'csv', 'model.csv'))
        test_df.to_csv(os.path.join(out_dir, 'csv', 'test.csv'))
        val_df.to_csv(os.path.join(out_dir, 'csv', 'val.csv'))
        print("Saving csv files...")

    return all_data_df, train_df, test_df, val_df


def get_configs(path):
    with open(path) as out:
        configs = yaml.load(out, Loader=yaml.FullLoader)

    for key, value in configs.items():
        print(key, ": ", value)

    workzone_dir = configs["WORKZONE_DIR"]
    non_workzone_dir = configs["NONWORKZONE_DIR"]
    output_dir = configs["OUTPUT_DIR"]
    model_name = configs["MODEL"]
    backbone = configs["BACKBONE"]
    epochs = configs["EPOCHS"]
    lr = configs["LR"]
    resize_shape = literal_eval(configs["RESIZE_SHAPE"])
    optimizer = configs["OPTIMIZER"]
    batch_size = configs["BATCH_SIZE"]
    log_step = configs["LOG_STEP"]

    # create output directory
    os.makedirs(output_dir, exist_ok=True)

    return [workzone_dir, non_workzone_dir, output_dir,
            model_name, backbone, epochs, lr,
            resize_shape, optimizer, batch_size, log_step]
