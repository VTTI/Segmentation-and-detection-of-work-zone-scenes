import argparse
import numpy as np
import os
import random
import torch

from model.zone_baseline import RunBaseline
from utils.zone_only_utils import get_configs


def set_seed(seed, device):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(device=device)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-f", help="path to config file", default="./configs/config_baseline.yaml")
    args.add_argument("--mode", "-m", help="mode for baseline model",
                      choices=["train", "test", "test_single", "test_video"], default="train")
    args.add_argument("--comment", "-c", help="comment for training", default="_")
    args.add_argument("--weight", "-w", help="path to model weights", default=None)
    args.add_argument("--device", "-d", help="set device number if you have multiple GPUs", default=0, type=int)
    args.add_argument("--root", "-r", help="root for data files", default="data/test")
    args = args.parse_args()
    return args.config, args.comment, args.mode, args.weight, args.device, args.root


def load_model(path, comment, mode):
    configs = get_configs(path)
    workzone_dir = configs[0]
    non_workzone_dir = configs[1]
    output_dir = configs[2]
    model_name = configs[3]
    print(model_name)
    backbone = configs[4]
    epochs = configs[5]
    lr = configs[6]
    resize_shape = configs[7]
    optimizer = configs[8]
    batch_size = configs[9]
    log_step = configs[10]

    baseline = RunBaseline(comment=comment,
                           workzone_dir=workzone_dir,
                           non_workzone_dir=non_workzone_dir,
                           model_name=model_name,
                           optimizer=optimizer,
                           num_epochs=epochs,
                           batch_size=batch_size,
                           log_step=log_step,
                           out_dir=output_dir,
                           lr=lr,
                           resize_shape=resize_shape,
                           mode=mode)

    return baseline


def main():
    path, comment, mode, weight, device, root = parse_args()
    set_seed(42, device)
    net = load_model(path, comment, mode)
    if mode == "train":
        net.train()
        print("Testing")
        net.test()
    elif mode == "test":
        # make sure that the weights are present in the output folder
        print("Testing")
        net.test()
    elif mode == "test_single":
        net.test_on_single_images(root=root, weight=weight)

    elif mode == "test_video":
        net.test_on_video(root=root, weight=weight)


if __name__ == "__main__":
    main()
