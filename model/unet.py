import glob
import logging
import os
import sys

import PIL
import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from matplotlib.lines import Line2D
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from datasets.dataset_unet import dataset_unet

plt.rcParams['figure.dpi'] = 100


class RunUNet:
    def __init__(self, comment, workzone_dir, non_workzone_dir, model_backbone, optimizer, num_epochs, batch_size,
                 log_step, out_dir, lr, resize_shape, mode="train"):

        self.comment = f"LR_{lr}_OPT_{optimizer}_BATCH_{batch_size}_SHAPE_{resize_shape}_{comment}"
        self.workzone_dir = workzone_dir
        self.non_workzone_dir = non_workzone_dir
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.log_step = log_step
        self.out_dir = os.path.join(out_dir, model_backbone)
        self.lr = lr
        self.resize_shape = resize_shape
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_backbone = model_backbone
        self.model = smp.Unet(self.model_backbone, encoder_weights="imagenet", classes=1, activation=None).to(
            self.device)

        self.writer = SummaryWriter(log_dir=os.path.join(self.out_dir, 'runs', f'{self.model_backbone}_{self.comment}'))
        self.opt = self.get_optimizer(self.optimizer, self.model, self.lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.opt, 'max', patience=3, verbose=True)

        # data loaders
        if (mode == "train") or (mode == "test"):
            self.train_set, self.test_set, self.val_set = dataset_unet(self.workzone_dir,
                                                                       out_dir=self.out_dir,
                                                                       resize_shape=self.resize_shape)
            self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=8)
            self.val_loader = DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=8)
            self.test_loader = DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=8)

        # relevant initializations
        os.makedirs(os.path.join(self.out_dir, 'weights', self.comment), exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, 'logs', self.comment), exist_ok=True)
        self.best_model_path = os.path.join(self.out_dir, 'weights', self.comment, f"unet_best_weight.pth")

    def train(self):
        # initialize logger
        filename = os.path.join(self.out_dir, 'logs', self.comment, f'{self.comment}_train.log')
        logging.basicConfig(filename=filename, filemode='w',
                            format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s', level=logging.INFO)
        # define a Handler which writes INFO messages or higher to the sys.stderr
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        # set a format which is simpler for console use
        formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
        # tell the handler to use this format
        console.setFormatter(formatter)
        # add the handler to the root logger
        logging.getLogger('').addHandler(console)

        logging.info("---Beginning Training---")
        logging.info(f"Total number of training example: {len(self.train_set)}")
        logging.info(f"Total number of validation example: {len(self.val_set)}")

        # initializations
        total_step = len(self.train_loader)
        best_val_score = None

        logging.info(
            f"Training Configurations model backbone: {self.model_backbone}; "
            f"optimizer: {self.optimizer}; lr: {self.lr}; "
            f"num epochs: {self.num_epochs}; batch size: {self.batch_size}; device:{self.device}")

        for epoch in range(self.num_epochs):
            logging.info(f"--- Epoch {epoch + 1} ---")
            self.model.train()

            for itr, data in enumerate(self.train_loader):
                self.opt.zero_grad()
                image_tensor = data["image"]
                true_masks = data["mask"]
                mask_weights = data["mask_weight"]

                image_tensor = image_tensor.to(device=self.device, dtype=torch.float32)  # (batch, C, H, W)
                true_masks = true_masks.to(device=self.device, dtype=torch.float32)  # (batch, 1, H, W)
                mask_weights = mask_weights.to(device=self.device,
                                               dtype=torch.float32)  # (batch, 1, H, W)

                # prediction
                pred_masks = self.model(image_tensor)  # (batch, 1, H, W)

                # loss
                loss = self.dice_bce_loss(pred_masks, true_masks, mask_weights)

                # backward pass
                loss.backward()
                # self.plot_grad_flow(self.model.named_parameters())

                self.opt.step()

                # log performance
                if itr % self.log_step == 0:
                    logging.info(
                        f'Running logs for epoch: {epoch + 1} '
                        f'Step: {itr}/{total_step} ---> Loss: {round(loss.item(), 4)}')

            logging.info('---Performing Validation---')
            train_score = self.eval(self.train_loader)  # dice coeff
            val_score = self.eval(self.val_loader)  # dice coeff

            # logging output
            logging.info(f'Training--->Score: {round(train_score, 4)}')
            logging.info(f'Validation--->Score: {round(val_score, 4)}')

            # updating best score
            if best_val_score is None:
                best_val_score = val_score
                torch.save(self.model.state_dict(), self.best_model_path)
                logging.info('model saved')
            elif val_score > best_val_score:
                best_val_score = val_score
                logging.info(f"Updating best val score to: {round(best_val_score, 4)}")
                torch.save(self.model.state_dict(), self.best_model_path)
                logging.info('model saved')

            # scheduler update
            self.scheduler.step(val_score)
            # summary writer
            self.writer.add_scalars('Score', {'model score': train_score, 'val score': val_score}, epoch + 1)
            logging.info("-----------------------------------------------------")
        self.writer.close()

    def test(self):

        print(f'---Performing testing on test set---')
        print(f'Total number of testing example: {len(self.test_set)}')

        test_score = self.eval(self.test_loader, _test_=True, path=self.best_model_path)
        print(f'Testing--->Score: {round(test_score, 4)}')

    def eval(self, loader, _test_=False, path=None):
        total_step = len(loader)
        score = 0

        if _test_ is True:
            print("Loading weights for testing")
            try:
                self.model.load_state_dict(torch.load(path, map_location=self.device))
            except Exception as e:
                print(e)
                print("Enter valid path of the model weights!")

        for data in tqdm(loader, file=sys.stdout):
            image_tensor, true_masks = data['image'], data['mask']
            image_tensor = image_tensor.to(device=self.device, dtype=torch.float32)  # (batch, C, H, W)
            true_masks = true_masks.to(device=self.device, dtype=torch.float32)  # (bacth, 1, H, W)
            self.model.eval()
            with torch.no_grad():
                pred_masks = self.model(image_tensor)
            score += self.dice_coeff(pred_masks, true_masks).item()

        return score / total_step

    @staticmethod
    def dice_bce_loss(inputs, targets, mask_weights, smooth=1., delta=1.):

        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        mask_weights = mask_weights.view(-1)

        # dice loss
        intersection = (inputs * targets).sum()
        dice_score = (2. * intersection + smooth) / (torch.square(inputs).sum() + torch.square(targets).sum() + smooth)

        # weighted bce loss
        bce_loss = F.binary_cross_entropy(inputs, targets, weight=delta * mask_weights,
                                          reduction='mean') + F.binary_cross_entropy(inputs, targets,
                                                                                     weight=delta * targets,
                                                                                     reduction='mean')
        return torch.clamp(-torch.log(dice_score), max=100) + bce_loss

    @staticmethod
    def plot_grad_flow(named_parameters):
        """Plots the gradients flowing through different layers in the net during training.
        Can be used for checking for possible gradient vanishing / exploding problems.

        Usage: Plug this function in Trainer class after loss.backwards() as
        "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow"""

        ave_grads = []
        max_grads = []
        layers = []
        for n, p in named_parameters:
            if p.requires_grad and ("bias" not in n):
                layers.append(n)
                ave_grads.append(p.grad.abs().mean())
                max_grads.append(p.grad.abs().max())
        plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.5, lw=4, color="c")
        plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.5, lw=1, color="b")
        plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
        plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(left=0, right=len(ave_grads))
        plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow")
        plt.grid(True)
        plt.legend([Line2D([0], [0], color="c", lw=4),
                    Line2D([0], [0], color="b", lw=4),
                    Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])

        plt.show()

    @staticmethod
    def dice_coeff(inputs, targets, smooth=1):  # calculating dice score for each batch
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)
        inputs = (inputs > 0.5).float()

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (torch.square(inputs).sum() + torch.square(targets).sum() + smooth)
        return dice

    @staticmethod
    def get_optimizer(optimizer, network, lr):
        if optimizer == 'ADAM':
            opt = optim.Adam(network.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=1e-4)
        elif optimizer == 'ASGD':
            opt = optim.ASGD(network.parameters(), lr=lr)
        elif optimizer == 'SGD':
            opt = optim.SGD(network.parameters(), lr=lr, momentum=0.99)
        elif optimizer == 'RMSprop':
            opt = optim.RMSprop(network.parameters(), lr=lr, momentum=0.99)
        elif optimizer == 'AdamW':
            opt = optim.AdamW(network.parameters(), lr=lr, weight_decay=1e-5)
        else:
            opt = optim.SGD(network.parameters(), lr=lr, momentum=0.99)
        return opt

    def test_on_single_images(self, root="./data/test/**/*.jpg", weight=None):
        test_data = list(glob.iglob(root, recursive=True))
        os.makedirs(os.path.join(self.out_dir, 'predictions', self.comment), exist_ok=True)
        if weight is None:  # no weight given, use default weights
            weight = self.best_model_path

        width, height = self.resize_shape
        self.model.load_state_dict(torch.load(weight, map_location=self.device))

        for path in tqdm(test_data):
            image = Image.open(path)
            image = image.resize((width, height), PIL.Image.BICUBIC)
            image = TF.to_tensor(image).unsqueeze(0).to(self.device)
            self.model.eval()
            with torch.no_grad():
                pred = self.model(image)
            pred = torch.sigmoid(pred)
            pred = (pred > 0.5).float()
            pred_image = make_grid(pred, pad_value=1, normalize=True)
            pred_image = transforms.ToPILImage()(pred_image)
            pred_image.save(os.path.join(self.out_dir, 'predictions', self.comment, path.split(os.sep)[-1].replace(".jpg", "_predicted.jpg")))
