import glob
import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw
from matplotlib.lines import Line2D
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from datasets.dataset_unet_patch import dataset_unet_patch

plt.rcParams['figure.dpi'] = 100


class RunUnetPatch:
    def __init__(self, comment, workzone_dir, non_workzone_dir, model_backbone, optimizer, num_epochs, batch_size,
                 log_step, out_dir, lr, crop_shape, mode):

        self.comment = f"LR_{lr}_OPT_{optimizer}_BATCH_{batch_size}_SHAPE_{crop_shape}_{comment}"
        self.workzone_dir = workzone_dir
        self.non_workzone_dir = non_workzone_dir
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.log_step = log_step
        self.out_dir = os.path.join(out_dir, model_backbone)
        self.lr = lr
        self.crop_shape = crop_shape
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_backbone = model_backbone
        self.model = smp.Unet(self.model_backbone, encoder_weights="imagenet", classes=1, activation=None).to(
            self.device)

        self.opt = self.get_optimizer(self.optimizer, self.model, self.lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.opt, 'max', patience=12, verbose=True, min_lr=1e-5)
        self.writer = SummaryWriter(log_dir=os.path.join(self.out_dir, 'runs', f'{self.model_backbone}_{self.comment}'))

        # data loaders
        if (mode == "train") or (mode == "test"):
            self.train_set, self.test_set, self.val_set = dataset_unet_patch(self.workzone_dir,
                                                                             out_dir=self.out_dir,
                                                                             crop_shape=self.crop_shape)
            self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=8)
            self.val_loader = DataLoader(self.val_set, batch_size=1, shuffle=False, num_workers=8)
            self.test_loader = DataLoader(self.test_set, batch_size=1, shuffle=False, num_workers=8)

        # relevant initializations
        os.makedirs(os.path.join(self.out_dir, 'weights', self.comment), exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, 'logs', self.comment), exist_ok=True)
        self.best_model_path = os.path.join(self.out_dir, 'weights', self.comment, f"{self.comment}_best_weight.pth")

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

    @staticmethod
    def normalize(image):
        return TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    @staticmethod
    def get_patches(image, ismask=False):
        kh, kw = (224, 224)  # kernel size
        dh, dw = (224, 224)  # stride

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
        # transform to tensor
        image, mask = TF.to_tensor(image).unsqueeze(0), TF.to_tensor(mask).unsqueeze(0)
        # normalize
        image = self.normalize(image)
        image = self.get_patches(image)
        mask = self.get_patches(mask, ismask=True)
        return image, mask

    @staticmethod
    def dice_bce_loss(inputs, targets, mask_weights, smooth=1, delta=1.):

        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        mask_weights = mask_weights.view(-1)

        # dice loss
        intersection = (inputs * targets).sum()
        dice_score = (2. * intersection + smooth) / (
                torch.sum(torch.square(inputs)) + torch.sum(torch.square(targets)) + smooth)

        # weighted bce loss
        bce_mask_loss = F.binary_cross_entropy(inputs, targets, weight=delta * targets, reduction='mean')
        bce_edge_loss = F.binary_cross_entropy(inputs, targets, weight=delta * mask_weights, reduction='mean')
        dice_loss = torch.clamp(-torch.log(dice_score), max=100)
        return dice_loss + bce_edge_loss + bce_mask_loss

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
        dice = (2. * intersection + smooth) / (torch.sum(inputs) + torch.sum(targets) + smooth)
        # print(torch.sum(inputs), inputs.sum())
        return dice

    @staticmethod
    def iou(inputs, targets, eps=1e-5):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)
        inputs = (inputs > 0.5).float()

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = torch.logical_and(inputs, targets) * 1.0
        union = torch.logical_or(inputs, targets) * 1.0
        iou_score = torch.sum(intersection) / (torch.sum(union) + eps)
        # print(torch.sum(intersection).item(), torch.sum(union).item(), iou_score.item())
        return iou_score

    @staticmethod
    def get_optimizer(optimizer, network, lr):
        if optimizer == 'ADAM':
            opt = optim.Adam(network.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=5e-5)
        elif optimizer == 'ASGD':
            opt = optim.ASGD(network.parameters(), lr=lr)
        elif optimizer == 'SGD':
            opt = optim.SGD(network.parameters(), lr=lr, momentum=0.99)
        elif optimizer == 'RMSprop':
            opt = optim.RMSprop(network.parameters(), lr=lr, momentum=0.99)
        elif optimizer == 'AdamW':
            opt = optim.AdamW(network.parameters(), lr=lr, weight_decay=5e-5)
        else:
            opt = optim.SGD(network.parameters(), lr=lr, momentum=0.99)

        return opt

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
            f"Training Configurations---> model backbone: {self.model_backbone}; optimizer: {self.optimizer}; "
            f"lr: {self.lr}; num epochs: {self.num_epochs}; batch size: {self.batch_size}; device:{self.device}")

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
                mask_weights = mask_weights.to(device=self.device, dtype=torch.float32)  # (batch, 1, H, W)

                # prediction
                pred_masks = self.model(image_tensor)  # (batch, 1, H, W)

                # loss
                loss = self.dice_bce_loss(pred_masks, true_masks, mask_weights)  # change

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
            train_score, _ = self.eval(self.train_loader)  # dice coeff
            val_score, _ = self.eval(self.val_loader)  # dice coeff

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

        DataLoader(dataset=self.test_set, batch_size=1, shuffle=False)
        score_dice, score_iou = self.eval(self.test_loader, split="test", path=self.best_model_path)
        print(f'Testing--->Dice score: {round(score_dice, 4)}, mIoU:{round(score_iou, 4)}')

    def eval(self, loader, split="train", path=None):

        h = self.crop_shape[0]
        w = self.crop_shape[1]
        total_step = len(loader)
        score_dice = 0
        score_iou = 0

        if split == "test":
            print("Loading weights for testing")
            try:
                self.model.load_state_dict(torch.load(path, map_location=self.device))
            except Exception as e:
                print(e)
                print("Enter valid path of the model weights!")

        for data in tqdm(loader, file=sys.stdout):
            image_tensor, true_masks = data['image'].view(-1, 3, h, w), data['mask'].view(-1, 1, h, w)
            image_tensor = image_tensor.to(device=self.device, dtype=torch.float32)  # (batch, 3, H, W)
            true_masks = true_masks.to(device=self.device, dtype=torch.float32)  # (bacth, 1, H, W)
            self.model.eval()
            with torch.no_grad():
                pred_masks = self.model(image_tensor)
            score_dice += self.dice_coeff(pred_masks, true_masks).item()
            score_iou += self.iou(pred_masks, true_masks).item()

        return score_dice / total_step, score_iou / total_step

    def test_on_single_images(self, root="./data/test/**/*.jpg", weight=None):
        test_data = list(glob.iglob(root, recursive=True))
        os.makedirs(os.path.join(self.out_dir, 'predictions', self.comment), exist_ok=True)
        if weight is None:  # no weight given, use default weights
            weight = self.best_model_path

        self.model.load_state_dict(torch.load(weight, map_location=self.device))

        for path in tqdm(test_data):
            image = Image.open(path)
            image_shape = image.size
            points = []
            mask = self.get_mask(points, image_shape)
            image, mask = self.preprocess(image, mask)
            rgb_image = make_grid(image, pad_value=1, nrow=3, normalize=True)
            rgb_image = transforms.ToPILImage()(rgb_image)
            rgb_image.save(os.path.join(self.out_dir, 'predictions', self.comment,
                                        path.split(os.sep)[-1].replace(".jpg", "_patch.jpg")))
            self.model.eval()
            with torch.no_grad():
                pred = self.model(image.to(self.device))
            pred = torch.sigmoid(pred)
            pred = (pred > 0.5).float()
            pred_image = make_grid(pred, pad_value=1, nrow=3, normalize=True)
            pred_image = transforms.ToPILImage()(pred_image)
            pred_image.save(os.path.join(self.out_dir, 'predictions', self.comment,
                                         path.split(os.sep)[-1].replace(".jpg", "_predicted_patch.jpg")))
