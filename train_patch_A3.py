"""
Adversarial patch training
"""

import PIL
import torch
import load_data
from tqdm import tqdm
from load_data import *
import gc
import matplotlib.pyplot as plt
from torch import autograd
from torchvision import transforms
import subprocess
import patch_config
import sys
import time
import os
import pandas as pd
import wandb
import warnings

warnings.filterwarnings("ignore")

# os.environ["CUDA_VISIBLE_DEVICES"] = '6'
#
# Expand to show
pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", 1000)


def adv_patch_update(adv_patch_cpu, adv_patch_cpu_original, adv_patch_mask_cpu, adv_patch_reversed_mask_cpu):
    aircraft_area = torch.mul(adv_patch_cpu_original, adv_patch_mask_cpu)
    adv_patch_area = torch.mul(adv_patch_cpu, adv_patch_reversed_mask_cpu)
    adv_patch_cpu = torch.add(input=aircraft_area, alpha=1, other=adv_patch_area)
    return adv_patch_cpu


class PatchTrainer(object):
    def __init__(self, mode):
        self.mode = mode
        self.epoch_length = 0
        self.config = patch_config.patch_configs[mode]()

        self.model = self.config.model.eval().cuda()
        self.prob_extractor = self.config.prob_extractor.cuda()
        # self.prob_extractor = MaxProbExtractor(0, 80, self.config).cuda()

        self.patch_applier = PatchApplier().cuda()
        self.patch_transformer = PatchTransformer().cuda()

        # self.nps_calculator = NPSCalculator(self.config.printfile, self.config.patch_size).cuda()
        self.total_variation = TotalVariation().cuda()

    def train(self):
        """
        Optimize a patch to generate an adversarial example.
        :return: Nothing
        """

        # img_size = self.darknet_model.height
        img_size = self.config.img_size
        batch_size = self.config.batch_size
        n_epochs = 1000
        max_lab = 59 + 1  # 5l
        # Generate stating point
        adv_patch_cpu = self.read_image("patches/patch_aircraft/image_cropped_masked_reversed_beta0.000100.png")  # Training from existing patch
        adv_patch_cpu_original = self.read_image("patches/patch_aircraft/image_cropped_masked_reversed_beta0.000100.png")
        adv_patch_mask_cpu = self.read_image("patches/patch_aircraft/aircraft_SOD4_cropped_binarization_thresh100_150.png")
        adv_patch_reversed_mask_cpu = self.read_image("patches/patch_aircraft/aircraft_SOD4_cropped_binarization_thresh100_150_reversed.png")
        adv_patch_cpu.requires_grad_(True)

        train_loader = torch.utils.data.DataLoader(
            InriaDataset(self.config.img_dir, self.config.lab_dir, max_lab, img_size, shuffle=True),
            batch_size=batch_size,
            shuffle=True,
            num_workers=10)
        self.epoch_length = len(train_loader)
        print(f'One epoch is {len(train_loader)}')

        optimizer = optim.Adam([adv_patch_cpu], lr=self.config.start_learning_rate, amsgrad=True)
        scheduler = self.config.scheduler_factory(optimizer)

        wandb.init(project="Adversarial-attack")
        wandb.watch_called = False  # Re-run the model without restarting the runtime, unnecessary after our next release
        wandb.watch(self.model, log="all")

        alpha = 1.5  # default_value = 2.5, best = 0.25
        wandb.log({
            "alpha": alpha,
            "Detector": self.mode,
            "Patch generation": "A3",
            "Patch size": self.config.patch_size
        })

        et0 = time.time()
        for epoch in range(n_epochs):
            ep_det_loss = 0
            ep_tv_loss = 0
            ep_loss = 0
            for i_batch, (img_batch, lab_batch) in tqdm(enumerate(train_loader), desc=f'Running epoch {epoch}', total=self.epoch_length):
                iteration = self.epoch_length * epoch + i_batch
                with autograd.detect_anomaly():
                    img_batch = img_batch.cuda()
                    lab_batch = lab_batch.cuda()
                    adv_patch = adv_patch_cpu.cuda()
                    adv_patch_original = adv_patch_cpu_original.cuda()
                    adv_patch_mask = adv_patch_mask_cpu.cuda()
                    adv_patch_reversed_mask = adv_patch_reversed_mask_cpu.cuda()
                    adv_patch = adv_patch_update(adv_patch, adv_patch_original, adv_patch_mask, adv_patch_reversed_mask)

                    adv_batch_t = self.patch_transformer(adv_patch, lab_batch, img_size, do_rotate=True, rand_loc=False)
                    p_img_batch = self.patch_applier(img_batch, adv_batch_t)
                    p_img_batch = F.interpolate(p_img_batch, (img_size, img_size))

                    output = self.model(p_img_batch)
                    # print(output, output.size())  # yolov2：torch.Size([1, 100, 32, 32]) 100 = (15 + 4 + 1) * 5 = (类别数 + 坐标 + 置信度) * 锚框数

                    extracted_prob = self.prob_extractor(output)
                    # print(max_prob, max_prob.size())  # tensor([0.81629], device='cuda:0', grad_fn=<MaxBackward0>) torch.Size([1])

                    tv = self.total_variation(adv_patch)
                    tv_loss = tv * alpha
                    det_loss = torch.mean(extracted_prob)
                    loss = det_loss + torch.max(tv_loss, torch.tensor(0.1).cuda())

                    ep_det_loss += det_loss.detach().cpu().numpy()
                    ep_tv_loss += tv_loss.detach().cpu().numpy()
                    ep_loss += loss

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    adv_patch_cpu.data.clamp_(0, 1)  # keep patch in image range

                    bt1 = time.time()
                    if i_batch % 100 == 0:
                        wandb.log({
                            "Patches": wandb.Image(adv_patch_cpu, caption="patch{}".format(iteration)),
                            "tv_loss": tv_loss,
                            "det_loss": det_loss,
                            "total_loss": loss,
                        })
                    if i_batch + 1 >= len(train_loader):
                        print('\n')
            et1 = time.time()
            # print(ep_det_loss, len(train_loader))
            ep_det_loss = ep_det_loss / len(train_loader)
            ep_tv_loss = ep_tv_loss / len(train_loader)
            ep_loss = ep_loss / len(train_loader)

            # im = transforms.ToPILImage('RGB')(adv_patch_cpu)
            # plt.imshow(im)
            # plt.savefig(f'pics/{time_str}_{self.config.patch_name}_{epoch}.png')

            scheduler.step(ep_loss)
            if True:
                print('  EPOCH NR: ', epoch),
                print('EPOCH LOSS: ', ep_loss)
                print('  DET LOSS: ', ep_det_loss)
                print('   TV LOSS: ', ep_tv_loss)
                print('EPOCH TIME: ', et1 - et0)
            et0 = time.time()

    def generate_patch(self, type):
        """
        Generate a random patch as a starting point for optimization.

        :param type: Can be 'gray' or 'random'. Whether or not generate a gray or a random patch.
        :return:
        """
        if type == 'gray':
            adv_patch_cpu = torch.full((3, self.config.patch_size, self.config.patch_size), 0.5)
        elif type == 'random':
            adv_patch_cpu = torch.rand((3, self.config.patch_size, self.config.patch_size))

        return adv_patch_cpu

    def read_image(self, path):
        """
        Read an input image to be used as a patch

        :param path: Path to the image to be read.
        :return: Returns the transformed patch as a pytorch Tensor.
        """
        patch_img = Image.open(path).convert('RGB')
        tf = transforms.Resize((self.config.patch_size, self.config.patch_size))
        patch_img = tf(patch_img)
        tf = transforms.ToTensor()
        adv_patch_cpu = tf(patch_img)
        return adv_patch_cpu


def main():
    trainer = PatchTrainer('yolov5n')
    trainer.train()


if __name__ == '__main__':
    main()
# CUDA_VISIBLE_DEVICES=0,1 python train_patch.py
