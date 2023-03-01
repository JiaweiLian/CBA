"""
Adversarial patch training
"""
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import time
import warnings
import pandas as pd
import torch
import wandb
# from tensorboardX import SummaryWriter
from torch import autograd
from torchvision import transforms
from tqdm import tqdm
import patch_config
from load_data import *

warnings.filterwarnings("ignore")

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
        self.InferenceDetector = self.config.InferenceDetector.cuda()
        # self.prob_extractor = MaxProbExtractor(0, 80, self.config).cuda()

        self.patch_applier = PatchApplier().cuda()
        self.patch_transformer = PatchTransformer_A3().cuda()

        # self.nps_calculator = NPSCalculator(self.config.printfile, self.config.patch_size).cuda()
        self.total_variation = TotalVariation().cuda()

    def train(self):
        """
        Optimize a patch to generate an adversarial example.
        :return: Nothing
        """

        # img_size = self.darknet_model.height
        img_size = 1024
        batch_size = self.config.batch_size
        n_epochs = 1000

        # max number of bbox in txt file(namely max number of lines in label file)
        # max_lab = 13 + 1  # person
        # max_lab = 34 + 1  # yolov2
        # max_lab = 50
        max_lab = 59 + 1  # yolov5l

        time_str = time.strftime("%Y%m%d-%H%M%S")

        # Generate stating point
        # adv_patch_cpu = self.generate_patch("gray")  # Training from gray patch
        adv_patch_cpu = self.read_image("patches/patch_aircraft/image_cropped_masked_reversed_beta0.000100.png")  # Training from existing patch
        adv_patch_cpu_original = self.read_image("patches/patch_aircraft/image_cropped_masked_reversed_beta0.000100.png")
        adv_patch_mask_cpu = self.read_image("patches/patch_aircraft/aircraft_SOD4_cropped_binarization_thresh100_150.png")
        adv_patch_reversed_mask_cpu = self.read_image("patches/patch_aircraft/aircraft_SOD4_cropped_binarization_thresh100_150_reversed.png")
        adv_patch_cpu.requires_grad_(True)  # 都加个这试试

        train_loader = torch.utils.data.DataLoader(
            InriaDataset(self.config.img_dir, self.config.lab_dir, max_lab, img_size, shuffle=False),
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

        alpha = 150  # default=1.5
        wandb.log({
            "alpha": alpha,
            "Detector": self.mode,
            "Patch generation": "A3",
            "Patch size": self.config.patch_size
        })

        et0 = time.time()
        for epoch in range(n_epochs):
            ep_det_loss = 0
            # ep_nps_loss = 0
            ep_tv_loss = 0
            ep_loss = 0
            bt0 = time.time()
            for i_batch, (img_batch, lab_batch) in tqdm(enumerate(train_loader), desc=f'Running epoch {epoch}',
                                                        total=self.epoch_length):
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
                    # print(p_img_batch[0], p_img_batch[0].size())  # Tensor: torch.Size([1, 3, 1024, 1024])

                    p_img_batch_cpu = p_img_batch.clone()
                    p_img_batch_cpu = p_img_batch_cpu[0].detach().cpu().numpy()
                    p_img_batch_cpu = p_img_batch_cpu.reshape(1024, 1024, 3)
                    p_img_batch_cpu = p_img_batch_cpu * 255

                    data = self.InferenceDetector(self.model, p_img_batch_cpu)
                    data['img'][0] = p_img_batch
                    output = self.model(return_loss=False, rescale=True, **data)
                    # print(output)
                    #########################################################
                    if self.mode in ['ssd', 'faster_rcnn', 'cascade_rcnn', 'retinanet', 'foveabox', 'free_anchor', 'fsaf', 'reppoints', 'deformable_detr', 'tood', 'atss', 'vfnet']:
                        if len(output[0][0]) == 0:
                            # max_prob = torch.tensor(float(0))
                            mean_prob = torch.tensor(float(0))
                        else:
                            # max_prob = output[0][0][0][4]
                            mean_prob = torch.mean(output[0][0][:, 4])
                    elif self.mode in ['swin', 'mask_rcnn']:
                        if len(output[0][0][0]) == 0:
                            # max_prob = torch.tensor(float(0))
                            mean_prob = torch.tensor(float(0))
                        else:
                            # max_prob = output[0][0][0][0][4]
                            mean_prob = torch.mean(output[0][0][0][:, 4])
                    #########################################################
                    # nps = self.nps_calculator(adv_patch)
                    tv = self.total_variation(adv_patch)

                    tv_loss = tv * alpha
                    # nps_loss = nps * beta

                    #################################
                    # det_loss = torch.mean(max_prob)
                    det_loss = torch.mean(mean_prob)
                    #################################
                    # print(det_loss)
                    ##############################################################################
                    loss = det_loss + torch.max(tv_loss, torch.tensor(0.1).cuda())
                    # loss = det_loss + torch.max(tv_loss, torch.tensor(0.1).cuda()) * 0.0  # 消融实验不能直接去掉Ltv，乘0.0即可
                    ##############################################################################

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
            ep_det_loss = ep_det_loss / len(train_loader)
            ep_tv_loss = ep_tv_loss / len(train_loader)
            ep_loss = ep_loss / len(train_loader)

            scheduler.step(ep_loss)
            if True:
                print('  EPOCH NR: ', epoch),
                print('EPOCH LOSS: ', ep_loss)
                print('  DET LOSS: ', ep_det_loss)
                # print('  NPS LOSS: ', ep_nps_loss)
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
        tf = transforms.ToTensor()  # 会将图像归一化
        adv_patch_cpu = tf(patch_img)
        return adv_patch_cpu


def main():
    trainer = PatchTrainer('ssd')
    trainer.train()


if __name__ == '__main__':
    main()

# CUDA_VISIBLE_DEVICES=3 python train_patch_A3_mmdetection.py
# pip install wandb --upgrade
