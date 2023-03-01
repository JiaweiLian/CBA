import sys
import time
import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageDraw

from models.common import DetectMultiBackend
from models_yolov3.common import DetectMultiBackend_yolov3
from utils import *
from darknet import *
from load_data import PatchTransformer, PatchApplier, InriaDataset,PatchTransformer_A3
import json

from utils_yolov5.torch_utils import select_device
from utils_yolov5.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                                  increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer,
                                  xyxy2xywh)
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# os.environ["CUDA_VISIBLE_DEVICES"] = '6'

pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", 1000)

if __name__ == '__main__':
    print("Setting everything up")

    # 2080
    # weights_yolov3 = '/home/mnt/ljw305/yolov3/runs/train/exp5/weights/best.pt'
    # data = '/home/mnt/ljw305/yolov3/data/DOTA1_0.yaml'
    # 3080
    weights_yolov3 = "/data1/lianjiawei/yolov3-master/runs/train/exp5/weights/best.pt"
    data = '/data1/lianjiawei/yolov3-master/data/DOTA1_0.yaml'

    imgdir = "datasets/RSOD-Dataset/aircraft/Annotation/JPEGImages"
    device = select_device('')

    #############################################################
    patch_generated_by = 'swin'
    patchfile = "patches/Patch_A3/" + patch_generated_by + ".png"
    savedir = "testing/" + patch_generated_by + "_A3_yolov3"
    #############################################################

    model = DetectMultiBackend_yolov3(weights_yolov3,
                                           device=device,
                                           dnn=False, ).eval()

    patch_applier = PatchApplier().cuda()
    patch_transformer = PatchTransformer_A3().cuda()

    batch_size = 1
    img_size = 1024

    patch_size = 50

    patch_img = Image.open(patchfile).convert('RGB')
    tf = transforms.Resize((patch_size, patch_size))
    patch_img = tf(patch_img)
    tf = transforms.ToTensor()
    adv_patch_cpu = tf(patch_img)
    adv_patch = adv_patch_cpu.cuda()

    clean_results = []
    noise_results = []
    patch_results = []

    def pil2yolov5(image):
        image = np.array(image, dtype=np.float32)
        image = np.transpose(image, (2, 0, 1))
        image = torch.from_numpy(image)
        image = image.unsqueeze(0)
        image /= 255.0
        image = image.cuda()
        return image

    print("Done")
    # Load orginal image
    n = 1
    for imgfile in os.listdir(imgdir):
        print(imgfile, '   ', n)
        n += 1
        if imgfile.endswith('.jpg') or imgfile.endswith('.png'):
            name = os.path.splitext(imgfile)[0]  # image name w/o extension
            txtname = name + '.txt'

            txtpath = os.path.join(savedir, 'clean/', 'labels-yolo/')
            if not os.path.exists(txtpath):
                os.makedirs(txtpath)
            txtpath = os.path.abspath(os.path.join(txtpath, txtname))

            imgfile = os.path.abspath(os.path.join(imgdir, imgfile))
            img = Image.open(imgfile).convert('RGB')
            w, h = img.size
            if w == h:
                padded_img = img
            else:
                dim_to_pad = 1 if w < h else 2
                if dim_to_pad == 1:
                    padding = (h - w) / 2
                    padded_img = Image.new('RGB', (h, h), color=(127, 127, 127))
                    padded_img.paste(img, (int(padding), 0))
                else:
                    padding = (w - h) / 2
                    padded_img = Image.new('RGB', (w, w), color=(127, 127, 127))
                    padded_img.paste(img, (0, int(padding)))
            resize = transforms.Resize((img_size, img_size))  # 1024*1024
            padded_img = resize(padded_img)
            # cleanname = name + ".jpg"

            padded_img_save_path = os.path.join(savedir, 'clean/')
            if not os.path.exists(padded_img_save_path):
                os.makedirs(padded_img_save_path)
            # padded_img.save(os.path.join(padded_img_save_path, cleanname))

            boxes = model(pil2yolov5(padded_img), augment=False, visualize=False)  # TODO:See the details of output
            boxes = non_max_suppression(boxes, conf_thres=0.4, iou_thres=0.45, classes=None, agnostic=False, max_det=1000)

            # Save clean labels-yolo
            textfile = open(txtpath, 'w+')
            for box in boxes[0]:
                cls_id = box[5]
                if cls_id == 0:  # if plane
                    x1 = box[0]
                    y1 = box[1]
                    x2 = box[2]
                    y2 = box[3]
                    width = (x2 - x1) / 1024.0
                    height = (y2 - y1) / 1024.0
                    x_center = (x1 + x2) / 2.0 / 1024.0
                    y_center = (y1 + y2) / 2.0 / 1024.0

                    textfile.write(f'{cls_id} {x_center} {y_center} {width} {height}\n')
                    clean_results.append({'image_id': name, 'bbox': [x_center.item() - width.item() / 2,
                                                                     y_center.item() - height.item() / 2,
                                                                     width.item(),
                                                                     height.item()],
                                          'score': box[4].item(),
                                          'category_id': 1})
            textfile.close()

            # lees deze labelfile terug in als tensor
            if os.path.getsize(txtpath):  # check to see if label file contains data.
                label = np.loadtxt(txtpath)
            else:
                label = np.ones([5])
            label = torch.from_numpy(label).float()
            if label.dim() == 1:
                label = label.unsqueeze(0)

            transform = transforms.ToTensor()
            padded_img = transform(padded_img).cuda()
            img_fake_batch = padded_img.unsqueeze(0)
            lab_fake_batch = label.unsqueeze(0).cuda()

            # transformeer patch en voeg hem toe aan beeld
            adv_batch_t = patch_transformer(adv_patch, lab_fake_batch, img_size, do_rotate=True, rand_loc=False)
            p_img_batch = patch_applier(img_fake_batch, adv_batch_t)
            p_img = p_img_batch.squeeze(0)
            p_img_pil = transforms.ToPILImage('RGB')(p_img.cpu())
            properpatchedname = name + "_p.png"

            proper_patched_img_save_path = os.path.join(savedir, 'proper_patched/')
            if not os.path.exists(proper_patched_img_save_path):
                os.makedirs(proper_patched_img_save_path)
            p_img_pil.save(os.path.join(proper_patched_img_save_path, properpatchedname))

            # genereer een label file voor het beeld met sticker
            txtname = properpatchedname.replace('.png', '.txt')

            txtpath = os.path.join(savedir, 'proper_patched/', 'labels-yolo/')
            if not os.path.exists(txtpath):
                os.makedirs(txtpath)
            txtpath = os.path.abspath(os.path.join(txtpath, txtname))

            boxes = model(pil2yolov5(p_img_pil), augment=False, visualize=False)  # TODO:See the details of output
            boxes = non_max_suppression(boxes, conf_thres=0.4, iou_thres=0.45, classes=None, agnostic=False, max_det=1000)

            textfile = open(txtpath, 'w+')
            for box in boxes[0]:
                cls_id = box[5]
                if cls_id == 0.0:  # if person
                    x1 = box[0]
                    y1 = box[1]
                    x2 = box[2]
                    y2 = box[3]
                    width = (x2 - x1) / 1024.0
                    height = (y2 - y1) / 1024.0
                    x_center = (x1 + x2) / 2.0 / 1024.0
                    y_center = (y1 + y2) / 2.0 / 1024.0

                    textfile.write(f'{cls_id} {x_center} {y_center} {width} {height}\n')
                    patch_results.append({'image_id': name, 'bbox': [x_center.item() - width.item() / 2,
                                                                     y_center.item() - height.item() / 2,
                                                                     width.item(),
                                                                     height.item()],
                                          'score': box[4].item(),
                                          'category_id': 1})
            textfile.close()

    with open(os.path.join(savedir, 'patch_results.json'), 'w') as fp:
        json.dump(patch_results, fp)
