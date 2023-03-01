import json
import warnings

import pandas as pd
from torchvision import transforms
from utils_yolov5.general import (non_max_suppression_mm)
from utils_yolov5.torch_utils import select_device

import patch_config
from darknet import *
from load_data import PatchTransformer, PatchApplier, PatchTransformer_A3
from models.common import DetectMultiBackend
from models_yolov3.common import DetectMultiBackend_yolov3
from utils import *
from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)

warnings.filterwarnings("ignore")

# os.environ["CUDA_VISIBLE_DEVICES"] = '6'

pd.set_option("display.max_rows", 1000)
pd.set_option("display.max_columns", 1000)

if __name__ == '__main__':
    print("Setting everything up")
    ###########################################
    patch_generated_by = 'swin'
    mode = 'swin'
    ###########################################
    config = patch_config.patch_configs[mode]()
    config_file = config.config_file
    checkpoint_file = config.checkpoint_file
    device = 'cuda:0'
    model = init_detector(config_file, checkpoint_file, device=device)
    imgdir = "datasets/RSOD-Dataset/aircraft/Annotation/JPEGImages"
    ############################################################
    patchfile = "patches/Patch_A3/" + patch_generated_by + ".png"
    savedir = "testing/" + patch_generated_by + "_A3_" + mode
    ############################################################

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

            boxes = inference_detector(model, np.asarray(padded_img))  # TODO:See the details of output

            if mode == 'swin':
                boxes = boxes[0]

            # Save clean labels-yolo
            textfile = open(txtpath, 'w+')
            for box in boxes[0]:
                cls_id = 0.0
                if box[4] >= 0.3:  # if plane
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

            boxes = inference_detector(model, np.asarray(p_img_pil))  # TODO:See the details of output

            if mode == 'swin':
                boxes = boxes[0]

            textfile = open(txtpath, 'w+')
            for box in boxes[0]:
                cls_id = 0.0
                if box[4] >= 0.3:  # if person
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
