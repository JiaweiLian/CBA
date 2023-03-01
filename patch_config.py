from torch import optim
from darknet import *
from load_data import MaxProbExtractor_yolov2, MaxProbExtractor_yolov5, MeanProbExtractor_yolov5, \
    MeanProbExtractor_yolov2
from mmdet.apis.inference import InferenceDetector
from models.common import DetectMultiBackend
from models_yolov3.common import DetectMultiBackend_yolov3
from utils_yolov5.torch_utils import select_device, time_sync
import os
from mmdet.apis import (async_inference_detector, InferenceDetector,
                        init_detector, show_result_pyplot)


class BaseConfig(object):
    """
    Default parameters for all config files.
    """

    def __init__(self):
        """
        Set the defaults.
        """
        # self.img_dir = "datasets/RSOD-Dataset/aircraft/Annotation/JPEGImages"
        self.img_dir = "testing/plane_random_loc/clean"
        # self.lab_dir = "datasets/RSOD-Dataset/aircraft/Annotation/JPEGImages/labels-yolo"
        self.lab_dir = "testing/yolov5l_center_150_1024_yolov5l/clean/labels-yolo"
        self.printfile = "non_printability/30values.txt"
        self.patch_size = 150
        self.start_learning_rate = 0.03
        self.patch_name = 'base'
        self.scheduler_factory = lambda x: optim.lr_scheduler.ReduceLROnPlateau(x, 'min', patience=50)
        self.max_tv = 0.165
        self.batch_size = 1
        self.img_size = 1024
        self.imgsz = (1024, 1024)
        self.conf_thres = 0.4  # confidence threshold
        self.iou_thres = 0.45  # NMS IOU threshold
        self.max_det = 1000  # maximum detections per image
        self.classes = None  # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms = False  # class-agnostic NMS
        # self.loss_target = lambda obj, cls: obj * cls  # self.loss_target(obj, cls) return obj * cls
        self.loss_target = lambda obj, cls: obj


class yolov2(BaseConfig):
    """
    Reproduce the results from the paper: Generate a patch that minimises object score.
    """

    def __init__(self):
        super().__init__()
        self.cfgfile = "cfg/yolo-dota.cfg"
        self.weightfile = "weights/yolo-dota.cfg_450000.weights"
        self.patch_name = 'ObjectOnlyPaper'
        self.model = Darknet(self.cfgfile)
        self.model.load_weights(self.weightfile)
        self.model = self.model.eval().cuda()
        # self.prob_extractor = MaxProbExtractor_yolov2(0, 15, self.loss_target)
        self.prob_extractor = MeanProbExtractor_yolov2(0, 15, self.loss_target, self.conf_thres, self.iou_thres,
                                                       self.max_det, self.model)


class yolov3(BaseConfig):
    """
    Reproduce the results from the paper: Generate a patch that minimises object score.
    """

    def __init__(self):
        super().__init__()
        self.patch_name = 'ObjectOnlyPaper'
        # 2080
        # self.weights_yolov3 = '/home/mnt/ljw305/yolov3/runs/train/exp5/weights/best.pt'
        # self.data = '/home/mnt/ljw305/yolov3/data/DOTA1_0.yaml'
        # 3080
        # self.weights_yolov3 = "/home/ljw/yolov3-master/runs/train/exp5/weights/best.pt"
        # self.data = '/home/ljw/yolov3-master/data/DOTA1_0.yaml'
        # 3090_2
        self.weights_yolov3 = "/data1/lianjiawei/yolov3-master/runs/train/exp5/weights/best.pt"
        self.data = '/data1/lianjiawei/yolov3-master/data/DOTA1_0.yaml'
        self.device = select_device('')
        self.model = DetectMultiBackend_yolov3(self.weights_yolov3,
                                               device=self.device,
                                               dnn=False, ).eval()
        # self.prob_extractor = MaxProbExtractor_yolov5(0, 15, self.loss_target)
        self.prob_extractor = MeanProbExtractor_yolov5(0, 15, self.loss_target, self.conf_thres, self.iou_thres,
                                                       self.max_det)


class yolov5n(BaseConfig):
    """
    Reproduce the results from the paper: Generate a patch that minimises object score.
    """

    def __init__(self):
        super().__init__()

        self.patch_name = 'ObjectOnlyPaper'
        # 2080
        # self.weights_yolov5 = '/home/mnt/ljw305/yolov5/runs/train/yolov5n/weights/best.pt'
        # self.data = '/home/mnt/ljw305/yolov5/data/DOTA1_0.yaml'
        # 3080
        self.weights_yolov5 = '/data1/lianjiawei/yolov5/runs/train/yolov5n/weights/best.pt'
        self.data = '/data1/lianjiawei/yolov5/data/DOTA1_0.yaml'
        self.device = select_device('')
        self.model = DetectMultiBackend(self.weights_yolov5,
                                        device=self.device,
                                        dnn=False,
                                        data=self.data).eval()
        # self.prob_extractor = MaxProbExtractor_yolov5(0, 15, self.loss_target)
        self.prob_extractor = MeanProbExtractor_yolov5(0, 15, self.loss_target, self.conf_thres, self.iou_thres,
                                                       self.max_det)


class yolov5s(BaseConfig):
    """
    Reproduce the results from the paper: Generate a patch that minimises object score.
    """

    def __init__(self):
        super().__init__()

        self.patch_name = 'ObjectOnlyPaper'
        # 2080
        # self.weights_yolov5 = '/home/mnt/ljw305/yolov5/runs/train/yolov5s/weights/best.pt'
        # self.data = '/home/mnt/ljw305/yolov5/data/DOTA1_0.yaml'
        # 3080
        self.weights_yolov5 = '/data1/lianjiawei/yolov5/runs/train/yolov5s/weights/best.pt'
        self.data = '/data1/lianjiawei/yolov5/data/DOTA1_0.yaml'
        self.device = select_device('')
        self.model = DetectMultiBackend(self.weights_yolov5,
                                        device=self.device,
                                        dnn=False,
                                        data=self.data).eval()
        # self.prob_extractor = MaxProbExtractor_yolov5(0, 15, self.loss_target)
        self.prob_extractor = MeanProbExtractor_yolov5(0, 15, self.loss_target, self.conf_thres, self.iou_thres,
                                                       self.max_det)


class yolov5m(BaseConfig):
    """
    Reproduce the results from the paper: Generate a patch that minimises object score.
    """

    def __init__(self):
        super().__init__()

        self.patch_name = 'ObjectOnlyPaper'
        # 2080
        # self.weights_yolov5 = '/home/mnt/ljw305/yolov5/runs/train/yolov5m/weights/best.pt'
        # self.data = '/home/mnt/ljw305/yolov5/data/DOTA1_0.yaml'
        # 3080
        self.weights_yolov5 = '/data1/lianjiawei/yolov5/runs/train/yolov5m/weights/best.pt'
        self.data = '/data1/lianjiawei/yolov5/data/DOTA1_0.yaml'
        self.device = select_device('')
        self.model = DetectMultiBackend(self.weights_yolov5,
                                        device=self.device,
                                        dnn=False,
                                        data=self.data).eval()
        # self.prob_extractor = MaxProbExtractor_yolov5(0, 15, self.loss_target)
        self.prob_extractor = MeanProbExtractor_yolov5(0, 15, self.loss_target, self.conf_thres, self.iou_thres,
                                                       self.max_det)


class yolov5l(BaseConfig):
    """
    Reproduce the results from the paper: Generate a patch that minimises object score.
    """

    def __init__(self):
        super().__init__()

        self.patch_name = 'ObjectOnlyPaper'
        # 2080
        # self.weights_yolov5 = '/home/mnt/ljw305/yolov5/runs/train/yolov5l/weights/best.pt'
        # self.data = '/home/mnt/ljw305/yolov5/data/DOTA1_0.yaml'
        # 3080
        self.weights_yolov5 = '/data1/lianjiawei/yolov5/runs/train/yolov5l/weights/best.pt'
        self.data = '/data1/lianjiawei/yolov5/data/DOTA1_0.yaml'
        self.device = select_device('')
        self.model = DetectMultiBackend(self.weights_yolov5,
                                        device=self.device,
                                        dnn=False,
                                        data=self.data).eval()
        # self.prob_extractor = MaxProbExtractor_yolov5(0, 15, self.loss_target)
        self.prob_extractor = MeanProbExtractor_yolov5(0, 15, self.loss_target, self.conf_thres, self.iou_thres,
                                                       self.max_det)


class yolov5x(BaseConfig):
    """
    Reproduce the results from the paper: Generate a patch that minimises object score.
    """

    def __init__(self):
        super().__init__()

        self.patch_name = 'ObjectOnlyPaper'
        # 2080
        # self.weights_yolov5 = '/home/mnt/ljw305/yolov5/runs/train/yolov5x/weights/best.pt'
        # self.data = '/home/mnt/ljw305/yolov5/data/DOTA1_0.yaml'
        # 3080
        self.weights_yolov5 = '/data1/lianjiawei/yolov5/runs/train/yolov5x/weights/best.pt'
        self.data = '/data1/lianjiawei/yolov5/data/DOTA1_0.yaml'
        self.device = select_device('')
        self.model = DetectMultiBackend(self.weights_yolov5,
                                        device=self.device,
                                        dnn=False,
                                        data=self.data).eval()
        # self.prob_extractor = MaxProbExtractor_yolov5(0, 15, self.loss_target)
        self.prob_extractor = MeanProbExtractor_yolov5(0, 15, self.loss_target, self.conf_thres, self.iou_thres,
                                                       self.max_det)


class faster_rcnn(BaseConfig):
    """
    Reproduce the results from the paper: Generate a patch that minimises object score.
    """

    def __init__(self):
        super().__init__()

        self.patch_name = 'ObjectOnlyPaper'
        self.config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
        # 2080
        # self.checkpoint_file = '/home/mnt/ljw305/mmdetection-master/work_dirs/faster-rcnn/epoch_12.pth'
        # 3080
        # self.checkpoint_file = '/home/ljw/mmdetection-master/work_dirs/faster-rcnn/epoch_12.pth'
        # 3090_2
        self.checkpoint_file = "/data1/lianjiawei/mmdetection-master/work_dirs/faster-rcnn/epoch_12.pth"
        self.device = 'cuda:0'
        self.model = init_detector(self.config_file, self.checkpoint_file, device=self.device)
        self.InferenceDetector = InferenceDetector()


class ssd(BaseConfig):
    """
    Reproduce the results from the paper: Generate a patch that minimises object score.
    """

    def __init__(self):
        super().__init__()

        self.patch_name = 'ObjectOnlyPaper'
        self.config_file = 'configs/ssd/ssdlite_mobilenetv2_scratch_600e_coco.py'
        # 2080
        # self.checkpoint_file = '/home/mnt/ljw305/mmdetection-master/work_dirs/ssd/epoch_120.pth'
        # 3080
        # self.checkpoint_file = '/home/ljw/mmdetection-master/work_dirs/ssd/epoch_120.pth'
        # 3090
        # self.checkpoint_file = "/data/lianjiawei/mmdetection-master/work_dirs/ssd/epoch_120.pth"
        # 3090_2
        self.checkpoint_file = "/data1/lianjiawei/mmdetection-master/work_dirs/ssd/epoch_120.pth"
        self.device = 'cuda:0'
        self.model = init_detector(self.config_file, self.checkpoint_file, device=self.device)
        self.InferenceDetector = InferenceDetector()


class swin(BaseConfig):
    """
    Reproduce the results from the paper: Generate a patch that minimises object score.
    """

    def __init__(self):
        super().__init__()

        self.patch_name = 'ObjectOnlyPaper'
        self.config_file = 'configs/swin/mask_rcnn_swin-t-p4-w7_fpn_1x_coco.py'
        # 2080
        # self.checkpoint_file = '/home/mnt/ljw305/mmdetection-master/work_dirs/swin/epoch_24.pth'
        # 3080
        # self.checkpoint_file = '/home/ljw/mmdetection-master/work_dirs/swin/epoch_24.pth'
        # 3090_2
        self.checkpoint_file = "/data1/lianjiawei/mmdetection-master/work_dirs/swin/epoch_24.pth"
        self.device = 'cuda:0'
        self.model = init_detector(self.config_file, self.checkpoint_file, device=self.device)
        self.InferenceDetector = InferenceDetector()


class cascade_rcnn(BaseConfig):
    """
    Reproduce the results from the paper: Generate a patch that minimises object score.
    """

    def __init__(self):
        super().__init__()

        self.patch_name = 'ObjectOnlyPaper'
        self.config_file = 'configs/cascade_rcnn/cascade_rcnn_r50_fpn_20e_coco.py'
        # 2080
        # self.checkpoint_file = '/home/mnt/ljw305/mmdetection-master/work_dirs/swin/epoch_24.pth'
        # 3080
        # self.checkpoint_file = '/home/ljw/mmdetection-master/work_dirs/swin/epoch_24.pth'
        # 3090
        self.checkpoint_file = "/data1/lianjiawei/mmdetection-master/work_dirs/cascade_rcnn/epoch_20.pth"
        self.device = 'cuda:0'
        self.model = init_detector(self.config_file, self.checkpoint_file, device=self.device)
        self.InferenceDetector = InferenceDetector()


class retinanet(BaseConfig):
    """
    Reproduce the results from the paper: Generate a patch that minimises object score.
    """

    def __init__(self):
        super().__init__()

        self.patch_name = 'ObjectOnlyPaper'
        self.config_file = 'configs/retinanet/retinanet_r50_fpn_2x_coco.py'
        # 2080
        # self.checkpoint_file = '/home/mnt/ljw305/mmdetection-master/work_dirs/swin/epoch_24.pth'
        # 3080
        # self.checkpoint_file = '/home/ljw/mmdetection-master/work_dirs/swin/epoch_24.pth'
        # 3090
        self.checkpoint_file = "/data/lianjiawei/mmdetection-master/work_dirs/retinanet/epoch_24.pth"
        self.device = 'cuda:0'
        self.model = init_detector(self.config_file, self.checkpoint_file, device=self.device)
        self.InferenceDetector = InferenceDetector()


class mask_rcnn(BaseConfig):
    """
    Reproduce the results from the paper: Generate a patch that minimises object score.
    """

    def __init__(self):
        super().__init__()

        self.patch_name = 'ObjectOnlyPaper'
        self.config_file = 'configs/mask_rcnn/mask_rcnn_r50_fpn_fp16_1x_coco.py'
        # 2080
        # self.checkpoint_file = '/home/mnt/ljw305/mmdetection-master/work_dirs/swin/epoch_24.pth'
        # 3080
        # self.checkpoint_file = '/home/ljw/mmdetection-master/work_dirs/swin/epoch_24.pth'
        # 3090_2
        self.checkpoint_file = "/data/lianjiawei/mmdetection-master/work_dirs/mask_rcnn/epoch_12.pth"
        self.device = 'cuda:0'
        self.model = init_detector(self.config_file, self.checkpoint_file, device=self.device)
        self.InferenceDetector = InferenceDetector()


class foveabox(BaseConfig):
    """
    Reproduce the results from the paper: Generate a patch that minimises object score.
    """

    def __init__(self):
        super().__init__()

        self.patch_name = 'ObjectOnlyPaper'
        self.config_file = 'configs/foveabox/fovea_r50_fpn_4x4_2x_coco.py'
        # 2080
        # self.checkpoint_file = '/home/mnt/ljw305/mmdetection-master/work_dirs/swin/epoch_24.pth'
        # 3080
        # self.checkpoint_file = '/home/ljw/mmdetection-master/work_dirs/swin/epoch_24.pth'
        # 3090_2
        self.checkpoint_file = "/data/lianjiawei/mmdetection-master/work_dirs/fovea_r50_fpn_4x4_2x_coco/epoch_24.pth"
        self.device = 'cuda:0'
        self.model = init_detector(self.config_file, self.checkpoint_file, device=self.device)
        self.InferenceDetector = InferenceDetector()


class free_anchor(BaseConfig):
    """
    Reproduce the results from the paper: Generate a patch that minimises object score.
    """

    def __init__(self):
        super().__init__()

        self.patch_name = 'ObjectOnlyPaper'
        self.config_file = 'configs/free_anchor/retinanet_free_anchor_r50_fpn_1x_coco.py'
        # 2080
        # self.checkpoint_file = '/home/mnt/ljw305/mmdetection-master/work_dirs/swin/epoch_24.pth'
        # 3080
        # self.checkpoint_file = '/home/ljw/mmdetection-master/work_dirs/swin/epoch_24.pth'
        # 3090_2
        self.checkpoint_file = "/data1/lianjiawei/mmdetection-master/work_dirs/retinanet_free_anchor_r50_fpn_1x_coco/epoch_12.pth"
        self.device = 'cuda:0'
        self.model = init_detector(self.config_file, self.checkpoint_file, device=self.device)
        self.InferenceDetector = InferenceDetector()


class fsaf(BaseConfig):
    """
    Reproduce the results from the paper: Generate a patch that minimises object score.
    """

    def __init__(self):
        super().__init__()

        self.patch_name = 'ObjectOnlyPaper'
        self.config_file = 'configs/fsaf/fsaf_r50_fpn_1x_coco.py'
        # 2080
        # self.checkpoint_file = '/home/mnt/ljw305/mmdetection-master/work_dirs/swin/epoch_24.pth'
        # 3080
        # self.checkpoint_file = '/home/ljw/mmdetection-master/work_dirs/swin/epoch_24.pth'
        # 3090_2
        self.checkpoint_file = "/data1/lianjiawei/mmdetection-master/work_dirs/fsaf_r50_fpn_1x_coco/epoch_12.pth"
        self.device = 'cuda:0'
        self.model = init_detector(self.config_file, self.checkpoint_file, device=self.device)
        self.InferenceDetector = InferenceDetector()


class reppoints(BaseConfig):
    """
    Reproduce the results from the paper: Generate a patch that minimises object score.
    """

    def __init__(self):
        super().__init__()

        self.patch_name = 'ObjectOnlyPaper'
        self.config_file = 'configs/reppoints/reppoints_moment_r50_fpn_1x_coco.py'
        # 2080
        # self.checkpoint_file = '/home/mnt/ljw305/mmdetection-master/work_dirs/swin/epoch_24.pth'
        # 3080
        # self.checkpoint_file = '/home/ljw/mmdetection-master/work_dirs/swin/epoch_24.pth'
        # 3090_2
        self.checkpoint_file = "/data1/lianjiawei/mmdetection-master/work_dirs/reppoints_moment_r50_fpn_1x_coco/epoch_12.pth"
        self.device = 'cuda:0'
        self.model = init_detector(self.config_file, self.checkpoint_file, device=self.device)
        self.InferenceDetector = InferenceDetector()


class tood(BaseConfig):
    """
    Reproduce the results from the paper: Generate a patch that minimises object score.
    """

    def __init__(self):
        super().__init__()

        self.patch_name = 'ObjectOnlyPaper'
        self.config_file = 'configs/tood/tood_r50_fpn_1x_coco.py'
        # 2080
        # self.checkpoint_file = '/home/mnt/ljw305/mmdetection-master/work_dirs/swin/epoch_24.pth'
        # 3080
        # self.checkpoint_file = '/home/ljw/mmdetection-master/work_dirs/swin/epoch_24.pth'
        # 3090_2
        self.checkpoint_file = "/data1/lianjiawei/mmdetection-master/work_dirs/tood_r50_fpn_1x_coco/epoch_12.pth"
        self.device = 'cuda:0'
        self.model = init_detector(self.config_file, self.checkpoint_file, device=self.device)
        self.InferenceDetector = InferenceDetector()


class atss(BaseConfig):
    """
    Reproduce the results from the paper: Generate a patch that minimises object score.
    """

    def __init__(self):
        super().__init__()

        self.patch_name = 'ObjectOnlyPaper'
        self.config_file = 'configs/atss/atss_r50_fpn_1x_coco.py'
        # 2080
        # self.checkpoint_file = '/home/mnt/ljw305/mmdetection-master/work_dirs/swin/epoch_24.pth'
        # 3080
        # self.checkpoint_file = '/home/ljw/mmdetection-master/work_dirs/swin/epoch_24.pth'
        # 3090
        self.checkpoint_file = "/data1/lianjiawei/mmdetection-master/work_dirs/atss_r50_fpn_1x_coco/epoch_12.pth"
        self.device = 'cuda:0'
        # 3090_2
        # self.checkpoint_file = "/data1/lianjiawei/mmdetection-master/work_dirs/atss_r50_fpn_1x_coco/epoch_12.pth"
        # self.device = 'cuda:0'
        self.model = init_detector(self.config_file, self.checkpoint_file, device=self.device)
        self.InferenceDetector = InferenceDetector()


class vfnet(BaseConfig):
    """
    Reproduce the results from the paper: Generate a patch that minimises object score.
    """

    def __init__(self):
        super().__init__()

        self.patch_name = 'ObjectOnlyPaper'
        self.config_file = 'configs/vfnet/vfnet_r50_fpn_1x_coco.py'
        # 2080
        # self.checkpoint_file = '/home/mnt/ljw305/mmdetection-master/work_dirs/swin/epoch_24.pth'
        # 3080
        # self.checkpoint_file = '/home/ljw/mmdetection-master/work_dirs/swin/epoch_24.pth'
        # 3090_2
        self.checkpoint_file = "/data1/lianjiawei/mmdetection-master/work_dirs/vfnet_r50_fpn_1x_coco/epoch_12.pth"
        self.device = 'cuda:0'
        self.model = init_detector(self.config_file, self.checkpoint_file, device=self.device)
        self.InferenceDetector = InferenceDetector()


patch_configs = {
    "base": BaseConfig,
    "yolov2": yolov2,
    "yolov3": yolov3,
    "yolov5n": yolov5n,
    "yolov5s": yolov5s,
    "yolov5m": yolov5m,
    "yolov5l": yolov5l,
    "yolov5x": yolov5x,
    "faster-rcnn": faster_rcnn,
    "swin": swin,
    "ssd": ssd,
    "cascade_rcnn": cascade_rcnn,
    "retinanet": retinanet,
    "mask_rcnn": mask_rcnn,
    "foveabox": foveabox,  # anchor-free
    "free_anchor": free_anchor,
    "fsaf": fsaf,  # anchor-free, single-shot
    "reppoints": reppoints,
    "tood": tood,  # one-stage
    "atss": atss,
    "vfnet": vfnet,
}
