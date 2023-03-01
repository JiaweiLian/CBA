import brambox as bb
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

# # Expand to show
# pd.set_option("display.max_rows", 1000)
# pd.set_option("display.max_columns", 1000)

# Settings
bb.logger.setConsoleLevel('ERROR')  # Only show error log messages
iou_threshold = 0.5

# Load annotations
anno2 = bb.io.load('anno_coco',
                   './testing/plane/clean/labels-yolo/annotations/aircraft_yolov2.json')

anno3 = bb.io.load('anno_coco',
                   './testing/plane/clean/labels-yolo/annotations/aircraft_yolov3.json')

anno5n = bb.io.load('anno_coco',
                    './testing/plane/clean/labels-yolo/annotations/aircraft_yolov5n.json')

anno5s = bb.io.load('anno_coco',
                    './testing/plane/clean/labels-yolo/annotations/aircraft_yolov5s.json')

anno5m = bb.io.load('anno_coco',
                    './testing/plane/clean/labels-yolo/annotations/aircraft_yolov5m.json')

anno5l = bb.io.load('anno_coco',
                    './testing/plane/clean/labels-yolo/annotations/aircraft_yolov5l.json')

anno5x = bb.io.load('anno_coco',
                    './testing/plane/clean/labels-yolo/annotations/aircraft_yolov5x.json')

annoF = bb.io.load('anno_coco',
                   './testing/plane/clean/labels-yolo/annotations/aircraft_faster-rcnn.json')

annoSSD = bb.io.load('anno_coco',
                     './testing/plane/clean/labels-yolo/annotations/aircraft_ssd.json')

annoSwin = bb.io.load('anno_coco',
                      './testing/plane/clean/labels-yolo/annotations/aircraft_swin.json')

det_clean = bb.io.load('det_coco',
                       './testing/yolov5s_center_150_1024_yolov5s/clean_results.json',
                       class_label_map=[lines.strip() for lines in open('./data/dota.names')])
pr_clean = bb.stat.pr(det_clean, anno5s, iou_threshold)
ap_clean = bb.stat.ap(pr_clean)
# Draw PR-curve
ax = pr_clean.plot('recall', 'precision', drawstyle='steps',
                   label=f'YOLO-Clean-AP = {round(100 * ap_clean, 2)}%')

# # Random noise results
# ########################################################################################################################
# det_noise = bb.io.load('det_coco',
#                        '/data1/lianjiawei/adversarial-yolo/testing/yolov2_center_150_1024_yolov2/noise_results.json',
#                        class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_noise = bb.stat.pr(det_noise, anno2, iou_threshold)
# ap_noise = bb.stat.ap(pr_noise)
# pr_noise.plot('recall', 'precision', drawstyle='steps',
#               label=f'YOLOv2-Noise-AP = {round(100 * ap_noise, 2)}%', ax=ax)
#
# det_noise = bb.io.load('det_coco',
#                        '/data1/lianjiawei/adversarial-yolo/testing/yolov3_center_150_1024_yolov3/noise_results.json',
#                        class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_noise = bb.stat.pr(det_noise, anno3, iou_threshold)
# ap_noise = bb.stat.ap(pr_noise)
# pr_noise.plot('recall', 'precision', drawstyle='steps',
#               label=f'YOLOv3-Noise-AP = {round(100 * ap_noise, 2)}%', ax=ax)
#
# det_noise = bb.io.load('det_coco',
#                        '/data1/lianjiawei/adversarial-yolo/testing/yolov5n_center_150_1024_yolov5n/noise_results.json',
#                        class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_noise = bb.stat.pr(det_noise, anno5n, iou_threshold)
# ap_noise = bb.stat.ap(pr_noise)
# pr_noise.plot('recall', 'precision', drawstyle='steps',
#               label=f'YOLOv5n-Noise-AP = {round(100 * ap_noise, 2)}%', ax=ax)
#
# det_noise = bb.io.load('det_coco',
#                        '/data1/lianjiawei/adversarial-yolo/testing/yolov5s_center_150_1024_yolov5s/noise_results.json',
#                        class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_noise = bb.stat.pr(det_noise, anno5s, iou_threshold)
# ap_noise = bb.stat.ap(pr_noise)
# pr_noise.plot('recall', 'precision', drawstyle='steps',
#               label=f'YOLOv5s-Noise-AP = {round(100 * ap_noise, 2)}%', ax=ax)
#
# det_noise = bb.io.load('det_coco',
#                        '/data1/lianjiawei/adversarial-yolo/testing/yolov2_center_150_1024_yolov5m/noise_results.json',
#                        class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_noise = bb.stat.pr(det_noise, anno5m, iou_threshold)
# ap_noise = bb.stat.ap(pr_noise)
# pr_noise.plot('recall', 'precision', drawstyle='steps',
#               label=f'YOLOv5m-Noise-AP = {round(100 * ap_noise, 2)}%', ax=ax)
#
# det_noise = bb.io.load('det_coco',
#                        '/data1/lianjiawei/adversarial-yolo/testing/yolov2_center_150_1024_yolov5l/noise_results.json',
#                        class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_noise = bb.stat.pr(det_noise, anno5l, iou_threshold)
# ap_noise = bb.stat.ap(pr_noise)
# pr_noise.plot('recall', 'precision', drawstyle='steps',
#               label=f'YOLOv5l-Noise-AP = {round(100 * ap_noise, 2)}%', ax=ax)
#
# det_noise = bb.io.load('det_coco',
#                        '/data1/lianjiawei/adversarial-yolo/testing/yolov2_center_150_1024_yolov5x/noise_results.json',
#                        class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_noise = bb.stat.pr(det_noise, anno5x, iou_threshold)
# ap_noise = bb.stat.ap(pr_noise)
# pr_noise.plot('recall', 'precision', drawstyle='steps',
#               label=f'YOLOv5x-Noise-AP = {round(100 * ap_noise, 2)}%', ax=ax)
#
# det_noise = bb.io.load('det_coco',
#                        '/data1/lianjiawei/adversarial-yolo/testing/faster-rcnn_center_150_1024_faster-rcnn/noise_results.json',
#                        class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_noise = bb.stat.pr(det_noise, annoF, iou_threshold)
# ap_noise = bb.stat.ap(pr_noise)
# pr_noise.plot('recall', 'precision', drawstyle='steps',
#               label=f'faster-rcnn-Noise-AP = {round(100 * ap_noise, 2)}%', ax=ax)
#
# det_noise = bb.io.load('det_coco',
#                        '/data1/lianjiawei/adversarial-yolo/testing/ssd_center_150_1024_ssd/noise_results.json',
#                        class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_noise = bb.stat.pr(det_noise, annoSSD, iou_threshold)
# ap_noise = bb.stat.ap(pr_noise)
# pr_noise.plot('recall', 'precision', drawstyle='steps',
#               label=f'SSD-Noise-AP = {round(100 * ap_noise, 2)}%', ax=ax)
#
# det_noise = bb.io.load('det_coco',
#                        '/data1/lianjiawei/adversarial-yolo/testing/swin_center_150_1024_swin/noise_results.json',
#                        class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_noise = bb.stat.pr(det_noise, annoSwin, iou_threshold)
# ap_noise = bb.stat.ap(pr_noise)
# pr_noise.plot('recall', 'precision', drawstyle='steps',
#               label=f'Swin-Noise-AP = {round(100 * ap_noise, 2)}%', ax=ax)
########################################################################################################################

# # Patch2
# ######################################################################################################################
# det_patch2 = bb.io.load('det_coco',
#                         '/data1/lianjiawei/adversarial-yolo/testing/yolov2_A3_yolov2/patch_results.json',
#                         class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_patch2 = bb.stat.pr(det_patch2, anno2, iou_threshold)
# ap_patch2 = bb.stat.ap(pr_patch2)
# pr_patch2.plot('recall', 'precision', drawstyle='steps',
#                label=f'YOLOv2-Patch2-AP = {round(100 * ap_patch2, 2)}%', ax=ax)
#
# det_patch2 = bb.io.load('det_coco',
#                         '/data1/lianjiawei/adversarial-yolo/testing/yolov2_A3_yolov3/patch_results.json',
#                         class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_patch2 = bb.stat.pr(det_patch2, anno3, iou_threshold)
# ap_patch2 = bb.stat.ap(pr_patch2)
# pr_patch2.plot('recall', 'precision', drawstyle='steps',
#                label=f'YOLOv3-Patch2-AP = {round(100 * ap_patch2, 2)}%', ax=ax)
#
# det_patch2 = bb.io.load('det_coco',
#                         '/data1/lianjiawei/adversarial-yolo/testing/yolov2_A3_yolov5n/patch_results.json',
#                         class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_patch2 = bb.stat.pr(det_patch2, anno5n, iou_threshold)
# ap_patch2 = bb.stat.ap(pr_patch2)
# pr_patch2.plot('recall', 'precision', drawstyle='steps',
#                label=f'YOLOv5n-Patch2-AP = {round(100 * ap_patch2, 2)}%', ax=ax)
#
# det_patch2 = bb.io.load('det_coco',
#                         '/data1/lianjiawei/adversarial-yolo/testing/yolov2_A3_yolov5s/patch_results.json',
#                         class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_patch2 = bb.stat.pr(det_patch2, anno5s, iou_threshold)
# ap_patch2 = bb.stat.ap(pr_patch2)
# pr_patch2.plot('recall', 'precision', drawstyle='steps',
#                label=f'YOLOv5s-Patch2-AP = {round(100 * ap_patch2, 2)}%', ax=ax)
#
# det_patch3 = bb.io.load('det_coco',
#                         '/data1/lianjiawei/adversarial-yolo/testing/yolov2_A3_yolov5m/patch_results.json',
#                         class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_patch3 = bb.stat.pr(det_patch3, anno5m, iou_threshold)
# ap_patch3 = bb.stat.ap(pr_patch3)
# pr_patch3.plot('recall', 'precision', drawstyle='steps',
#                label=f'YOLOv5m-Patch2-AP = {round(100 * ap_patch3, 2)}%', ax=ax)
#
# det_patch4 = bb.io.load('det_coco',
#                         '/data1/lianjiawei/adversarial-yolo/testing/yolov2_A3_yolov5l/patch_results.json',
#                         class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_patch4 = bb.stat.pr(det_patch4, anno5l, iou_threshold)
# ap_patch4 = bb.stat.ap(pr_patch4)
# pr_patch4.plot('recall', 'precision', drawstyle='steps',
#                label=f'YOLOv5l-Patch2-AP = {round(100 * ap_patch4, 2)}%', ax=ax)
#
# det_patch4 = bb.io.load('det_coco',
#                         '/data1/lianjiawei/adversarial-yolo/testing/yolov2_A3_yolov5x/patch_results.json',
#                         class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_patch4 = bb.stat.pr(det_patch4, anno5x, iou_threshold)
# ap_patch4 = bb.stat.ap(pr_patch4)
# pr_patch4.plot('recall', 'precision', drawstyle='steps',
#                label=f'YOLOv5x-Patch2-AP = {round(100 * ap_patch4, 2)}%', ax=ax)
#
# det_patch4 = bb.io.load('det_coco',
#                         '/data1/lianjiawei/adversarial-yolo/testing/yolov2_A3_faster-rcnn/patch_results.json',
#                         class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_patch4 = bb.stat.pr(det_patch4, annoF, iou_threshold)
# ap_patch4 = bb.stat.ap(pr_patch4)
# pr_patch4.plot('recall', 'precision', drawstyle='steps',
#                label=f'faster-rcnn-Patch2-AP = {round(100 * ap_patch4, 2)}%', ax=ax)
#
# det_patch4 = bb.io.load('det_coco',
#                         '/data1/lianjiawei/adversarial-yolo/testing/yolov2_A3_ssd/patch_results.json',
#                         class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_patch4 = bb.stat.pr(det_patch4, annoSSD, iou_threshold)
# ap_patch4 = bb.stat.ap(pr_patch4)
# pr_patch4.plot('recall', 'precision', drawstyle='steps',
#                label=f'SSD-Patch2-AP = {round(100 * ap_patch4, 2)}%', ax=ax)
#
# det_patch4 = bb.io.load('det_coco',
#                         '/data1/lianjiawei/adversarial-yolo/testing/yolov2_A3_swin/patch_results.json',
#                         class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_patch4 = bb.stat.pr(det_patch4, annoSwin, iou_threshold)
# ap_patch4 = bb.stat.ap(pr_patch4)
# pr_patch4.plot('recall', 'precision', drawstyle='steps',
#                label=f'Swin-Patch2-AP = {round(100 * ap_patch4, 2)}%', ax=ax)
########################################################################################################################

# # Patch3
# ########################################################################################################################
# det_patch2 = bb.io.load('det_coco',
#                         '/data1/lianjiawei/adversarial-yolo/testing/yolov3_A3_yolov2/patch_results.json',
#                         class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_patch2 = bb.stat.pr(det_patch2, anno2, iou_threshold)
# ap_patch2 = bb.stat.ap(pr_patch2)
# pr_patch2.plot('recall', 'precision', drawstyle='steps',
#                label=f'YOLOv2-Patch3-AP = {round(100 * ap_patch2, 2)}%', ax=ax)
#
# det_patch2 = bb.io.load('det_coco',
#                         '/data1/lianjiawei/adversarial-yolo/testing/yolov3_A3_yolov3/patch_results.json',
#                         class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_patch2 = bb.stat.pr(det_patch2, anno3, iou_threshold)
# ap_patch2 = bb.stat.ap(pr_patch2)
# pr_patch2.plot('recall', 'precision', drawstyle='steps',
#                label=f'YOLOv3-Patch3-AP = {round(100 * ap_patch2, 2)}%', ax=ax)
#
# det_patch2 = bb.io.load('det_coco',
#                         '/data1/lianjiawei/adversarial-yolo/testing/yolov3_A3_yolov5n/patch_results.json',
#                         class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_patch2 = bb.stat.pr(det_patch2, anno5n, iou_threshold)
# ap_patch2 = bb.stat.ap(pr_patch2)
# pr_patch2.plot('recall', 'precision', drawstyle='steps',
#                label=f'YOLOv5n-Patch3-AP = {round(100 * ap_patch2, 2)}%', ax=ax)
#
# det_patch2 = bb.io.load('det_coco',
#                         '/data1/lianjiawei/adversarial-yolo/testing/yolov3_A3_yolov5s/patch_results.json',
#                         class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_patch2 = bb.stat.pr(det_patch2, anno5s, iou_threshold)
# ap_patch2 = bb.stat.ap(pr_patch2)
# pr_patch2.plot('recall', 'precision', drawstyle='steps',
#                label=f'YOLOv5s-Patch3-AP = {round(100 * ap_patch2, 2)}%', ax=ax)
#
# det_patch2 = bb.io.load('det_coco',
#                         '/data1/lianjiawei/adversarial-yolo/testing/yolov3_A3_yolov5m/patch_results.json',
#                         class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_patch2 = bb.stat.pr(det_patch2, anno5m, iou_threshold)
# ap_patch2 = bb.stat.ap(pr_patch2)
# pr_patch2.plot('recall', 'precision', drawstyle='steps',
#                label=f'YOLOv5m-Patch3-AP = {round(100 * ap_patch2, 2)}%', ax=ax)
#
# det_patch2 = bb.io.load('det_coco',
#                         '/data1/lianjiawei/adversarial-yolo/testing/yolov3_A3_yolov5l/patch_results.json',
#                         class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_patch2 = bb.stat.pr(det_patch2, anno5l, iou_threshold)
# ap_patch2 = bb.stat.ap(pr_patch2)
# pr_patch2.plot('recall', 'precision', drawstyle='steps',
#                label=f'YOLOv5l-Patch3-AP = {round(100 * ap_patch2, 2)}%', ax=ax)
#
# det_patch2 = bb.io.load('det_coco',
#                         '/data1/lianjiawei/adversarial-yolo/testing/yolov3_A3_yolov5x/patch_results.json',
#                         class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_patch2 = bb.stat.pr(det_patch2, anno5x, iou_threshold)
# ap_patch2 = bb.stat.ap(pr_patch2)
# pr_patch2.plot('recall', 'precision', drawstyle='steps',
#                label=f'YOLOv5x-Patch3-AP = {round(100 * ap_patch2, 2)}%', ax=ax)
#
# det_patch4 = bb.io.load('det_coco',
#                         '/data1/lianjiawei/adversarial-yolo/testing/yolov3_A3_faster-rcnn/patch_results.json',
#                         class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_patch4 = bb.stat.pr(det_patch4, annoF, iou_threshold)
# ap_patch4 = bb.stat.ap(pr_patch4)
# pr_patch4.plot('recall', 'precision', drawstyle='steps',
#                label=f'faster-rcnn-Patch3-AP = {round(100 * ap_patch4, 2)}%', ax=ax)
#
# det_patch4 = bb.io.load('det_coco',
#                         '/data1/lianjiawei/adversarial-yolo/testing/yolov3_A3_ssd/patch_results.json',
#                         class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_patch4 = bb.stat.pr(det_patch4, annoSSD, iou_threshold)
# ap_patch4 = bb.stat.ap(pr_patch4)
# pr_patch4.plot('recall', 'precision', drawstyle='steps',
#                label=f'SSD-Patch3-AP = {round(100 * ap_patch4, 2)}%', ax=ax)
#
# det_patch4 = bb.io.load('det_coco',
#                         '/data1/lianjiawei/adversarial-yolo/testing/yolov3_A3_swin/patch_results.json',
#                         class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_patch4 = bb.stat.pr(det_patch4, annoSwin, iou_threshold)
# ap_patch4 = bb.stat.ap(pr_patch4)
# pr_patch4.plot('recall', 'precision', drawstyle='steps',
#                label=f'Swin-Patch3-AP = {round(100 * ap_patch4, 2)}%', ax=ax)
#############################################################################################################################

# # Patch5n
# ########################################################################################################################
# det_patch11 = bb.io.load('det_coco',
#                          '/data1/lianjiawei/adversarial-yolo/testing/yolov5n_A3_yolov2/patch_results.json',
#                          class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_patch11 = bb.stat.pr(det_patch11, anno2, iou_threshold)
# ap_patch11 = bb.stat.ap(pr_patch11)
# pr_patch11.plot('recall', 'precision', drawstyle='steps',
#                 label=f'YOLOv2-Patch5n-AP = {round(100 * ap_patch11, 2)}%', ax=ax)
#
# det_patch11 = bb.io.load('det_coco',
#                          '/data1/lianjiawei/adversarial-yolo/testing/yolov5n_A3_yolov3/patch_results.json',
#                          class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_patch11 = bb.stat.pr(det_patch11, anno3, iou_threshold)
# ap_patch11 = bb.stat.ap(pr_patch11)
# pr_patch11.plot('recall', 'precision', drawstyle='steps',
#                 label=f'YOLOv3-Patch5n-AP = {round(100 * ap_patch11, 2)}%', ax=ax)
#
# det_patch9 = bb.io.load('det_coco',
#                         '/data1/lianjiawei/adversarial-yolo/testing/yolov5n_A3_yolov5n/patch_results.json',
#                         class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_patch9 = bb.stat.pr(det_patch9, anno5n, iou_threshold)
# ap_patch9 = bb.stat.ap(pr_patch9)
# pr_patch9.plot('recall', 'precision', drawstyle='steps',
#                label=f'YOLOv5n-Patch5n-AP = {round(100 * ap_patch9, 2)}%', ax=ax)
#
# det_patch10 = bb.io.load('det_coco',
#                          '/data1/lianjiawei/adversarial-yolo/testing/yolov5n_A3_yolov5s/patch_results.json',
#                          class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_patch10 = bb.stat.pr(det_patch10, anno5s, iou_threshold)
# ap_patch10 = bb.stat.ap(pr_patch10)
# pr_patch10.plot('recall', 'precision', drawstyle='steps',
#                 label=f'YOLOv5s-Patch5n-AP = {round(100 * ap_patch10, 2)}%', ax=ax)
#
# det_patch12 = bb.io.load('det_coco',
#                          '/data1/lianjiawei/adversarial-yolo/testing/yolov5n_A3_yolov5m/patch_results.json',
#                          class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_patch12 = bb.stat.pr(det_patch12, anno5m, iou_threshold)
# ap_patch12 = bb.stat.ap(pr_patch12)
# pr_patch12.plot('recall', 'precision', drawstyle='steps',
#                 label=f'YOLOv5m-Patch5n-AP = {round(100 * ap_patch12, 2)}%', ax=ax)
#
# det_patch13 = bb.io.load('det_coco',
#                          '/data1/lianjiawei/adversarial-yolo/testing/yolov5n_A3_yolov5l/patch_results.json',
#                          class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_patch13 = bb.stat.pr(det_patch13, anno5l, iou_threshold)
# ap_patch13 = bb.stat.ap(pr_patch13)
# pr_patch13.plot('recall', 'precision', drawstyle='steps',
#                 label=f'YOLOv5l-Patch5n-AP = {round(100 * ap_patch13, 2)}%', ax=ax)
#
# det_patch13 = bb.io.load('det_coco',
#                          '/data1/lianjiawei/adversarial-yolo/testing/yolov5n_A3_yolov5x/patch_results.json',
#                          class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_patch13 = bb.stat.pr(det_patch13, anno5x, iou_threshold)
# ap_patch13 = bb.stat.ap(pr_patch13)
# pr_patch13.plot('recall', 'precision', drawstyle='steps',
#                 label=f'YOLOv5x-Patch5n-AP = {round(100 * ap_patch13, 2)}%', ax=ax)
#
# det_patch4 = bb.io.load('det_coco',
#                         '/data1/lianjiawei/adversarial-yolo/testing/yolov5n_A3_faster-rcnn/patch_results.json',
#                         class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_patch4 = bb.stat.pr(det_patch4, annoF, iou_threshold)
# ap_patch4 = bb.stat.ap(pr_patch4)
# pr_patch4.plot('recall', 'precision', drawstyle='steps',
#                label=f'faster-rcnn-Patch5n-AP = {round(100 * ap_patch4, 2)}%', ax=ax)
#
# det_patch4 = bb.io.load('det_coco',
#                         '/data1/lianjiawei/adversarial-yolo/testing/yolov5n_A3_ssd/patch_results.json',
#                         class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_patch4 = bb.stat.pr(det_patch4, annoSSD, iou_threshold)
# ap_patch4 = bb.stat.ap(pr_patch4)
# pr_patch4.plot('recall', 'precision', drawstyle='steps',
#                label=f'SSD-Patch5n-AP = {round(100 * ap_patch4, 2)}%', ax=ax)
#
# det_patch4 = bb.io.load('det_coco',
#                         '/data1/lianjiawei/adversarial-yolo/testing/yolov5n_A3_swin/patch_results.json',
#                         class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_patch4 = bb.stat.pr(det_patch4, annoSwin, iou_threshold)
# ap_patch4 = bb.stat.ap(pr_patch4)
# pr_patch4.plot('recall', 'precision', drawstyle='steps',
#                label=f'Swin-Patch5n-AP = {round(100 * ap_patch4, 2)}%', ax=ax)
########################################################################################################################

# # Patch5s
# ######################################################################################################################
# det_patch8 = bb.io.load('det_coco',
#                         '/data1/lianjiawei/adversarial-yolo/testing/yolov5s_A3_yolov2/patch_results.json',
#                         class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_patch8 = bb.stat.pr(det_patch8, anno2, iou_threshold)
# ap_patch8 = bb.stat.ap(pr_patch8)
# pr_patch8.plot('recall', 'precision', drawstyle='steps',
#                label=f'YOLOv2-Patch5s-AP = {round(100 * ap_patch8, 2)}%', ax=ax)
#
# det_patch8 = bb.io.load('det_coco',
#                         '/data1/lianjiawei/adversarial-yolo/testing/yolov5s_A3_yolov3/patch_results.json',
#                         class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_patch8 = bb.stat.pr(det_patch8, anno3, iou_threshold)
# ap_patch8 = bb.stat.ap(pr_patch8)
# pr_patch8.plot('recall', 'precision', drawstyle='steps',
#                label=f'YOLOv3-Patch5s-AP = {round(100 * ap_patch8, 2)}%', ax=ax)
#
# det_patch7 = bb.io.load('det_coco',
#                         '/data1/lianjiawei/adversarial-yolo/testing/yolov5s_A3_yolov5n/patch_results.json',
#                         class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_patch7 = bb.stat.pr(det_patch7, anno5n, iou_threshold)
# ap_patch7 = bb.stat.ap(pr_patch7)
# pr_patch7.plot('recall', 'precision', drawstyle='steps',
#                label=f'YOLOv5n-Patch5s-AP = {round(100 * ap_patch7, 2)}%', ax=ax)
#
# det_patch = bb.io.load('det_coco',
#                        '/data1/lianjiawei/adversarial-yolo/testing/yolov5s_A3_yolov5s/patch_results.json',
#                        class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_patch = bb.stat.pr(det_patch, anno5s, iou_threshold)
# ap_patch = bb.stat.ap(pr_patch)
# pr_patch.plot('recall', 'precision', drawstyle='steps',
#               label=f'YOLOv5s-Patch5s-AP = {round(100 * ap_patch, 2)}%', ax=ax)
#
# det_patch6 = bb.io.load('det_coco',
#                         '/data1/lianjiawei/adversarial-yolo/testing/yolov5s_A3_yolov5m/patch_results.json',
#                         class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_patch6 = bb.stat.pr(det_patch6, anno5m, iou_threshold)
# ap_patch6 = bb.stat.ap(pr_patch6)
# pr_patch6.plot('recall', 'precision', drawstyle='steps',
#                label=f'YOLOv5m-Patch5s-AP = {round(100 * ap_patch6, 2)}%', ax=ax)
#
# det_patch5 = bb.io.load('det_coco',
#                         '/data1/lianjiawei/adversarial-yolo/testing/yolov5s_A3_yolov5l/patch_results.json',
#                         class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_patch5 = bb.stat.pr(det_patch5, anno5l, iou_threshold)
# ap_patch5 = bb.stat.ap(pr_patch5)
# pr_patch5.plot('recall', 'precision', drawstyle='steps',
#                label=f'YOLOv5l-Patch5s-AP = {round(100 * ap_patch5, 2)}%', ax=ax)
#
# det_patch5 = bb.io.load('det_coco',
#                         '/data1/lianjiawei/adversarial-yolo/testing/yolov5s_A3_yolov5x/patch_results.json',
#                         class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_patch5 = bb.stat.pr(det_patch5, anno5x, iou_threshold)
# ap_patch5 = bb.stat.ap(pr_patch5)
# pr_patch5.plot('recall', 'precision', drawstyle='steps',
#                label=f'YOLOv5x-Patch5s-AP = {round(100 * ap_patch5, 2)}%', ax=ax)
#
# det_patch4 = bb.io.load('det_coco',
#                         '/data1/lianjiawei/adversarial-yolo/testing/yolov5s_A3_faster-rcnn/patch_results.json',
#                         class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_patch4 = bb.stat.pr(det_patch4, annoF, iou_threshold)
# ap_patch4 = bb.stat.ap(pr_patch4)
# pr_patch4.plot('recall', 'precision', drawstyle='steps',
#                label=f'faster-rcnn-Patch5s-AP = {round(100 * ap_patch4, 2)}%', ax=ax)
#
# det_patch4 = bb.io.load('det_coco',
#                         '/data1/lianjiawei/adversarial-yolo/testing/yolov5s_A3_ssd/patch_results.json',
#                         class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_patch4 = bb.stat.pr(det_patch4, annoSSD, iou_threshold)
# ap_patch4 = bb.stat.ap(pr_patch4)
# pr_patch4.plot('recall', 'precision', drawstyle='steps',
#                label=f'SSD-Patch5s-AP = {round(100 * ap_patch4, 2)}%', ax=ax)
#
# det_patch4 = bb.io.load('det_coco',
#                         '/data1/lianjiawei/adversarial-yolo/testing/yolov5s_A3_swin/patch_results.json',
#                         class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_patch4 = bb.stat.pr(det_patch4, annoSwin, iou_threshold)
# ap_patch4 = bb.stat.ap(pr_patch4)
# pr_patch4.plot('recall', 'precision', drawstyle='steps',
#                label=f'Swin-Patch5s-AP = {round(100 * ap_patch4, 2)}%', ax=ax)
########################################################################################################################

# Patch5m
# ########################################################################################################################
# det_patch8 = bb.io.load('det_coco',
#                         '/data1/lianjiawei/adversarial-yolo/testing/yolov5m_A3_yolov2/patch_results.json',
#                         class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_patch8 = bb.stat.pr(det_patch8, anno2, iou_threshold)
# ap_patch8 = bb.stat.ap(pr_patch8)
# pr_patch8.plot('recall', 'precision', drawstyle='steps',
#                label=f'YOLOv2-Patch5m-AP = {round(100 * ap_patch8, 2)}%', ax=ax)
#
# det_patch8 = bb.io.load('det_coco',
#                         '/data1/lianjiawei/adversarial-yolo/testing/yolov5m_A3_yolov3/patch_results.json',
#                         class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_patch8 = bb.stat.pr(det_patch8, anno3, iou_threshold)
# ap_patch8 = bb.stat.ap(pr_patch8)
# pr_patch8.plot('recall', 'precision', drawstyle='steps',
#                label=f'YOLOv3-Patch5m-AP = {round(100 * ap_patch8, 2)}%', ax=ax)
#
# det_patch7 = bb.io.load('det_coco',
#                         '/data1/lianjiawei/adversarial-yolo/testing/yolov5m_A3_yolov5n/patch_results.json',
#                         class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_patch7 = bb.stat.pr(det_patch7, anno5n, iou_threshold)
# ap_patch7 = bb.stat.ap(pr_patch7)
# pr_patch7.plot('recall', 'precision', drawstyle='steps',
#                label=f'YOLOv5n-Patch5m-AP = {round(100 * ap_patch7, 2)}%', ax=ax)
#
# det_patch = bb.io.load('det_coco',
#                        '/data1/lianjiawei/adversarial-yolo/testing/yolov5m_A3_yolov5s/patch_results.json',
#                        class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_patch = bb.stat.pr(det_patch, anno5s, iou_threshold)
# ap_patch = bb.stat.ap(pr_patch)
# pr_patch.plot('recall', 'precision', drawstyle='steps',
#               label=f'YOLOv5s-Patch5m-AP = {round(100 * ap_patch, 2)}%', ax=ax)
#
# det_patch6 = bb.io.load('det_coco',
#                         '/data1/lianjiawei/adversarial-yolo/testing/yolov5m_A3_yolov5m/patch_results.json',
#                         class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_patch6 = bb.stat.pr(det_patch6, anno5m, iou_threshold)
# ap_patch6 = bb.stat.ap(pr_patch6)
# pr_patch6.plot('recall', 'precision', drawstyle='steps',
#                label=f'YOLOv5m-Patch5m-AP = {round(100 * ap_patch6, 2)}%', ax=ax)
#
# det_patch5 = bb.io.load('det_coco',
#                         '/data1/lianjiawei/adversarial-yolo/testing/yolov5m_A3_yolov5l/patch_results.json',
#                         class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_patch5 = bb.stat.pr(det_patch5, anno5l, iou_threshold)
# ap_patch5 = bb.stat.ap(pr_patch5)
# pr_patch5.plot('recall', 'precision', drawstyle='steps',
#                label=f'YOLOv5l-Patch5m-AP = {round(100 * ap_patch5, 2)}%', ax=ax)
#
# det_patch5 = bb.io.load('det_coco',
#                         '/data1/lianjiawei/adversarial-yolo/testing/yolov5m_A3_yolov5x/patch_results.json',
#                         class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_patch5 = bb.stat.pr(det_patch5, anno5x, iou_threshold)
# ap_patch5 = bb.stat.ap(pr_patch5)
# pr_patch5.plot('recall', 'precision', drawstyle='steps',
#                label=f'YOLOv5x-Patch5m-AP = {round(100 * ap_patch5, 2)}%', ax=ax)
#
# det_patch4 = bb.io.load('det_coco',
#                         '/data1/lianjiawei/adversarial-yolo/testing/yolov5m_A3_faster-rcnn/patch_results.json',
#                         class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_patch4 = bb.stat.pr(det_patch4, annoF, iou_threshold)
# ap_patch4 = bb.stat.ap(pr_patch4)
# pr_patch4.plot('recall', 'precision', drawstyle='steps',
#                label=f'faster-rcnn-Patch5m-AP = {round(100 * ap_patch4, 2)}%', ax=ax)
#
# det_patch4 = bb.io.load('det_coco',
#                         '/data1/lianjiawei/adversarial-yolo/testing/yolov5m_A3_ssd/patch_results.json',
#                         class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_patch4 = bb.stat.pr(det_patch4, annoSSD, iou_threshold)
# ap_patch4 = bb.stat.ap(pr_patch4)
# pr_patch4.plot('recall', 'precision', drawstyle='steps',
#                label=f'SSD-Patch5m-AP = {round(100 * ap_patch4, 2)}%', ax=ax)
#
# det_patch4 = bb.io.load('det_coco',
#                         '/data1/lianjiawei/adversarial-yolo/testing/yolov5m_A3_swin/patch_results.json',
#                         class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_patch4 = bb.stat.pr(det_patch4, annoSwin, iou_threshold)
# ap_patch4 = bb.stat.ap(pr_patch4)
# pr_patch4.plot('recall', 'precision', drawstyle='steps',
#                label=f'Swin-Patch5m-AP = {round(100 * ap_patch4, 2)}%', ax=ax)
########################################################################################################################

# Patch5l
########################################################################################################################
# det_patch8 = bb.io.load('det_coco',
#                         '/data1/lianjiawei/adversarial-yolo/testing/yolov5l_A3_yolov2/patch_results.json',
#                         class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_patch8 = bb.stat.pr(det_patch8, anno2, iou_threshold)
# ap_patch8 = bb.stat.ap(pr_patch8)
# pr_patch8.plot('recall', 'precision', drawstyle='steps',
#                label=f'YOLOv2-Patch5l-AP = {round(100 * ap_patch8, 2)}%', ax=ax)
#
# det_patch8 = bb.io.load('det_coco',
#                         '/data1/lianjiawei/adversarial-yolo/testing/yolov5l_A3_yolov3/patch_results.json',
#                         class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_patch8 = bb.stat.pr(det_patch8, anno3, iou_threshold)
# ap_patch8 = bb.stat.ap(pr_patch8)
# pr_patch8.plot('recall', 'precision', drawstyle='steps',
#                label=f'YOLOv3-Patch5l-AP = {round(100 * ap_patch8, 2)}%', ax=ax)
#
# det_patch7 = bb.io.load('det_coco',
#                         '/data1/lianjiawei/adversarial-yolo/testing/yolov5l_A3_yolov5n/patch_results.json',
#                         class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_patch7 = bb.stat.pr(det_patch7, anno5n, iou_threshold)
# ap_patch7 = bb.stat.ap(pr_patch7)
# pr_patch7.plot('recall', 'precision', drawstyle='steps',
#                label=f'YOLOv5n-Patch5l-AP = {round(100 * ap_patch7, 2)}%', ax=ax)
#
# det_patch = bb.io.load('det_coco',
#                        '/data1/lianjiawei/adversarial-yolo/testing/yolov5l_A3_yolov5s/patch_results.json',
#                        class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_patch = bb.stat.pr(det_patch, anno5s, iou_threshold)
# ap_patch = bb.stat.ap(pr_patch)
# pr_patch.plot('recall', 'precision', drawstyle='steps',
#               label=f'YOLOv5s-Patch5l-AP = {round(100 * ap_patch, 2)}%', ax=ax)
#
# det_patch6 = bb.io.load('det_coco',
#                         '/data1/lianjiawei/adversarial-yolo/testing/yolov5l_A3_yolov5m/patch_results.json',
#                         class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_patch6 = bb.stat.pr(det_patch6, anno5m, iou_threshold)
# ap_patch6 = bb.stat.ap(pr_patch6)
# pr_patch6.plot('recall', 'precision', drawstyle='steps',
#                label=f'YOLOv5m-Patch5l-AP = {round(100 * ap_patch6, 2)}%', ax=ax)
#
# det_patch5 = bb.io.load('det_coco',
#                         '/data1/lianjiawei/adversarial-yolo/testing/yolov5l_A3_yolov5l/patch_results.json',
#                         class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_patch5 = bb.stat.pr(det_patch5, anno5l, iou_threshold)
# ap_patch5 = bb.stat.ap(pr_patch5)
# pr_patch5.plot('recall', 'precision', drawstyle='steps',
#                label=f'YOLOv5l-Patch5l-AP = {round(100 * ap_patch5, 2)}%', ax=ax)
#
# det_patch5 = bb.io.load('det_coco',
#                         '/data1/lianjiawei/adversarial-yolo/testing/yolov5l_A3_yolov5x/patch_results.json',
#                         class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_patch5 = bb.stat.pr(det_patch5, anno5x, iou_threshold)
# ap_patch5 = bb.stat.ap(pr_patch5)
# pr_patch5.plot('recall', 'precision', drawstyle='steps',
#                label=f'YOLOv5x-Patch5l-AP = {round(100 * ap_patch5, 2)}%', ax=ax)
#
# det_patch4 = bb.io.load('det_coco',
#                         '/data1/lianjiawei/adversarial-yolo/testing/yolov5l_A3_faster-rcnn/patch_results.json',
#                         class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_patch4 = bb.stat.pr(det_patch4, annoF, iou_threshold)
# ap_patch4 = bb.stat.ap(pr_patch4)
# pr_patch4.plot('recall', 'precision', drawstyle='steps',
#                label=f'faster-rcnn-Patch5l-AP = {round(100 * ap_patch4, 2)}%', ax=ax)
#
# det_patch4 = bb.io.load('det_coco',
#                         '/data1/lianjiawei/adversarial-yolo/testing/yolov5l_A3_ssd/patch_results.json',
#                         class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_patch4 = bb.stat.pr(det_patch4, annoSSD, iou_threshold)
# ap_patch4 = bb.stat.ap(pr_patch4)
# pr_patch4.plot('recall', 'precision', drawstyle='steps',
#                label=f'SSD-Patch5l-AP = {round(100 * ap_patch4, 2)}%', ax=ax)
#
# det_patch4 = bb.io.load('det_coco',
#                         '/data1/lianjiawei/adversarial-yolo/testing/yolov5l_A3_swin/patch_results.json',
#                         class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_patch4 = bb.stat.pr(det_patch4, annoSwin, iou_threshold)
# ap_patch4 = bb.stat.ap(pr_patch4)
# pr_patch4.plot('recall', 'precision', drawstyle='steps',
#                label=f'Swin-Patch5l-AP = {round(100 * ap_patch4, 2)}%', ax=ax)
########################################################################################################################

# Patch5x
########################################################################################################################
# det_patch8 = bb.io.load('det_coco',
#                         '/data1/lianjiawei/adversarial-yolo/testing/yolov5x_A3_yolov2/patch_results.json',
#                         class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_patch8 = bb.stat.pr(det_patch8, anno2, iou_threshold)
# ap_patch8 = bb.stat.ap(pr_patch8)
# pr_patch8.plot('recall', 'precision', drawstyle='steps',
#                label=f'YOLOv2-Patch5x-AP = {round(100 * ap_patch8, 2)}%', ax=ax)
#
# det_patch8 = bb.io.load('det_coco',
#                         '/data1/lianjiawei/adversarial-yolo/testing/yolov5x_A3_yolov3/patch_results.json',
#                         class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_patch8 = bb.stat.pr(det_patch8, anno3, iou_threshold)
# ap_patch8 = bb.stat.ap(pr_patch8)
# pr_patch8.plot('recall', 'precision', drawstyle='steps',
#                label=f'YOLOv3-Patch5x-AP = {round(100 * ap_patch8, 2)}%', ax=ax)
#
# det_patch7 = bb.io.load('det_coco',
#                         '/data1/lianjiawei/adversarial-yolo/testing/yolov5x_A3_yolov5n/patch_results.json',
#                         class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_patch7 = bb.stat.pr(det_patch7, anno5n, iou_threshold)
# ap_patch7 = bb.stat.ap(pr_patch7)
# pr_patch7.plot('recall', 'precision', drawstyle='steps',
#                label=f'YOLOv5n-Patch5x-AP = {round(100 * ap_patch7, 2)}%', ax=ax)
#
# det_patch = bb.io.load('det_coco',
#                        '/data1/lianjiawei/adversarial-yolo/testing/yolov5x_A3_yolov5s/patch_results.json',
#                        class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_patch = bb.stat.pr(det_patch, anno5s, iou_threshold)
# ap_patch = bb.stat.ap(pr_patch)
# pr_patch.plot('recall', 'precision', drawstyle='steps',
#               label=f'YOLOv5s-Patch5x-AP = {round(100 * ap_patch, 2)}%', ax=ax)
#
# det_patch6 = bb.io.load('det_coco',
#                         '/data1/lianjiawei/adversarial-yolo/testing/yolov5x_A3_yolov5m/patch_results.json',
#                         class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_patch6 = bb.stat.pr(det_patch6, anno5m, iou_threshold)
# ap_patch6 = bb.stat.ap(pr_patch6)
# pr_patch6.plot('recall', 'precision', drawstyle='steps',
#                label=f'YOLOv5m-Patch5x-AP = {round(100 * ap_patch6, 2)}%', ax=ax)
#
# det_patch5 = bb.io.load('det_coco',
#                         '/data1/lianjiawei/adversarial-yolo/testing/yolov5x_A3_yolov5l/patch_results.json',
#                         class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_patch5 = bb.stat.pr(det_patch5, anno5l, iou_threshold)
# ap_patch5 = bb.stat.ap(pr_patch5)
# pr_patch5.plot('recall', 'precision', drawstyle='steps',
#                label=f'YOLOv5l-Patch5x-AP = {round(100 * ap_patch5, 2)}%', ax=ax)
#
# det_patch5 = bb.io.load('det_coco',
#                         '/data1/lianjiawei/adversarial-yolo/testing/yolov5x_A3_yolov5x/patch_results.json',
#                         class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_patch5 = bb.stat.pr(det_patch5, anno5x, iou_threshold)
# ap_patch5 = bb.stat.ap(pr_patch5)
# pr_patch5.plot('recall', 'precision', drawstyle='steps',
#                label=f'YOLOv5x-Patch5x-AP = {round(100 * ap_patch5, 2)}%', ax=ax)
#
# det_patch4 = bb.io.load('det_coco',
#                         '/data1/lianjiawei/adversarial-yolo/testing/yolov5x_A3_faster-rcnn/patch_results.json',
#                         class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_patch4 = bb.stat.pr(det_patch4, annoF, iou_threshold)
# ap_patch4 = bb.stat.ap(pr_patch4)
# pr_patch4.plot('recall', 'precision', drawstyle='steps',
#                label=f'faster-rcnn-Patch5x-AP = {round(100 * ap_patch4, 2)}%', ax=ax)
#
# det_patch4 = bb.io.load('det_coco',
#                         '/data1/lianjiawei/adversarial-yolo/testing/yolov5x_A3_ssd/patch_results.json',
#                         class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_patch4 = bb.stat.pr(det_patch4, annoSSD, iou_threshold)
# ap_patch4 = bb.stat.ap(pr_patch4)
# pr_patch4.plot('recall', 'precision', drawstyle='steps',
#                label=f'SSD-Patch5x-AP = {round(100 * ap_patch4, 2)}%', ax=ax)
#
# det_patch4 = bb.io.load('det_coco',
#                         '/data1/lianjiawei/adversarial-yolo/testing/yolov5x_A3_swin/patch_results.json',
#                         class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_patch4 = bb.stat.pr(det_patch4, annoSwin, iou_threshold)
# ap_patch4 = bb.stat.ap(pr_patch4)
# pr_patch4.plot('recall', 'precision', drawstyle='steps',
#                label=f'Swin-Patch5x-AP = {round(100 * ap_patch4, 2)}%', ax=ax)
########################################################################################################################

# faster-rcnn
########################################################################################################################
# det_patch8 = bb.io.load('det_coco',
#                         '/data1/lianjiawei/adversarial-yolo/testing/faster-rcnn_A3_yolov2/patch_results.json',
#                         class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_patch8 = bb.stat.pr(det_patch8, anno2, iou_threshold)
# ap_patch8 = bb.stat.ap(pr_patch8)
# pr_patch8.plot('recall', 'precision', drawstyle='steps',
#                label=f'YOLOv2-PatchF-AP = {round(100 * ap_patch8, 2)}%', ax=ax)
# #
# det_patch8 = bb.io.load('det_coco',
#                         '/data1/lianjiawei/adversarial-yolo/testing/faster-rcnn_A3_yolov3/patch_results.json',
#                         class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_patch8 = bb.stat.pr(det_patch8, anno3, iou_threshold)
# ap_patch8 = bb.stat.ap(pr_patch8)
# pr_patch8.plot('recall', 'precision', drawstyle='steps',
#                label=f'YOLOv3-PatchF-AP = {round(100 * ap_patch8, 2)}%', ax=ax)
# #
# det_patch7 = bb.io.load('det_coco',
#                         '/data1/lianjiawei/adversarial-yolo/testing/faster-rcnn_A3_yolov5n/patch_results.json',
#                         class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_patch7 = bb.stat.pr(det_patch7, anno5n, iou_threshold)
# ap_patch7 = bb.stat.ap(pr_patch7)
# pr_patch7.plot('recall', 'precision', drawstyle='steps',
#                label=f'YOLOv5n-PatchF-AP = {round(100 * ap_patch7, 2)}%', ax=ax)
# #
# det_patch = bb.io.load('det_coco',
#                        '/data1/lianjiawei/adversarial-yolo/testing/faster-rcnn_A3_yolov5s/patch_results.json',
#                        class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_patch = bb.stat.pr(det_patch, anno5s, iou_threshold)
# ap_patch = bb.stat.ap(pr_patch)
# pr_patch.plot('recall', 'precision', drawstyle='steps',
#               label=f'YOLOv5s-PatchF-AP = {round(100 * ap_patch, 2)}%', ax=ax)
# #
# det_patch6 = bb.io.load('det_coco',
#                         '/data1/lianjiawei/adversarial-yolo/testing/faster-rcnn_A3_yolov5m/patch_results.json',
#                         class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_patch6 = bb.stat.pr(det_patch6, anno5m, iou_threshold)
# ap_patch6 = bb.stat.ap(pr_patch6)
# pr_patch6.plot('recall', 'precision', drawstyle='steps',
#                label=f'YOLOv5m-PatchF-AP = {round(100 * ap_patch6, 2)}%', ax=ax)
# #
# det_patch5 = bb.io.load('det_coco',
#                         '/data1/lianjiawei/adversarial-yolo/testing/faster-rcnn_A3_yolov5l/patch_results.json',
#                         class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_patch5 = bb.stat.pr(det_patch5, anno5l, iou_threshold)
# ap_patch5 = bb.stat.ap(pr_patch5)
# pr_patch5.plot('recall', 'precision', drawstyle='steps',
#                label=f'YOLOv5l-PatchF-AP = {round(100 * ap_patch5, 2)}%', ax=ax)
#
# det_patch5 = bb.io.load('det_coco',
#                         '/data1/lianjiawei/adversarial-yolo/testing/faster-rcnn_A3_yolov5x/patch_results.json',
#                         class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_patch5 = bb.stat.pr(det_patch5, anno5x, iou_threshold)
# ap_patch5 = bb.stat.ap(pr_patch5)
# pr_patch5.plot('recall', 'precision', drawstyle='steps',
#                label=f'YOLOv5x-PatchF-AP = {round(100 * ap_patch5, 2)}%', ax=ax)
# #
# det_patch5 = bb.io.load('det_coco',
#                         '/data1/lianjiawei/adversarial-yolo/testing/faster-rcnn_A3_faster-rcnn/patch_results.json',
#                         class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_patch5 = bb.stat.pr(det_patch5, annoF, iou_threshold)
# ap_patch5 = bb.stat.ap(pr_patch5)
# pr_patch5.plot('recall', 'precision', drawstyle='steps',
#                label=f'faster-rcnn-PatchF-AP = {round(100 * ap_patch5, 2)}%', ax=ax)
#
# det_patch4 = bb.io.load('det_coco',
#                         '/data1/lianjiawei/adversarial-yolo/testing/faster-rcnn_A3_ssd/patch_results.json',
#                         class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_patch4 = bb.stat.pr(det_patch4, annoSSD, iou_threshold)
# ap_patch4 = bb.stat.ap(pr_patch4)
# pr_patch4.plot('recall', 'precision', drawstyle='steps',
#                label=f'SSD-PatchF-AP = {round(100 * ap_patch4, 2)}%', ax=ax)
#
# det_patch4 = bb.io.load('det_coco',
#                         '/data1/lianjiawei/adversarial-yolo/testing/faster-rcnn_A3_swin/patch_results.json',
#                         class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_patch4 = bb.stat.pr(det_patch4, annoSwin, iou_threshold)
# ap_patch4 = bb.stat.ap(pr_patch4)
# pr_patch4.plot('recall', 'precision', drawstyle='steps',
#                label=f'Swin-PatchF-AP = {round(100 * ap_patch4, 2)}%', ax=ax)
########################################################################################################################

# PatchSSD
########################################################################################################################
# det_patch2 = bb.io.load('det_coco',
#                         '/data1/lianjiawei/adversarial-yolo/testing/ssd_A3_yolov2/patch_results.json',
#                         class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_patch2 = bb.stat.pr(det_patch2, anno2, iou_threshold)
# ap_patch2 = bb.stat.ap(pr_patch2)
# pr_patch2.plot('recall', 'precision', drawstyle='steps',
#                label=f'yolov2-PatchSSD-AP = {round(100 * ap_patch2, 2)}%', ax=ax)
# #
# det_patch2 = bb.io.load('det_coco',
#                         '/data1/lianjiawei/adversarial-yolo/testing/ssd_A3_yolov3/patch_results.json',
#                         class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_patch2 = bb.stat.pr(det_patch2, anno3, iou_threshold)
# ap_patch2 = bb.stat.ap(pr_patch2)
# pr_patch2.plot('recall', 'precision', drawstyle='steps',
#                label=f'yolov3-PatchSSD-AP = {round(100 * ap_patch2, 2)}%', ax=ax)
# #
# det_patch2 = bb.io.load('det_coco',
#                         '/data1/lianjiawei/adversarial-yolo/testing/ssd_A3_yolov5n/patch_results.json',
#                         class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_patch2 = bb.stat.pr(det_patch2, anno5n, iou_threshold)
# ap_patch2 = bb.stat.ap(pr_patch2)
# pr_patch2.plot('recall', 'precision', drawstyle='steps',
#                label=f'yolov5n-PatchSSD-AP = {round(100 * ap_patch2, 2)}%', ax=ax)
# #
# det_patch2 = bb.io.load('det_coco',
#                         '/data1/lianjiawei/adversarial-yolo/testing/ssd_A3_yolov5s/patch_results.json',
#                         class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_patch2 = bb.stat.pr(det_patch2, anno5s, iou_threshold)
# ap_patch2 = bb.stat.ap(pr_patch2)
# pr_patch2.plot('recall', 'precision', drawstyle='steps',
#                label=f'yolov5s-PatchSSD-AP = {round(100 * ap_patch2, 2)}%', ax=ax)
# #
# det_patch2 = bb.io.load('det_coco',
#                         '/data1/lianjiawei/adversarial-yolo/testing/ssd_A3_yolov5m/patch_results.json',
#                         class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_patch2 = bb.stat.pr(det_patch2, anno5m, iou_threshold)
# ap_patch2 = bb.stat.ap(pr_patch2)
# pr_patch2.plot('recall', 'precision', drawstyle='steps',
#                label=f'yolov5m-PatchSSD-AP = {round(100 * ap_patch2, 2)}%', ax=ax)
# #
# det_patch2 = bb.io.load('det_coco',
#                         '/data1/lianjiawei/adversarial-yolo/testing/ssd_A3_yolov5l/patch_results.json',
#                         class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_patch2 = bb.stat.pr(det_patch2, anno5l, iou_threshold)
# ap_patch2 = bb.stat.ap(pr_patch2)
# pr_patch2.plot('recall', 'precision', drawstyle='steps',
#                label=f'yolov5l-PatchSSD-AP = {round(100 * ap_patch2, 2)}%', ax=ax)
#
# det_patch2 = bb.io.load('det_coco',
#                         '/data1/lianjiawei/adversarial-yolo/testing/ssd_A3_yolov5x/patch_results.json',
#                         class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_patch2 = bb.stat.pr(det_patch2, anno5x, iou_threshold)
# ap_patch2 = bb.stat.ap(pr_patch2)
# pr_patch2.plot('recall', 'precision', drawstyle='steps',
#                label=f'yolov5x-PatchSSD-AP = {round(100 * ap_patch2, 2)}%', ax=ax)
# #
# det_patch2 = bb.io.load('det_coco',
#                         '/data1/lianjiawei/adversarial-yolo/testing/ssd_A3_faster-rcnn/patch_results.json',
#                         class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_patch2 = bb.stat.pr(det_patch2, annoF, iou_threshold)
# ap_patch2 = bb.stat.ap(pr_patch2)
# pr_patch2.plot('recall', 'precision', drawstyle='steps',
#                label=f'faster-rcnn-PatchSSD-AP = {round(100 * ap_patch2, 2)}%', ax=ax)
#
# det_patch2 = bb.io.load('det_coco',
#                         '/data1/lianjiawei/adversarial-yolo/testing/ssd_A3_ssd/patch_results.json',
#                         class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_patch2 = bb.stat.pr(det_patch2, annoSSD, iou_threshold)
# ap_patch2 = bb.stat.ap(pr_patch2)
# pr_patch2.plot('recall', 'precision', drawstyle='steps',
#                label=f'SSD-PatchSSD-AP = {round(100 * ap_patch2, 2)}%', ax=ax)
#
# det_patch2 = bb.io.load('det_coco',
#                         '/data1/lianjiawei/adversarial-yolo/testing/ssd_A3_swin/patch_results.json',
#                         class_label_map=[lines.strip() for lines in open('./data/dota.names')])
# pr_patch2 = bb.stat.pr(det_patch2, annoSwin, iou_threshold)
# ap_patch2 = bb.stat.ap(pr_patch2)
# pr_patch2.plot('recall', 'precision', drawstyle='steps',
#                label=f'Swin-PatchSSD-AP = {round(100 * ap_patch2, 2)}%', ax=ax)
########################################################################################################################

# PatchSwin
########################################################################################################################
det_patch2 = bb.io.load('det_coco',
                        '/data1/lianjiawei/adversarial-yolo/testing/swin_A3_yolov2/patch_results.json',
                        class_label_map=[lines.strip() for lines in open('./data/dota.names')])
pr_patch2 = bb.stat.pr(det_patch2, anno2, iou_threshold)
ap_patch2 = bb.stat.ap(pr_patch2)
pr_patch2.plot('recall', 'precision', drawstyle='steps',
               label=f'yolov2-PatchSwin-AP = {round(100 * ap_patch2, 2)}%', ax=ax)

det_patch2 = bb.io.load('det_coco',
                        '/data1/lianjiawei/adversarial-yolo/testing/swin_A3_yolov3/patch_results.json',
                        class_label_map=[lines.strip() for lines in open('./data/dota.names')])
pr_patch2 = bb.stat.pr(det_patch2, anno3, iou_threshold)
ap_patch2 = bb.stat.ap(pr_patch2)
pr_patch2.plot('recall', 'precision', drawstyle='steps',
               label=f'yolov3-PatchSwin-AP = {round(100 * ap_patch2, 2)}%', ax=ax)

det_patch2 = bb.io.load('det_coco',
                        '/data1/lianjiawei/adversarial-yolo/testing/swin_A3_yolov5n/patch_results.json',
                        class_label_map=[lines.strip() for lines in open('./data/dota.names')])
pr_patch2 = bb.stat.pr(det_patch2, anno5n, iou_threshold)
ap_patch2 = bb.stat.ap(pr_patch2)
pr_patch2.plot('recall', 'precision', drawstyle='steps',
               label=f'yolov5n-PatchSwin-AP = {round(100 * ap_patch2, 2)}%', ax=ax)

det_patch2 = bb.io.load('det_coco',
                        '/data1/lianjiawei/adversarial-yolo/testing/swin_A3_yolov5s/patch_results.json',
                        class_label_map=[lines.strip() for lines in open('./data/dota.names')])
pr_patch2 = bb.stat.pr(det_patch2, anno5s, iou_threshold)
ap_patch2 = bb.stat.ap(pr_patch2)
pr_patch2.plot('recall', 'precision', drawstyle='steps',
               label=f'yolov5s-PatchSwin-AP = {round(100 * ap_patch2, 2)}%', ax=ax)

det_patch2 = bb.io.load('det_coco',
                        '/data1/lianjiawei/adversarial-yolo/testing/swin_A3_yolov5m/patch_results.json',
                        class_label_map=[lines.strip() for lines in open('./data/dota.names')])
pr_patch2 = bb.stat.pr(det_patch2, anno5m, iou_threshold)
ap_patch2 = bb.stat.ap(pr_patch2)
pr_patch2.plot('recall', 'precision', drawstyle='steps',
               label=f'yolov5m-PatchSwin-AP = {round(100 * ap_patch2, 2)}%', ax=ax)

det_patch2 = bb.io.load('det_coco',
                        '/data1/lianjiawei/adversarial-yolo/testing/swin_A3_yolov5l/patch_results.json',
                        class_label_map=[lines.strip() for lines in open('./data/dota.names')])
pr_patch2 = bb.stat.pr(det_patch2, anno5l, iou_threshold)
ap_patch2 = bb.stat.ap(pr_patch2)
pr_patch2.plot('recall', 'precision', drawstyle='steps',
               label=f'yolov5l-PatchSwin-AP = {round(100 * ap_patch2, 2)}%', ax=ax)

det_patch2 = bb.io.load('det_coco',
                        '/data1/lianjiawei/adversarial-yolo/testing/swin_A3_yolov5x/patch_results.json',
                        class_label_map=[lines.strip() for lines in open('./data/dota.names')])
pr_patch2 = bb.stat.pr(det_patch2, anno5x, iou_threshold)
ap_patch2 = bb.stat.ap(pr_patch2)
pr_patch2.plot('recall', 'precision', drawstyle='steps',
               label=f'yolov5x-PatchSwin-AP = {round(100 * ap_patch2, 2)}%', ax=ax)

det_patch2 = bb.io.load('det_coco',
                        '/data1/lianjiawei/adversarial-yolo/testing/swin_A3_faster-rcnn/patch_results.json',
                        class_label_map=[lines.strip() for lines in open('./data/dota.names')])
pr_patch2 = bb.stat.pr(det_patch2, annoF, iou_threshold)
ap_patch2 = bb.stat.ap(pr_patch2)
pr_patch2.plot('recall', 'precision', drawstyle='steps',
               label=f'faster-rcnn-PatchSwin-AP = {round(100 * ap_patch2, 2)}%', ax=ax)

det_patch2 = bb.io.load('det_coco',
                        '/data1/lianjiawei/adversarial-yolo/testing/swin_A3_ssd/patch_results.json',
                        class_label_map=[lines.strip() for lines in open('./data/dota.names')])
pr_patch2 = bb.stat.pr(det_patch2, annoSSD, iou_threshold)
ap_patch2 = bb.stat.ap(pr_patch2)
pr_patch2.plot('recall', 'precision', drawstyle='steps',
               label=f'SSD-PatchSwin-AP = {round(100 * ap_patch2, 2)}%', ax=ax)
#
det_patch2 = bb.io.load('det_coco',
                        '/data1/lianjiawei/adversarial-yolo/testing/swin_A3_swin/patch_results.json',
                        class_label_map=[lines.strip() for lines in open('./data/dota.names')])
pr_patch2 = bb.stat.pr(det_patch2, annoSwin, iou_threshold)
ap_patch2 = bb.stat.ap(pr_patch2)
pr_patch2.plot('recall', 'precision', drawstyle='steps',
               label=f'Swin-PatchSwin-AP = {round(100 * ap_patch2, 2)}%', ax=ax)
########################################################################################################################

# Draw a dashed line(y = x)
plt.plot([0, 1.05], [0, 1.05], '--', color='gray')

ax.set_xlim([0, 1.05])
ax.set_ylim([0, 1.05])
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
# plt.legend(bbox_to_anchor=(num1, num2), loc=4, borderaxespad=num4)
# 'best': 0, 'upper right': 1, 'upper left': 2, 'lower left': 3, 'lower right': 4
plt.gcf().subplots_adjust(bottom=0.3, right=0.63, left=0.05)  # 在此添加修改参数的代码
plt.legend(fontsize=8, bbox_to_anchor=(1.01, 1.1), borderaxespad=0)
plt.title("Misc Attack")
# plt.savefig("pr_curve_Misc Attack.png")
plt.show()

# CUDA_VISIBLE_DEVICES=7 python pr_curve_all.py
# 在Mobaxterm上可以运行
