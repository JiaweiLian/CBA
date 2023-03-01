import sys
import time
from PIL import Image, ImageDraw
from utils import *
from darknet import Darknet


os.environ["CUDA_VISIBLE_DEVICES"] = '2'


def detect(cfgfile, weightfile, imgfile):
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if m.num_classes == 20:
        namesfile = 'data/voc.names'
    elif m.num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/dota.names'

    use_cuda = 1
    if use_cuda:
        m.cuda()

    img = Image.open(imgfile).convert('RGB')
    sized = img.resize((m.width, m.height))

    # Simen: niet nodig om dit in loop te doen? 
    # for i in range(2):
    start = time.time()
    boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
    finish = time.time()

    # if i == 1:
    print('%s: Predicted in %f seconds.' % (imgfile, (finish - start)))

    class_names = load_class_names(namesfile)
    image_name = imgfile.split('/')[-1]
    plot_boxes(img, boxes, image_name, class_names)


def detect_all(cfgfile, weightfile, imgdir, savedir): # TODO: Detection use yolo2, yolo3, yolo5 and faster-rcnn.
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if m.num_classes == 20:
        namesfile = 'data/voc.names'
    elif m.num_classes == 80:
        namesfile = 'data/coco.names'
    elif m.num_classes == 15:
        namesfile = 'data/dota.names'
    else:
        print('Please input right num_classes!')

    use_cuda = 1
    if use_cuda:
        m.cuda()

    for imgfile in os.listdir(imgdir):
        if imgfile.endswith('.jpg') or imgfile.endswith('.png'):
            imgfile = os.path.abspath(os.path.join(imgdir, imgfile))
            img = Image.open(imgfile).convert('RGB')
            sized = img.resize((m.width, m.height))
            boxes = do_detect(m, sized, 0.3, 0.4, use_cuda)
            class_names = load_class_names(namesfile)
            image_name = imgfile.split('/')[-1]

            if not os.path.exists(savedir):
                os.makedirs(savedir)
            plot_boxes(img, boxes, image_name, class_names, savedir)


def detect_cv2(cfgfile, weightfile, imgfile):
    import cv2
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if m.num_classes == 20:
        namesfile = 'data/voc.names'
    elif m.num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/names'

    use_cuda = 1
    if use_cuda:
        m.cuda()

    img = cv2.imread(imgfile)
    sized = cv2.resize(img, (m.width, m.height))
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

    for i in range(2):
        start = time.time()
        boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
        finish = time.time()
        if i == 1:
            print('%s: Predicted in %f seconds.' % (imgfile, (finish - start)))

    class_names = load_class_names(namesfile)
    plot_boxes_cv2(img, boxes, savename='predictions.jpg', class_names=class_names)


def detect_skimage(cfgfile, weightfile, imgfile):
    from skimage import io
    from skimage.transform import resize
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if m.num_classes == 20:
        namesfile = 'data/voc.names'
    elif m.num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/names'

    use_cuda = 1
    if use_cuda:
        m.cuda()

    img = io.imread(imgfile)
    sized = resize(img, (m.width, m.height)) * 255

    for i in range(2):
        start = time.time()
        boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
        finish = time.time()
        if i == 1:
            print('%s: Predicted in %f seconds.' % (imgfile, (finish - start)))

    class_names = load_class_names(namesfile)
    plot_boxes_cv2(img, boxes, savename='predictions.jpg', class_names=class_names)


if __name__ == '__main__':

    # detect("cfg/yolo-dota.cfg",
    #        "weights/yolo-dota.cfg_450000.weights",
    #        "testing/plane/proper_patched/aircraft_4_p.png")

    # / testing / yolov2_mean_center_50_1024_yolov2 / proper_patched /
    detect_all("cfg/yolo-dota.cfg",
               "weights/yolo-dota.cfg_450000.weights",
               "detect_result/physical_patch/",
               "detect_result/results_paper/test")

#     if len(sys.argv) == 4:
#         cfgfile = sys.argv[1]
#         weightfile = sys.argv[2]
#         imgfile = sys.argv[3]
#         # detect("cfg/yolo.cfg", "weights/yolov2.weights", "test/img/crop001024.png")
#         detect(cfgfile, weightfile, imgfile)
#         # detect_cv2(cfgfile, weightfile, imgfile)
#         # detect_skimage(cfgfile, weightfile, imgfile)
#     else:
#         print('Usage: ')
#         print('  python detect.py cfgfile weightfile imgfile')
#         # detect('cfg/tiny-yolo-voc.cfg', 'tiny-yolo-voc.weights', 'data/person.jpg', version=1)
#         # python detect.py "cfg/yolo.cfg" "weights/yolov2.weights" "test/img/crop001024.png"
#
# python detect.py "cfg/yolo.cfg" "weights/yolov2.weights" "datasets/INRIADATA/original_images/test/pos/person_and_bike_188.png"
# python detect.py "cfg/yolo-dota.cfg" "weights/yolo-dota.cfg_450000.weights" "datasets/RSOD-Dataset/aircraft/JPEGImages/aircraft_23.jpg"
# python detect.py "cfg/yolo-dota.cfg" "weights/yolo-dota.cfg_450000.weights" "testing/plane/proper_patched/aircraft_4_p.png"
