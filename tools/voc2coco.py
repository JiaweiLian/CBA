# coding:utf-8

# pip install lxml

import os
import glob
import json
import shutil
import numpy as np
import xml.etree.ElementTree as ET

START_BOUNDING_BOX_ID = 1


def get(root, name):
    return root.findall(name)


def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise NotImplementedError('Can not find %s in %s.' % (name, root.tag))
    if length > 0 and len(vars) != length:
        raise NotImplementedError('The size of %s is supposed to be %d, but is %d.' % (name, length, len(vars)))
    if length == 1:
        vars = vars[0]
    return vars


def convert(xml_list, json_file):
    json_dict = {"info": ['none'], "license": ['none'], "images": [], "annotations": [], "categories": []}
    categories = pre_define_categories.copy()
    bnd_id = START_BOUNDING_BOX_ID
    all_categories = {}
    for index, line in enumerate(xml_list):
        # print("Processing %s"%(line))
        xml_f = line
        tree = ET.parse(xml_f)
        root = tree.getroot()

        filename = os.path.basename(xml_f)[:-4] + ".jpg"

        # image_id = filename.split('.')[0][-3:].replace('t', '0')
        # image_id = int(image_id.replace('_', '0'))
        image_id = filename.split('.')[0]
        #         print('filename is {}'.format(image_id))

        size = get_and_check(root, 'size', 1)
        width = int(get_and_check(size, 'width', 1).text)
        height = int(get_and_check(size, 'height', 1).text)
        image = {'file_name': filename, 'height': height, 'width': width, 'id': image_id}
        json_dict['images'].append(image)
        ## Cruuently we do not support segmentation
        #  segmented = get_and_check(root, 'segmented', 1).text
        #  assert segmented == '0'
        for obj in get(root, 'object'):
            category = get_and_check(obj, 'name', 1).text
            if category in all_categories:
                all_categories[category] += 1
            else:
                all_categories[category] = 1
            if category not in categories:
                if only_care_pre_define_categories:
                    continue
                new_id = len(categories) + 1
                print(
                    "[warning] category '{}' not in 'pre_define_categories'({}), create new id: {} automatically".format(
                        category, pre_define_categories, new_id))
                categories[category] = new_id
            category_id = categories[category]
            bndbox = get_and_check(obj, 'bndbox', 1)
            xmin = int(float(get_and_check(bndbox, 'xmin', 1).text))
            ymin = int(float(get_and_check(bndbox, 'ymin', 1).text))
            xmax = int(float(get_and_check(bndbox, 'xmax', 1).text))
            ymax = int(float(get_and_check(bndbox, 'ymax', 1).text))
            assert (xmax > xmin), "xmax <= xmin, {}".format(line)
            assert (ymax > ymin), "ymax <= ymin, {}".format(line)
            o_width = abs(xmax - xmin)
            o_height = abs(ymax - ymin)
            ann = {'area': o_width * o_height, 'iscrowd': 0, 'image_id':
                image_id, 'bbox': [xmin, ymin, o_width, o_height],
                   'category_id': category_id, 'id': bnd_id, 'ignore': 0,
                   'segmentation': []}
            json_dict['annotations'].append(ann)
            bnd_id = bnd_id + 1

    for cate, cid in categories.items():
        cat = {'supercategory': 'none', 'id': cid, 'name': cate}
        json_dict['categories'].append(cat)
    json_fp = open(json_file, 'w')
    json_str = json.dumps(json_dict)
    json_fp.write(json_str)
    json_fp.close()
    print("------------create {} done--------------".format(json_file))
    print("find {} categories: {} -->>> your pre_define_categories {}: {}".format(len(all_categories),
                                                                                  all_categories.keys(),
                                                                                  len(pre_define_categories),
                                                                                  pre_define_categories.keys()))
    print("category: id --> {}".format(categories))
    print(categories.keys())
    print(categories.values())


if __name__ == '__main__':
    # xml标注文件夹
    xml_dir = 'xml'
    # 训练数据的josn文件
    save_json_train = 'aircraft.json'
    # # 验证数据的josn文件
    # save_json_val = 'D:\\AI_Kaggle_Study\\voc2coco\\dog_json\\dog_val.json'

    # 类别，这里只有dog一个类别，如果是多个类别，往classes中添加类别名字即可，比如['dog', 'person', 'cat']
    classes = ['aircraft']
    pre_define_categories = {}
    for i, cls in enumerate(classes):
        pre_define_categories[cls] = i + 1

    only_care_pre_define_categories = True

    # 训练数据集比例
    train_ratio = 1

    print('xml_dir is {}'.format(xml_dir))
    xml_list = glob.glob(xml_dir + "/*.xml")
    xml_list = np.sort(xml_list)
    #     print('xml_list is {}'.format(xml_list))
    # np.random.seed(100)
    # np.random.shuffle(xml_list)

    train_num = int(len(xml_list) * train_ratio)
    print('训练样本数目是 {}'.format(train_num))
    print('测试样本数目是 {}'.format(len(xml_list) - train_num))
    xml_list_train = xml_list[:train_num]
    # xml_list_val = xml_list[train_num:]

    # 对训练数据集对应的xml进行coco转换
    convert(xml_list_train, save_json_train)
    # # 对验证数据集的xml进行coco转换
    # convert(xml_list_val, save_json_val)


