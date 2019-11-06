
import os
import json
import numpy as np

from .detection import DETECTION
from ..paths import get_file_path


import xml.etree.ElementTree as ET
import os
import json



class VOC(DETECTION):

    def __init__(self, db_config, split=None, sys_config=None):
        assert split is None or sys_config is not None
        super(VOC, self).__init__(db_config)
        self._mean    = np.array([0.40789654, 0.44719302, 0.47026115], dtype=np.float32)
        self._std     = np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571], dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)

        self._voc_cls_ids = [
            1,
            #  2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
            # 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            # 24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
            # 37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
            # 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
            # 58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
            # 72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
            # 82, 84, 85, 86, 87, 88, 89, 90
        ]

        self._voc_cls_names = [
            'person',
            #  'bicycle', 'car', 'motorcycle', 'airplane',
            # 'bus', 'train', 'truck', 'boat', 'traffic light',
            # 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            # 'bird', 'cat', 'dog', 'horse','sheep', 'cow', 'elephant',
            # 'bear', 'zebra','giraffe', 'backpack', 'umbrella',
            # 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
            # 'snowboard','sports ball', 'kite', 'baseball bat',
            # 'baseball glove', 'skateboard', 'surfboard',
            # 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
            # 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
            # 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            # 'donut', 'cake', 'chair', 'couch', 'potted plant',
            # 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            # 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
            # 'oven', 'toaster', 'sink', 'refrigerator', 'book',
            # 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            # 'toothbrush'
        ]

        # self._cls2voc  = {1:1,}
        # self._voc2cls  = {1:1,}
        # self._voc2name = {1:'person',}
        # self._name2voc = {1:'person',}
        self._cls2voc  = {ind + 1: voc_id for ind, voc_id in enumerate(self._voc_cls_ids)}
        self._voc2cls  = {voc_id: cls_id for cls_id, voc_id in self._cls2voc.items()}
        self._voc2name = {cls_id: cls_name for cls_id, cls_name in zip(self._voc_cls_ids, self._voc_cls_names)}
        self._name2voc = {cls_name: cls_id for cls_name, cls_id in self._voc2name.items()}

        if split is not None:
            voc_dir = os.path.join('/home/rock/CornerNet-Lite-master/data/', "coco_pd")
            #coco_dir='/media/diskData/huanglong_data/coco'
            # self._split     = {
            #     "trainval": "train2017",
            #     "minival":  "",
            #     "testdev":  "val2017"
            # }[split]
            self._data_dir  = os.path.join(voc_dir, 'images')
            # '/home/rock/CornerNet-Lite-master/data/coco_pd/Annotations'
            self.xml_path = os.path.join(voc_dir, "Annotations")

            self._detections, self._eval_ids = self._load_voc_annos()
            self._image_ids = list(self._detections.keys())
            self._db_inds   = np.arange(len(self._image_ids))


    def _load_voc_annos(self):
        eval_ids = {}
        detections = {}
        i = 0
        xml_path='/home/rock/CornerNet-Lite-master/data/coco_pd/Annotations'
        for f in os.listdir(xml_path):
            res = []
            if not f.endswith('.xml'):
                continue
            name = f.rstrip('.xml') + str('.jpg')
            eval_ids[name] = i
            i = i + 1

            bndbox = dict()
            size = dict()
            current_image_id = None
            current_category_id = None
            file_name = None
            size['width'] = None
            size['height'] = None
            size['depth'] = None

            xml_file = os.path.join(xml_path, f)
            # print(xml_file)

            tree = ET.parse(xml_file)
            root = tree.getroot()
            if root.tag != 'annotation':
                raise Exception('pascal voc xml root element should be annotation, rather than {}'.format(root.tag))

                # elem is <folder>, <filename>, <size>, <object>
            for elem in root:
                current_parent = elem.tag
                current_sub = None
                object_name = None

                if elem.tag == 'folder':
                    continue

                if elem.tag == 'filename':
                    file_name = elem.text

                for subelem in elem:
                    bndbox['xmin'] = None
                    bndbox['xmax'] = None
                    bndbox['ymin'] = None
                    bndbox['ymax'] = None

                    current_sub = subelem.tag
                    if current_parent == 'object' and subelem.tag == 'name':
                        object_name = subelem.text


                    elif current_parent == 'size':
                        if size[subelem.tag] is not None:
                            raise Exception('xml structure broken at size tag.')
                        size[subelem.tag] = int(subelem.text)

                        # option is <xmin>, <ymin>, <xmax>, <ymax>, when subelem is <bndbox>
                    for option in subelem:
                        if current_sub == 'bndbox':
                            if bndbox[option.tag] is not None:
                                raise Exception('xml structure corrupted at bndbox tag.')
                            bndbox[option.tag] = int(option.text)

                            # only after parse the <object> tag
                    if bndbox['xmin'] is not None:
                        bbox = []

                        # x
                        bbox.append(bndbox['xmin'])
                        # y
                        bbox.append(bndbox['ymin'])
                        # w
                        bbox.append(bndbox['xmax'])
                        # h
                        bbox.append(bndbox['ymax'])
                        bbox.append(1)
                        res.append(bbox)
                        # print(res)
            if len(res) == 0:
                detections[name] = np.zeros((0, 5), dtype=np.float32)
            else:
                detections[name] = np.array(res, dtype=np.float32)
        return detections, eval_ids

    def image_path(self, ind):
        file_name = self._image_ids[ind]
        return os.path.join(self._data_dir, file_name)

    def detections(self, ind):
        file_name = self._image_ids[ind]
        return self._detections[file_name].copy()

    def cls2name(self, cls):
        voc = self._cls2voc[cls]
        return self._voc2name[voc]
    # def cls2name(self, cls):
    #     return "person"


if __name__ == '__main__':

    detections, eval_ids = _load_voc_annos(xml_path)
    print(detections, eval_ids)





















































# #class VOCDataset(XMLDataset):
#
#     # CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
#     #            'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
#     #            'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
#     #            'tvmonitor')
#
#
# import argparse
# import os.path as osp
# import xml.etree.ElementTree as ET
#
# import mmcv
# import numpy as np
#
#
#
# label_ids= {'person':1}
#
# #label_ids = {name: i + 1 for i, name in enumerate(voc_classes())}
#
#
# def parse_xml(args):
#     xml_path, img_path = args
#     tree = ET.parse(xml_path)
#     root = tree.getroot()
#     size = root.find('size')
#     w = int(size.find('width').text)
#     h = int(size.find('height').text)
#     bboxes = []
#     labels = []
#     bboxes_ignore = []
#     labels_ignore = []
#     for obj in root.findall('object'):
#         name = obj.find('name').text
#         label = label_ids[name]
#         difficult = int(obj.find('difficult').text)
#         bnd_box = obj.find('bndbox')
#         bbox = [
#             int(bnd_box.find('xmin').text),
#             int(bnd_box.find('ymin').text),
#             int(bnd_box.find('xmax').text),
#             int(bnd_box.find('ymax').text)
#         ]
#         if difficult:
#             bboxes_ignore.append(bbox)
#             labels_ignore.append(label)
#         else:
#             bboxes.append(bbox)
#             labels.append(label)
#     if not bboxes:
#         bboxes = np.zeros((0, 4))
#         labels = np.zeros((0, ))
#     else:
#         bboxes = np.array(bboxes, ndmin=2) - 1
#         labels = np.array(labels)
#     if not bboxes_ignore:
#         bboxes_ignore = np.zeros((0, 4))
#         labels_ignore = np.zeros((0, ))
#     else:
#         bboxes_ignore = np.array(bboxes_ignore, ndmin=2) - 1
#         labels_ignore = np.array(labels_ignore)
#     annotation = {
#         'filename': img_path,
#         'width': w,
#         'height': h,
#         'ann': {
#             'bboxes': bboxes.astype(np.float32),
#             'labels': labels.astype(np.int64),
#             'bboxes_ignore': bboxes_ignore.astype(np.float32),
#             'labels_ignore': labels_ignore.astype(np.int64)
#         }
#     }
#     return annotation
#
#
# def cvt_annotations(devkit_path, years, split, out_file):
#     if not isinstance(years, list):
#         years = [years]
#     annotations = []
#     for year in years:
#         filelist = osp.join(devkit_path,
#                             'VOC{}/ImageSets/Main/{}.txt'.format(year, split))
#         if not osp.isfile(filelist):
#             print('filelist does not exist: {}, skip voc{} {}'.format(
#                 filelist, year, split))
#             return
#         img_names = mmcv.list_from_file(filelist)
#         xml_paths = [
#             osp.join(devkit_path,
#                      'VOC{}/Annotations/{}.xml'.format(year, img_name))
#             for img_name in img_names
#         ]
#         img_paths = [
#             'VOC{}/JPEGImages/{}.jpg'.format(year, img_name)
#             for img_name in img_names
#         ]
#         part_annotations = mmcv.track_progress(parse_xml,
#                                                list(zip(xml_paths, img_paths)))
#         annotations.extend(part_annotations)
#     mmcv.dump(annotations, out_file)
#     return annotations
#
#
# def parse_args():
#     parser = argparse.ArgumentParser(
#         description='Convert PASCAL VOC annotations to mmdetection format')
#     parser.add_argument('devkit_path', help='pascal voc devkit path')
#     parser.add_argument('-o', '--out-dir', help='output path')
#     args = parser.parse_args()
#     return args
#
#
# def main():
#     #args = parse_args()
#     # devkit_path = args.devkit_path
#     # out_dir = args.out_dir if args.out_dir else devkit_path
#     # mmcv.mkdir_or_exist(out_dir)
#     devkit_path='/media/diskData/Datasets_PD/PD_ZX/VOC2014/'
#     out_dir='/home/rock/pd/'
#     years = []
#     if osp.isdir(osp.join(devkit_path, 'VOC2007')):
#         years.append('2007')
#     if osp.isdir(osp.join(devkit_path, 'VOC2012')):
#         years.append('2012')
#     if '2007' in years and '2012' in years:
#         years.append(['2007', '2012'])
#     if not years:
#         raise IOError('The devkit path {} contains neither "VOC2007" nor '
#                       '"VOC2012" subfolder'.format(devkit_path))
#     for year in years:
#         if year == '2007':
#             prefix = 'voc07'
#         elif year == '2012':
#             prefix = 'voc12'
#         elif year == ['2007', '2012']:
#             prefix = 'voc0712'
#         for split in ['train', 'val', 'trainval']:
#             dataset_name = prefix + '_' + split
#             print('processing {} ...'.format(dataset_name))
#             cvt_annotations(devkit_path, year, split,
#                             osp.join(out_dir, dataset_name + '.pkl'))
#         if not isinstance(year, list):
#             dataset_name = prefix + '_test'
#             print('processing {} ...'.format(dataset_name))
#             cvt_annotations(devkit_path, year, 'test',
#                             osp.join(out_dir, dataset_name + '.pkl'))
#     print('Done!')
#
#
# if __name__ == '__main__':
#     main()