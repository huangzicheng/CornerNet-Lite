
# -*- coding: utf-8 -*-
# @Time    : 2019/8/27 10：48
# @Author  :Rock
# @File    : voc.py
# just for object detection

import os
import json
import numpy as np

from .detection import DETECTION
from ..paths import get_file_path


import xml.etree.ElementTree as ET
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
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,12, 13,
            14, 15, 16, 17, 18, 19, 20, 
        ]

        self._voc_cls_names = [
                      'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
                       'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
                       'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
                       'tvmonitor',

        ]

        self._cls2voc  = {ind + 1: voc_id for ind, voc_id in enumerate(self._voc_cls_ids)}
        self._voc2cls  = {voc_id: cls_id for cls_id, voc_id in self._cls2voc.items()}
        self._voc2name = {cls_id: cls_name for cls_id, cls_name in zip(self._voc_cls_ids, self._voc_cls_names)}
        self._name2voc = {cls_name: cls_id for cls_id, cls_name in self._voc2name.items()}

        if split is not None:
            voc_dir = os.path.join('/home/rock/CornerNet-Lite-master/data/', "VOC2012")

            self._data_dir  = os.path.join(voc_dir, 'JPEGImages')
          
            self.xml_path = os.path.join(voc_dir, "Annotations")

            self._detections, self._eval_ids = self._load_voc_annos()
            self._image_ids = list(self._detections.keys())
            self._db_inds   = np.arange(len(self._image_ids))


    def _load_voc_annos(self):
        eval_ids = {}
        detections = {}
        i = 0
        xml_path='/home/rock/CornerNet-Lite-master/data/VOC2012/Annotations'
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
                            bndbox[option.tag] = int(float(option.text))

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
                        category = self._name2voc[object_name]
                        bbox.append(category)
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


if __name__ == '__main__':

    detections, eval_ids = _load_voc_annos(xml_path)
    print(detections, eval_ids)





















































