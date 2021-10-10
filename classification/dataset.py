'''
@author:Xie Yuhan
@time:2021.10.10
@use:datasets for training/testing data
'''

import torch.utils.data as data
import os
import numpy as np
import cv2


class my_dataset(data.Dataset):
    def __init__(self, config, is_train):
        self.type = 'train' if is_train else 'test'
        self.root = os.path.join(config.DATASET.ROOT, self.type)
        self.img_h = config.DATASET.IMG_H
        self.img_w = config.DATASET.IMG_W
        self.mean = config.DATASET.MEAN
        self.std = config.DATASET.STD

        txt_file = config.DATASET.TRAIN_TXT if is_train else config.DATASET.TEST_TXT
        # 读入标签
        self.labels = []
        with open(txt_file, 'r', encoding='utf-8') as file:
            contents = file.readlines()
            for c in contents:
                imgname = c.split(' ')[0]
                label = c.split(' ')[-1]
                # 图像名字，标签（key，value）
                self.labels.append({imgname: label})
        print("load {} images!".format(self.__len__()))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # 读取数据
        img_name = list(self.labels[idx].keys())[0]
        label = list(self.labels[idx].values())[0]
        img = cv2.imread(os.path.join(self.root, img_name))
        # 调整大小
        img = cv2.resize(img, (self.img_h, self.img_w))
        # 归一化
        img = img.astype(np.float32)
        img = (img / 255. - self.mean) / self.std
        # 调整格式：C,H,W
        img = img.transpose([2, 0, 1])
        label = int(label)

        return img, label
