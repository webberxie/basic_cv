import cv2
import numpy as np
from math import *

class enhance:
    def __init__(self,img):
        self.image = img

    # 旋转
    def rotate(self, image, degree, output_size):
        height, width = image.shape[:2]
        heightNew = int(width * abs(np.sin(radians(degree))) + height * abs(np.cos(radians(degree))))
        widthNew = int(height * abs(np.sin(radians(degree))) + width * abs(np.cos(radians(degree))))

        matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)  # 构造旋转矩阵，（旋转中心，角度，缩放比例）

        matRotation[0, 2] += (widthNew - width) / 2
        matRotation[1, 2] += (heightNew - height) / 2
        img_rota = cv2.warpAffine(image, matRotation, (widthNew, heightNew),
                                  borderValue=(0, 0, 0))  # 进行仿射变换，（输入图像，输出图像，输出尺寸，边界取值）
        rows, cols = img_rota.shape  # 旋转后图像的行，列，通道

        # print('img_src.shape:', img_src.shape)
        max_len = max(rows, cols)
        img_bg = np.zeros((max_len, max_len, 1), np.uint8)
        img_bg.fill(0)  # 填充黑色
        # padding至正方形
        if rows > cols:
            len_padding = int((max_len - cols) / 2)
            if (max_len - 2 * len_padding) - cols > 0:
                img_bg[:, len_padding: -len_padding - 1, :] = img_rota
            elif (max_len - 2 * len_padding) - cols < 0:
                img_bg[:, len_padding: -len_padding + 1, :] = img_rota
            else:
                img_bg[:, len_padding: -len_padding, :] = img_rota

        elif rows < cols:
            len_padding = int((max_len - rows) / 2)
            if (max_len - 2 * len_padding) - rows > 0:
                img_bg[len_padding: -len_padding - 1, :, :] = img_rota
            elif (max_len - 2 * len_padding) - rows < 0:
                img_bg[len_padding: -len_padding + 1, :, :] = img_rota
            else:
                img_bg[len_padding: -len_padding, :, :] = img_rota
        else:
            img_bg = img_rota
        # 将图像缩放至输出指定大小
        img_bg = cv2.resize(img_bg,(output_size,output_size))
        return img_bg

    # 翻转
    def flip(self,image,fliptype):
        '''

        :param image: 输入图像
        :param fliptype: 1，-1，0（共三种翻转模式）
        :return:
        '''
        img_flip = cv2.flip(image,fliptype,dst=None)
        return img_flip
