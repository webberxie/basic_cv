'''
@Author: Xie Yuhan
@Time: 2021.10.10
@use: to get format label txt
'''
import os
train_root = './train/'
test_root = './test/'
train_txt_name = 'train.txt'
test_txt_name = 'test.txt'
train_txt = open(train_txt_name,'w', encoding='utf-8')
test_txt = open(test_txt_name,'w',encoding='utf-8')

for root,dirs_name,files_name in os.walk(train_root):
    if len(dirs_name) == 0:  # 此时进入某一图像类别文件夹
        for img_name in files_name:
            type = img_name.split('.')[0]
            label = 0 if type=='cat' else 1
            result = img_name + ' ' + str(label) + '\n'
            train_txt.write(result)
train_txt.close()

for root,dirs_name,files_name in os.walk(test_root):
    if len(dirs_name) == 0:  # 此时进入某一图像类别文件夹
        for img_name in files_name:
            type = img_name.split('.')[0]
            label = 0 if type=='cat' else 1
            result = img_name + ' ' + str(label) + '\n'
            test_txt.write(result)
test_txt.close()