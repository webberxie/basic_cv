'''
@author:Xie Yuhan
@time:2021.10.10
@use: training code for classification
'''
import torch
import numpy as np
from torch import nn
from torch import optim
import torchvision.models as model_set
from torch.utils.data import DataLoader

import dataset
import utils
import os
import models

# 获取参数配置
config = utils.parse_arg()

# 获取模型
model = model_set.mobilenet_v2(pretrained=True,progress=True)
# 修改分类类别
model.classifier = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(model.last_channel, config.MODEL.NUM_CLASSES),
    nn.LogSoftmax(dim=1),
)

# get device
if torch.cuda.is_available():
    device = torch.device("cuda:{}".format(config.GPUID))
else:
    device = torch.device("cpu:0")

model = model.to(device)
# 展示模型信息
utils.model_info(model)

# 设置训练集和测试集
train_dataset = dataset.my_dataset(config=config,is_train=True)
train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU,
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

test_dataset = dataset.my_dataset(config=config,is_train=False)
test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU,
        shuffle=config.TEST.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

# 设置优化器与损失函数
#optimizer = optim.Adam(model.parameters(),lr=0.001)
optimizer = utils.get_optimizer(config, model)

criterion = nn.NLLLoss()

if __name__ == '__main__':
    for epoch in range(config.TRAIN.END_EPOCH):
        train_loss = 0.0
        train_acc = 0.0
        for i, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            predict = model(imgs)
            loss = criterion(predict,labels)

            # 清零过往梯度，反向计算梯度，逐级传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计准确率与损失
            _, predicted = torch.max(predict.data, 1)
            # torch.max返回输出结果中，按dim=1行排列的每一行最大数据及他的索引，丢弃数据，保留索引
            train_acc += (predicted == labels).sum().item()
            train_loss += loss.item()

            if i % 10 == 0:
                # 阶段性打印训练结果
                print('EPOCHS : {}/{}'.format(epoch + 1, config.TRAIN.END_EPOCH),
                      'Loss : {:.4f}'.format(train_loss / (i+1)),
                      'Acc : {:.4f}'.format(train_acc / ((i+1)*config.TRAIN.BATCH_SIZE_PER_GPU)))

        print('EPOCHS : {}/{}'.format(epoch + 1, config.TRAIN.END_EPOCH),
              'Loss : {:.4f}'.format(train_loss * config.TRAIN.BATCH_SIZE_PER_GPU / len(train_dataset)),
              'Acc : {:.4f}'.format(train_acc / len(train_dataset)))
        utils.accuracy_test(model,test_loader)
        torch.save(model, os.path.join(config.OUTPUT_DIR, "checkpoint_{}_acc_{:.4f}.pth".format(epoch, train_acc / len(train_dataset))))



