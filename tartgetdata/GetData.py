import os
project_index = os.getcwd().find('AdversialExamples')
root = os.getcwd()[0:project_index] + 'AdversialExamples'
import sys
sys.path.append(root)
import torchvision as tv
from torchvision import transforms
from torch.utils.data import DataLoader
from config.Config import opt
import cv2
import numpy as np

"""
下载MNIST数据集并打包成dataloader
"""

def __gray2RGB(img):
    img = np.array(img)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

train_transform = transforms.Compose(
    [
        transforms.Lambda(lambda img: __gray2RGB(img)),  # 将灰度图转换成3通道
        transforms.ToTensor(),
    ]
)
test_transform = train_transform

abs_file = __file__
cd = abs_file[:abs_file.rfind("/")]
train_dataset = tv.datasets.MNIST(root=cd, train=True, transform=train_transform, download=True)
test_dataset = tv.datasets.MNIST(root=cd, train=False, transform=test_transform, download=True)


dataloaders = {}
dataset_sizes = {}


train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=opt.batch_size, num_workers=opt.num_workers)
test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=opt.batch_size, num_workers=opt.num_workers)


dataloaders['train'] = train_dataloader
dataloaders['val'] = test_dataloader


dataset_sizes['train'] = len(train_dataset)
dataset_sizes['val'] = len(test_dataset)


class_names = [str(x) for x in range(10)]


if __name__ == '__main__':
    print(train_dataset[0][0].numpy().shape)
    cv2.imshow("img", train_dataset[0][0].numpy().transpose(1,2,0))
    cv2.waitKey(0)