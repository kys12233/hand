from torch.utils.data import Dataset,DataLoader
# from config import *
import cv2
import os
import torch
from torchvision import transforms
import numpy as np


def one_hot(number):
    label_return = np.zeros(10)
    label_return[number] = 1
    return label_return
  
class MNIST_Dataset(Dataset):
    def __init__(self,label_path=None,transform=None) -> None:
        super(MNIST_Dataset).__init__()
        self.data = open(label_path).readlines()
        self.transform = transform #这个地方的transform是个形参，外面的实参进来

    def __len__(self):
        return len(self.data) #返回标签的长度
    
    def __getitem__(self, index):
        #获取图片的相对路径和标签,img_name和label_str是字符串
        img_name,label = self.data[index].replace("\n","").split(",")
        image = cv2.imread(img_name)

        # print(image.shape)
        
        image = cv2.resize(image,(224,224))

        image = self.transform(image)

        label = int(label)
        label = one_hot(label)

        # print(image.shape)
        # exit()

        return image,np.float32(label),img_name #返回图片，标签



if __name__ == '__main__':


    from torchvision import transforms
    train_transform = transforms.Compose([
        transforms.ToTensor() #将数据从hwc转化为chw，并放到0到1之间
    ])

    train_dataset = MNIST_Dataset(
        label_path=r'F:\Vscode_python_programs\HandwrittenNumeralRecognition\data_to\train_label.txt',
        transform=train_transform,
    )
    for i in train_dataset:
        print(i[0].shape,i[1])
        exit()