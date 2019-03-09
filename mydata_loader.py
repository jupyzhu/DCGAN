import os
from torchvision import datasets, transforms
import torch
from torch.utils.data import Dataset
from PIL import Image


def mnist_loader(img_size, batch_size):
    transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    mnist = datasets.MNIST('data', train=True, download=True, transform=transform)
    mnist_loader = torch.utils.data.DataLoader(mnist, batch_size=batch_size, shuffle=True)
    print('MNIST: number image of train: {} '.format(len(mnist.train_data)))
    return mnist_loader


class CelebA(Dataset):  # 继承了Dataset类
    def __init__(self, data_dir, transform=None):
        super(CelebA, self).__init__()
        self.path = data_dir
        self.transform = transform
        self.img_list = os.listdir(data_dir)

    def __len__(self):  # 返回数据集的大小
        return len(self.img_list)

    def __getitem__(self, idx):  # 实现数据集的下标索引，返回对应的图像和标记
        img_idx = self.img_list[idx]  # 索引DataFrame的第idx行,从0行开始,对应图片第一张
        img = Image.open(self.path+img_idx)
        if self.transform:
            img = self.transform(img)  # pre-process
        return img, idx


def celeba_loader(img_size, batch_size, data_dir='CelebA/'):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    img_list = os.listdir(data_dir)
    celeba=CelebA(data_dir, transform=transform)
    celeba_loader = torch.utils.data.DataLoader(celeba, batch_size=batch_size, shuffle=True)
    print(f"CelebA: number image of train: {len(img_list)} ")
    return celeba_loader