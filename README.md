# DCGAN（Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks）
pytorch，mnist，CelebA，python,这是DCGAN实现
## 准备包涵下面函数的库  
import os, time, sys  
import pickle  
import imageio  
import torch  
import torch.optim as optim  
from torch.autograd import Variable  
from torchvision import datasets, transforms  
from torch.utils.data import Dataset  
from PIL import Image  
import torch.nn as nn  
import torch.nn.functional as F  
## 使用数据集，其中mnist数据集代码自动下载，celeba数据集要自己下载，用那个jpg的图像包然后解压出来，将该地址赋值给main.py的变量celea_data  
### 运行使用 python3.x平台
python main.py
