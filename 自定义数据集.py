import os
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))])
# 自定义数据集时必须继承这个类torch.utils.data.Dataset，抽象类
class dataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
         imgs = os.listdir(root)
         # 这里self.imgs是一个图片list
         self.imgs = [os.path.join(root,img) for img in imgs]
         print(self.imgs)
         self.transforms = transforms
    def __getitem__(self, index):
        img_path = self.imgs[index]
        label = 0 if 'fire' in img_path.split('\\')[-1].split('_')[0] else 1
        data = Image.open(img_path)
        if self.transforms:
            data = self.transforms(data)
        return data, label
    def __len__(self):
        return len(self.imgs)
dataset1 = dataset(r'C:\Users\aaaac\Desktop\te\1', transforms=transform)
# img_path = r'C:\Users\aaaac\Desktop\te\1\2.jpg'
# print(img_path.split('\\')[-1].split('_')[0])
#img, label =dataset[0]
# for img, label in dataset:
#      print(img.size(),label)
# 定义数据加载时无需继承父类，只需要直接调用torch.utils.data.DataLoader类
dataset = torch.utils.data.DataLoader(dataset=dataset1,batch_size=1,shuffle=True,num_workers=0)
# for index,data in enumerate(dataset,0):
#     img = data[0]
#     labels = data[1]
#     print(img.size())
#     print(labels)
dataiter = iter(dataset)
imgs, labels = next(dataiter)
print(imgs.size())
