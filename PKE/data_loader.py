import os
import random
from random import shuffle
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image

class ImageFolder(data.Dataset):
    def __init__(self, root,image_size=224,mode='train',augmentation_prob=0.4):
        self.root = root
        self.fingervein=os.path.join(self.root,"fingervein")
        self.fingerprint=os.path.join(self.root,"fingerprint")
        self.palmprint=os.path.join(self.root,"palmprint")

        self.GT_paths = root+'_GT/'
        self.fingervein_GT=os.path.join(self.GT_paths,"fingervein")
        self.fingerprint_GT=os.path.join(self.GT_paths,"fingerprint")
        self.palmprint_GT=os.path.join(self.GT_paths,"palmprint")

        self.fingervein_image_paths=[]
        self.fingerprint_image_paths=[]
        self.palmprint_image_paths=[]
        fingervein_dir_paths = list(map(lambda x: os.path.join(self.fingervein, x), os.listdir(self.fingervein))) 
        for dir_path in fingervein_dir_paths:
          if ".DS_Store" in dir_path:
            continue
          self.fingervein_image_paths+=[os.path.join(dir_path,l) for l in os.listdir(dir_path)]
        fingerprint_dir_paths = list(map(lambda x: os.path.join(self.fingerprint, x), os.listdir(self.fingerprint))) 
        for dir_path in fingerprint_dir_paths:
          if ".DS_Store" in dir_path:
            continue
          self.fingerprint_image_paths+=[os.path.join(dir_path,l) for l in os.listdir(dir_path)]
        palmprint_dir_paths = list(map(lambda x: os.path.join(self.palmprint, x), os.listdir(self.palmprint))) 
        for dir_path in palmprint_dir_paths:
          if ".DS_Store" in dir_path:
            continue
          self.palmprint_image_paths+=[os.path.join(dir_path,l) for l in os.listdir(dir_path)]
      
        self.image_size = image_size
        self.mode = mode
        self.RotationDegree = [0,0,180,180]
        self.augmentation_prob = augmentation_prob
        print("image count in {}".format(self.mode))
        print("fingervein %d"%len(self.fingervein_image_paths))
        print("fingerprint %d"%len(self.fingerprint_image_paths))
        print("palmprint %d"%len(self.palmprint_image_paths))

    def __getitem__(self, index):
        fingervein_image_path = self.fingervein_image_paths[index]
        fingerprint_image_path = self.fingerprint_image_paths[index]
        palmprint_image_path = self.palmprint_image_paths[index]
        
        dirname=fingervein_image_path.split('/')[-2]
        filename = fingervein_image_path.split('/')[-1]

        fingervein_GT_path = os.path.join(self.fingervein_GT ,dirname,filename)
        fingerprint_GT_path = os.path.join(self.fingerprint_GT ,fingerprint_image_path.split('/')[-2],fingerprint_image_path.split('/')[-1])
        palmprint_GT_path = os.path.join(self.palmprint_GT ,dirname,filename)

        fingervein_image = Image.open(fingervein_image_path).convert("L")
        fingerprint_image = Image.open(fingerprint_image_path).convert("L")
        palmprint_image = Image.open(palmprint_image_path).convert("L")
        tosize=(256,512)
        
        image=Image.merge("RGB",(fingerprint_image.resize(tosize),fingervein_image.resize(tosize),palmprint_image.resize(tosize)))
        if not self.mode=="infer":
          fingervein_GT = Image.open(fingervein_GT_path).convert("L")
          fingerprint_GT = Image.open(fingerprint_GT_path).convert("L")
          palmprint_GT = Image.open(palmprint_GT_path).convert("L")
          GT=Image.merge("RGB",(fingerprint_GT.resize(tosize),fingervein_GT.resize(tosize),palmprint_GT.resize(tosize)))

        Transform = []

        Transform.append(T.ToTensor())
        Transform = T.Compose(Transform)
        
        image = Transform(image)
        if not self.mode=="infer":
          GT = Transform(GT)
        
        if not self.mode=="infer":
            return image, GT, dirname+"/"+filename
        else:
            return image, dirname+"/"+filename

    def __len__(self):
        return len(self.fingervein_image_paths)

def get_loader(image_path, image_size, batch_size, num_workers=2, mode='train',augmentation_prob=0.4):
    
    dataset = ImageFolder(root = image_path, image_size =image_size, mode=mode,augmentation_prob=augmentation_prob)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)
    return data_loader
