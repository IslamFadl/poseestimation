import cv2
import re
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import albumentations as A


class ArtifitialImages(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None, train_transform=None):
        regexp = re.compile(r'frame_(\d+)_')
        self.frames = list(set([int(regexp.search(str(item)).group(1)) for item in os.listdir(img_dir)]))  # group: Return the string matched by the RE
        self.file_names = os.listdir(img_dir)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.train_transform = train_transform

    def __len__(self):
        return len(self.frames)  # number of distinct frames

    def __getitem__(self, idx):
        frame_number = self.frames[idx]
        file_names = [item for item in self.file_names if 'frame_' + str(frame_number) + '_' in (item)]

        for i, f in enumerate(file_names):
            # read image file
            img_path = os.path.join(self.img_dir, f)
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            convert_tensor = transforms.ToTensor()
            t = convert_tensor(image)

        angle = re.findall(r'\d+', f)
        angle = float(angle[1])


        #z_reg = re.compile(r'angle_(-?\d+).png$')
        #angle = float(z_reg.search(str(file_names[0])).group(1))

        if self.transform is not None:
            image = self.transform(image=image)['image']
        if self.target_transform is not None:
            angle = self.target_transform(angle)
        return t, angle



class DrillImages(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None, train_transform=None):
        regexp = re.compile(r'frame_(\d+)_')
        self.frames = list(set([int(regexp.search(str(item)).group(1)) for item in os.listdir(img_dir)]))
        self.file_names = os.listdir(img_dir)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.train_transform = train_transform

    def __len__(self):
        return len(self.frames)  # number of distinct frames

    def __getitem__(self, idx):
        frame_number = self.frames[idx]
        file_names = [item for item in self.file_names if 'frame_' + str(frame_number) + '_' in (item)]

        for i, f in enumerate(file_names):
            # read image file
            img_path = os.path.join(self.img_dir, f)
            image = cv2.imread(img_path)

            # border_mode: 4 By default
            # border_mode: 0: BORDER_CONSTANT, 1: BORDER_REPLICATE, 2 : BORDER_REFLECT, 3: BORDER_WRAP , 4: BORDER_REFLECT_101

            transform = A.Compose([
                A.PadIfNeeded(min_height=640, min_width=640, border_mode=1, value=[0, 0, 0], p=1),
                A.RandomBrightnessContrast(p=0.1),
                A.HorizontalFlip(p=0.1),
                A.MotionBlur(p=0.1),
                A.OpticalDistortion(p=0.1),
                A.GaussNoise(p=0.1),
                A.GridDropout(p=0.1),
                A.GaussNoise(var_limit=350.0, p=1.0),
                A.ChannelShuffle(p=0.1),
                A.Resize(224, 224)
            ])

            image = transform(image=image)["image"]

            # convert image to grayscle
            image = cv2.cvtColor(image,
                                 cv2.COLOR_BGR2GRAY)  # >>print(type(drill_dataset[0][0]))  > torch.Size([3, 640, 640]) instead of torch.Size([9, 640, 640])

            convert_tensor = transforms.ToTensor()
            # convert_tensor = ToTensorV2()    # with Albumentations library

            t = convert_tensor(image)
            if i == 0:
                t1 = t
            else:
                t1 = torch.cat([t1, t], dim=0)

        #z_reg = re.compile(r'Z_(-?\d+).png$')
        #angle = float(z_reg.search(str(file_names[0])).group(1))
        angle = re.findall(r'\d+', f)
        angle = float(angle[1])

        if self.transform is not None:
            image = self.transform(image=image)['image']
        if self.target_transform is not None:
            angle = self.target_transform(angle)
        return t1, angle