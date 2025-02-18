import os.path

import pandas as pd
from numpy.random import randint
from torch.utils import data
import glob
import os

from dataloader.Record import VideoRecord
from dataloader.video_transform import *
import numpy as np
import re
from natsort import natsorted
import cv2
 

class SwinFlowDataset(data.Dataset):
    def __init__(self, list_file, num_segments, duration, transform, image_size):

        self.list_file = list_file
        self.duration = duration
        self.num_segments = num_segments
        self.transform = transform
        self.image_size = image_size
        self._parse_list()

    def _parse_list(self):
        #
        # Data Form: [video_id, num_frames, label]
        #
        tmp = [x.strip().split(' ') for x in open(self.list_file)]
        tmp = [item for item in tmp]
        self.video_list = [VideoRecord(item) for item in tmp]
        print(('video number:%d' % (len(self.video_list))))

    def _get_indices(self, record):
        """Split all videos into seg parts, then select the <mid item> in each part

        Args:
            record 

        Returns:
            List : frames' id 
        """        # 
        # Split all videos into seg parts, then select the <mid frame> in each part
        #
        if record.num_frames > self.num_segments + self.duration - 1:
            tick = (record.num_frames - self.duration + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.pad(np.array(list(range(record.num_frames))), (0, self.num_segments - record.num_frames), 'edge')
        return offsets

    def __getitem__(self, index):
        record = self.video_list[index]
        
        segment_indices = self._get_indices(record)
        return self.get(record, segment_indices)
    
    def get(self, record, indices):
        """return rgb images and flow images

        Args:
            record (_type_): _description_
            indices (_type_): _description_

        Returns:
            _type_: _description_
        """        
        video_frames_path = glob.glob(os.path.join(record.path, '*.bmp'))
        video_frames_path = natsorted(video_frames_path)
        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.duration):
                seg_imgs = [Image.open(os.path.join(video_frames_path[p])).convert('RGB')]
                images.extend(seg_imgs)
                if p < record.num_frames - 1:
                    p += 1
        flows = list()
        for i in range(len(indices)):
            if i < (len(images) - 1):


                # 读取两帧图像
                frame1 = cv2.imread(os.path.join(video_frames_path[indices[i]]), cv2.IMREAD_GRAYSCALE)
                frame2 = cv2.imread(os.path.join(video_frames_path[indices[i+1]]), cv2.IMREAD_GRAYSCALE)
                frame1 = cv2.resize(frame1,(self.image_size, self.image_size))
                frame2 = cv2.resize(frame2, (self.image_size, self.image_size))
                # 计算TV-L1光流
                flow = cv2.calcOpticalFlowFarneback(frame1, frame2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                # 获取水平和垂直光流
                u = flow[:, :, 0]
                v = flow[:, :, 1]

                # 计算光流场的梯度
                du_dx, du_dy = np.gradient(u)
                dv_dx, dv_dy = np.gradient(v)

                # 计算法向应变分量
                epsilon_xx = du_dx
                epsilon_yy = dv_dy

                # 计算剪切应变分量
                epsilon_xy = 0.5 * (dv_dx + du_dy)

                # 计算光学应变（标量）
                strain = np.sqrt(epsilon_xx**2 + epsilon_yy**2 + 0.5 * (epsilon_xy**2))

                # 将光流和光学应变组合成最终的输出
                output = np.stack((u, v, strain), axis=-1)
                flows.extend(output)

        images.pop(0)
        images = self.transform(images)
        images = torch.reshape(images, (-1, 3, self.image_size, self.image_size)) #[T, 3, 224, 224]
        flows = torch.tensor(flows)
        flows = torch.reshape(flows, (-1, 3, self.image_size, self.image_size))
        return images, record.label, flows

    def __len__(self):
        return len(self.video_list)


class SwinDataset(data.Dataset):
    def __init__(self, list_file, num_segments, duration, transform, image_size):

        self.list_file = list_file
        self.duration = duration
        self.num_segments = num_segments
        self.transform = transform
        self.image_size = image_size
        self._parse_list()

    def _parse_list(self):
        #
        # Data Form: [video_id, num_frames, label]
        #
        tmp = [x.strip().split(' ') for x in open(self.list_file)]
        tmp = [item for item in tmp]
        self.video_list = [VideoRecord(item) for item in tmp]
        print(('video number:%d' % (len(self.video_list))))

    def _get_indices(self, record):
        """Split all videos into seg parts, then select the <mid item> in each part

        Args:
            record 

        Returns:
            List : frames' id 
        """        # 
        # Split all videos into seg parts, then select the <mid frame> in each part
        #
        if record.num_frames > self.num_segments + self.duration - 1:
            tick = (record.num_frames - self.duration + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.pad(np.array(list(range(record.num_frames))), (0, self.num_segments - record.num_frames), 'edge')
        return offsets

    def __getitem__(self, index):
        record = self.video_list[index]
        
        segment_indices = self._get_indices(record)
        return self.get(record, segment_indices)
    
    def get(self, record, indices):
        """return rgb images and flow images

        Args:
            record (_type_): _description_
            indices (_type_): _description_

        Returns:
            _type_: _description_
        """        
        video_frames_path = glob.glob(os.path.join(record.path, '*.bmp'))
        video_frames_path = natsorted(video_frames_path)
        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.duration):
                seg_imgs = [Image.open(os.path.join(video_frames_path[p])).convert('RGB')]
                images.extend(seg_imgs)
                if p < record.num_frames - 1:
                    p += 1
        flows = list()
        
        # images.pop(0)
        
        # images_ = [self.transform(image) for image in images] #[T, 3,224,224]
        images_ = self.transform(images)
        images_ = torch.reshape(images_, (-1, 3, self.image_size, self.image_size)) #[T, 3, 224, 224]
        return images_, record.label, flows
    

    def __len__(self):
        return len(self.video_list)
    
import torchvision.transforms as transforms
def train_data_loader(list_file, num_segments, duration, image_size, args):
    train_transforms = torchvision.transforms.Compose([
        # GroupResize(image_size),
        Stack(),

        ToTorchFormatTensor(),
        GroupNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
            
    
    if args.flow == True:
        train_data = SwinFlowDataset(list_file=list_file,
                                num_segments=num_segments,
                                duration=duration,
                                transform=train_transforms,
                                image_size=image_size)
    else:
        train_data = SwinDataset(list_file=list_file,
                                num_segments=num_segments,
                                duration=duration,
                                transform=train_transforms,
                                image_size=image_size)

    return train_data


def test_data_loader(list_file, num_segments, duration, image_size, args):

    test_transform = torchvision.transforms.Compose([
        # GroupResize(image_size),
        Stack(),

        ToTorchFormatTensor(),
        GroupNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                                        ])
    if args.flow == True:
        test_data = SwinFlowDataset(list_file=list_file,
                                num_segments=num_segments,
                                duration=duration,
                                transform=test_transform,
                                image_size=image_size)
    else:
        test_data =  SwinDataset(list_file=list_file,
                                num_segments=num_segments,
                                duration=duration,
                                transform=test_transform,
                                image_size=image_size)
    return test_data

def val_data_loader(list_file, num_segments, duration, image_size, args):

    test_transform = torchvision.transforms.Compose(
        [        
        # GroupResize(image_size),
        Stack(),

        ToTorchFormatTensor(),
        GroupNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                                        ])
    if args.flow == True:
        test_data = SwinFlowDataset(list_file=list_file,
                                num_segments=num_segments,
                                duration=duration,
                                transform=test_transform,
                                image_size=image_size)
    else:
        test_data = SwinFlowDataset(list_file=list_file,    
                        num_segments=num_segments,
                        duration=duration,
                        transform=test_transform,
                        image_size=image_size)
    return test_data

# def train_data_loader_baseline(list_file, num_segments, duration, image_size, args):
    



#     if args.dataset == 'DAiSEE':
#             #TODO:这里是否有必要进行？
#         train_transforms = torchvision.transforms.Compose([GroupResize(image_size),
#                                                 Stack(),
#                                                 ToTorchFormatTensor()])

#     train_data = SwinDataset(list_file=list_file,
#                               num_segments=num_segments,
#                               duration=duration,
#                               mode='train',
#                               transform=train_transforms,
#                               image_size=image_size)

#     return train_data

if __name__ == '__main__':
    train_data = train_data_loader(list_file='annotation/EmotiW_Train_set.txt',num_segments=16,duration=1,image_size=224)

    print(train_data[0])