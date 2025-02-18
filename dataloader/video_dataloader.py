import os.path

import pandas as pd
from numpy.random import randint
from torch.utils import data
import glob
import os


from dataloader.video_transform import *
import numpy as np
import re
from natsort import natsorted
 
#EAR_l = (||y_37 - y_41|| + ||y_38 - y_40||) / 2 * ||x_36 - x_39||
#EAR_r = (||y_43 - y_47|| + ||y_44 - y_46||) / 2 * ||x_42 - x_45||
#EAR = (EAR_l + EAR_r) /2
#pose_Tx,pose_Ty

# features = [
#     "y_37", 'y_41', 'y_38', 'y_40', 'x_36', 'x_39',
#     'y_43', 'y_47', 'y_44', 'y_46', 'x_42', 'x_45',
#     'pose_Tx', 'pose_Ty',
#     "AU01_r", "AU02_r", "AU04_r",
#     "AU05_r", "AU06_r", "AU07_r",
#     "AU09_r", "AU10_r", "AU12_r",
#     "AU14_r", "AU15_r", "AU17_r",
#     "AU20_r", "AU23_r", "AU25_r",
#     "AU26_r", "AU45_r", "AU01_c",
#     "AU02_c", "AU04_c", "AU05_c",
#     "AU06_c", "AU07_c", "AU09_c",
#     "AU10_c", "AU12_c", "AU14_c",
#     "AU15_c", "AU17_c", "AU20_c",
#     "AU23_c", "AU25_c", "AU26_c",
#     "AU28_c", "AU45_c",
#     "gaze_0_x", "gaze_0_y", "gaze_0_z",
#     "gaze_1_x", "gaze_1_y", "gaze_1_z"
# ]
# eye_features = [
#     "y_37", 'y_41', 'y_38', 'y_40', 'x_36', 'x_39',
# 'y_43', 'y_47', 'y_44', 'y_46', 'x_42', 'x_45',
# ]


'''EmotiW'''
features = [
"gaze_0_x", "gaze_0_y", "gaze_0_z",
"gaze_1_x", "gaze_1_y", "gaze_1_z",
"gaze_angle_x", "gaze_angle_y",
"pose_Tx", "pose_Ty", "pose_Tz",
"pose_Rx", "pose_Ry", "pose_Rz",
"AU01_r", "AU02_r", "AU04_r",
"AU05_r", "AU06_r", "AU07_r",
"AU09_r", "AU10_r", "AU12_r",
"AU14_r", "AU15_r", "AU17_r",
"AU20_r", "AU23_r", "AU25_r",
"AU26_r", "AU45_r", "AU01_c",
"AU02_c", "AU04_c", "AU05_c",
"AU06_c", "AU07_c", "AU09_c",
"AU10_c", "AU12_c", "AU14_c",
"AU15_c", "AU17_c", "AU20_c",
"AU23_c", "AU25_c", "AU26_c",
"AU28_c", "AU45_c",
"y_37", 'y_41', 'y_38', 'y_40', 'x_36', 'x_39',
'y_43', 'y_47', 'y_44', 'y_46', 'x_42', 'x_45',
]

target_features = [
"gaze_0_x", "gaze_0_y", "gaze_0_z",
"gaze_1_x", "gaze_1_y", "gaze_1_z",
# "pose_Tx", "pose_Ty", "pose_Tz",
# "pose_Rx", "pose_Ry", "pose_Rz",
"gaze_angle_x", "gaze_angle_y",

"AU01_r", "AU02_r", "AU04_r",
"AU05_r", "AU06_r", "AU07_r",
"AU09_r", "AU10_r", "AU12_r",
"AU14_r", "AU15_r", "AU17_r",
"AU20_r", "AU23_r", "AU25_r",
"AU26_r", "AU45_r", "AU01_c",
"AU02_c", "AU04_c", "AU05_c",
"AU06_c", "AU07_c", "AU09_c",
"AU10_c", "AU12_c", "AU14_c",
"AU15_c", "AU17_c", "AU20_c",
"AU23_c", "AU25_c", "AU26_c",
"AU28_c", "AU45_c", 'EAR',
"y_37", 'y_41', 'y_38', 'y_40', 'x_36', 'x_39',
'y_43', 'y_47', 'y_44', 'y_46', 'x_42', 'x_45',
]

# target_features = [
# "gaze_0_x", "gaze_0_y", "gaze_0_z",
# "gaze_1_x", "gaze_1_y", "gaze_1_z",
# # "pose_Tx", "pose_Ty", "pose_Tz",
# # "pose_Rx", "pose_Ry", "pose_Rz",
# "gaze_angle_x", "gaze_angle_y",

# "AU01_r", "AU02_r", "AU04_r",
# "AU05_r", "AU06_r", "AU07_r",
# "AU09_r", "AU10_r", "AU12_r",
# "AU14_r", "AU15_r", "AU17_r",
# "AU20_r", "AU23_r", "AU25_r",
# "AU26_r", "AU45_r", "AU01_c",
# "AU02_c", "AU04_c", "AU05_c",
# "AU06_c", "AU07_c", "AU09_c",
# "AU10_c", "AU12_c", "AU14_c",
# "AU15_c", "AU17_c", "AU20_c",
# "AU23_c", "AU25_c", "AU26_c",
# "AU28_c", "AU45_c", 
# "y_37", 'y_41', 'y_38', 'y_40', 'x_36', 'x_39',
# 'y_43', 'y_47', 'y_44', 'y_46', 'x_42', 'x_45',
# ]




# target_features = [
# # "gaze_0_x", "gaze_0_y", "gaze_0_z",
# # "gaze_1_x", "gaze_1_y", "gaze_1_z",
# # "pose_Tx", "pose_Ty", "pose_Tz",
# # "pose_Rx", "pose_Ry", "pose_Rz",
# "gaze_angle_x", "gaze_angle_y",

# "AU01_r", "AU02_r", "AU04_r",
# "AU05_r", "AU06_r", "AU07_r",
# "AU09_r", "AU10_r", "AU12_r",
# "AU14_r", "AU15_r", "AU17_r",
# "AU20_r", "AU23_r", "AU25_r",
# "AU26_r", "AU45_r", "AU01_c",
# "AU02_c", "AU04_c", "AU05_c",
# "AU06_c", "AU07_c", "AU09_c",
# "AU10_c", "AU12_c", "AU14_c",
# "AU15_c", "AU17_c", "AU20_c",
# "AU23_c", "AU25_c", "AU26_c",
# "AU28_c", "AU45_c", 
# ]


#1.针对pose的消融实验

'''EmotiW'''



#59.48
# features = [

# "gaze_0_x", "gaze_0_y", "gaze_0_z",
# "gaze_1_x", "gaze_1_y", "gaze_1_z",
# "gaze_angle_x", "gaze_angle_y",
# "pose_Tx", "pose_Ty", "pose_Tz",
# "pose_Rx", "pose_Ry", "pose_Rz",
# "AU01_r", "AU02_r", "AU04_r",
# "AU05_r", "AU06_r", "AU07_r",
# "AU09_r", "AU10_r", "AU12_r",
# "AU14_r", "AU15_r", "AU17_r",
# "AU20_r", "AU23_r", "AU25_r",
# "AU26_r", "AU45_r", "AU01_c",
# "AU02_c", "AU04_c", "AU05_c",
# "AU06_c", "AU07_c", "AU09_c",
# "AU10_c", "AU12_c", "AU14_c",
# "AU15_c", "AU17_c", "AU20_c",
# "AU23_c", "AU25_c", "AU26_c",
# "AU28_c", "AU45_c",

# # ]
#59.18
# features = [
# "gaze_angle_x", "gaze_angle_y",
# "pose_Tx", "pose_Ty", "pose_Tz",
# "pose_Rx", "pose_Ry", "pose_Rz",
# "AU01_r", "AU02_r", "AU04_r",
# "AU05_r", "AU06_r", "AU07_r",
# "AU09_r", "AU10_r", "AU12_r",
# "AU14_r", "AU15_r", "AU17_r",
# "AU20_r", "AU23_r", "AU25_r",
# "AU26_r", "AU45_r", "AU01_c",
# "AU02_c", "AU04_c", "AU05_c",
# "AU06_c", "AU07_c", "AU09_c",
# "AU10_c", "AU12_c", "AU14_c",
# "AU15_c", "AU17_c", "AU20_c",
# "AU23_c", "AU25_c", "AU26_c",
# "AU28_c", "AU45_c",
# "gaze_0_x", "gaze_0_y", "gaze_0_z",
# "gaze_1_x", "gaze_1_y", "gaze_1_z",
# ]

# 58.96
# features = [

# "AU01_r", "AU02_r", "AU04_r",
# "AU05_r", "AU06_r", "AU07_r",
# "AU09_r", "AU10_r", "AU12_r",
# "AU14_r", "AU15_r", "AU17_r",
# "AU20_r", "AU23_r", "AU25_r",
# "AU26_r", "AU45_r", "AU01_c",
# "AU02_c", "AU04_c", "AU05_c",
# "AU06_c", "AU07_c", "AU09_c",
# "AU10_c", "AU12_c", "AU14_c",
# "AU15_c", "AU17_c", "AU20_c",
# "AU23_c", "AU25_c", "AU26_c",
# "AU28_c", "AU45_c",
# "gaze_0_x", "gaze_0_y", "gaze_0_z",
# "gaze_1_x", "gaze_1_y", "gaze_1_z",
# "pose_Tx", "pose_Ty", "pose_Tz",
# "pose_Rx", "pose_Ry", "pose_Rz",
# "gaze_angle_x", "gaze_angle_y",

# ]


# features = [
# "pose_Tx", "pose_Ty", "pose_Tz",
# "pose_Rx", "pose_Ry", "pose_Rz",
# "AU01_r", "AU02_r", "AU04_r",
# "AU05_r", "AU06_r", "AU07_r",
# "AU09_r", "AU10_r", "AU12_r",
# "AU14_r", "AU15_r", "AU17_r",
# "AU20_r", "AU23_r", "AU25_r",
# "AU26_r", "AU45_r", "AU01_c",
# "AU02_c", "AU04_c", "AU05_c",
# "AU06_c", "AU07_c", "AU09_c",
# "AU10_c", "AU12_c", "AU14_c",
# "AU15_c", "AU17_c", "AU20_c",
# "AU23_c", "AU25_c", "AU26_c",
# "AU28_c", "AU45_c",
# "gaze_0_x", "gaze_0_y", "gaze_0_z",
# "gaze_1_x", "gaze_1_y", "gaze_1_z",

# "gaze_angle_x", "gaze_angle_y",

# ]

# features = [
# "pose_Tx", "pose_Ty", "pose_Tz",
# "pose_Rx", "pose_Ry", "pose_Rz",
# "AU01_r", "AU02_r", "AU04_r",
# "AU05_r", "AU06_r", "AU07_r",
# "AU09_r", "AU10_r", "AU12_r",
# "AU14_r", "AU15_r", "AU17_r",
# "AU20_r", "AU23_r", "AU25_r",
# "AU26_r", "AU45_r", "AU01_c",
# "AU02_c", "AU04_c", "AU05_c",
# "AU06_c", "AU07_c", "AU09_c",
# "AU10_c", "AU12_c", "AU14_c",
# "AU15_c", "AU17_c", "AU20_c",
# "AU23_c", "AU25_c", "AU26_c",
# "AU28_c", "AU45_c",
# "gaze_0_x", "gaze_0_y", "gaze_0_z",
# "gaze_1_x", "gaze_1_y", "gaze_1_z",

# "gaze_angle_x", "gaze_angle_y",
# ]

# features = [
# "gaze_angle_x", "gaze_angle_y",
# "pose_Tx", "pose_Ty", "pose_Tz",
# "pose_Rx", "pose_Ry", "pose_Rz",
# "AU01_r", "AU02_r", "AU04_r",
# "AU05_r", "AU06_r", "AU07_r",
# "AU09_r", "AU10_r", "AU12_r",
# "AU14_r", "AU15_r", "AU17_r",
# "AU20_r", "AU23_r", "AU25_r",
# "AU26_r", "AU45_r", "AU01_c",
# "AU02_c", "AU04_c", "AU05_c",
# "AU06_c", "AU07_c", "AU09_c",
# "AU10_c", "AU12_c", "AU14_c",
# "AU15_c", "AU17_c", "AU20_c",
# "AU23_c", "AU25_c", "AU26_c",
# "AU28_c", "AU45_c",
# "gaze_0_x", "gaze_0_y", "gaze_0_z",
# "gaze_1_x", "gaze_1_y", "gaze_1_z",
# ]



class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])

    @property
    def csv_file(self):
        return self._data[3]

class VideoDataset(data.Dataset):
    def __init__(self, list_file, num_segments, duration, mode, transform, image_size):

        self.list_file = list_file
        self.duration = duration
        self.num_segments = num_segments
        self.transform = transform
        self.image_size = image_size
        self.mode = mode
        self._parse_list()

        pass

    def _parse_list(self):
        #
        # Data Form: [video_id, num_frames, class_idx]
        #
        tmp = [x.strip().split(' ') for x in open(self.list_file)]
        tmp = [item for item in tmp]
        self.video_list = [VideoRecord(item) for item in tmp]
        print(('video number:%d' % (len(self.video_list))))

    def _get_train_indices(self, record):
        # 
        # Split all videos into seg parts, then select frame in each part randomly
        #
        average_duration = (record.num_frames - self.duration + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments)
        elif record.num_frames > self.num_segments:
            offsets = np.sort(randint(record.num_frames - self.duration + 1, size=self.num_segments))
        else:
            offsets = np.pad(np.array(list(range(record.num_frames))), (0, self.num_segments - record.num_frames), 'edge')
        return offsets

    def _get_test_indices(self, record):
        # 
        # Split all videos into seg parts, then select frame in the mid of each part
        #
        if record.num_frames > self.num_segments + self.duration - 1:
            tick = (record.num_frames - self.duration + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.pad(np.array(list(range(record.num_frames))), (0, self.num_segments - record.num_frames), 'edge')
        return offsets

    def __getitem__(self, index):
        record = self.video_list[index]
        if self.mode == 'train':
            segment_indices = self._get_train_indices(record)
        elif self.mode == 'test':
            segment_indices = self._get_test_indices(record)

        return self.get(record, segment_indices)
    #TODO: 返回的数据不仅要包含图像，标签，还要包含GAP Features
    def get(self, record, indices):
        video_frames_path = glob.glob(os.path.join(record.path, '*.bmp'))
        video_frames_path = natsorted(video_frames_path)
        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.duration):
                # print(p)
                # print(len(video_frames_path))
                seg_imgs = [Image.open(os.path.join(video_frames_path[p])).convert('RGB')]
                # print(os.path.join(video_frames_path[p]))

                images.extend(seg_imgs)
                if p < record.num_frames - 1:
                    p += 1

        csv_file = record.csv_file
        df = pd.read_csv(csv_file)
        
         
      
        df = df[features]
        # print(df)
        df['EAR'] = ((abs(df['y_37'] - df['y_41']) + abs(df['y_38'] - df['y_40'])) / (2 * abs(df['x_36'] - df['x_39'])) + (abs(df['y_43'] - df['y_47']) + abs(df['y_44'] - df['y_46']) ) / (2 * abs(df['x_42'] - df['x_45']))) / 2
        

        df = df[target_features]

        # df = df[['pose_Tx', 'pose_Ty','EAR']]
        
        # df = df[target_features] 
        df['EAR'] = df['EAR'].replace([np.inf, -np.inf], 0)




       
        df = df.iloc[indices]
        #[T, 49] 维度为49
        gap = torch.tensor(df.to_numpy(), dtype=torch.float32) #[T, 49]
        feature_min , _ = gap.min(dim=1, keepdim=True)
        feature_max , _ = gap.max(dim=1, keepdim=True)
        gap = (gap - feature_min) / (feature_max - feature_min)
        # print(gap)

        images = self.transform(images)
        images = torch.reshape(images, (-1, 3, self.image_size, self.image_size)) #[T, 3, 224, 224]
        # print(images)
        return images, record.label, gap

    def __len__(self):
        return len(self.video_list)


def train_data_loader(list_file, num_segments, duration, image_size, args):
   
    train_transforms = torchvision.transforms.Compose([
        ColorJitter(brightness=0.5),
        GroupRandomSizedCrop(image_size),
        GroupRandomHorizontalFlip(),
        Stack(),
        ToTorchFormatTensor()])
        
    
    
    train_data = VideoDataset(list_file=list_file,
                              num_segments=num_segments,
                              duration=duration,
                              mode='train',
                              transform=train_transforms,
                              image_size=image_size)

    return train_data


def test_data_loader(list_file, num_segments, duration, image_size):
    
    test_transform = torchvision.transforms.Compose([GroupResize(image_size),
                                                     Stack(),
                                                     ToTorchFormatTensor()])
    
    test_data = VideoDataset(list_file=list_file,
                             num_segments=num_segments,
                             duration=duration,
                             mode='test',
                             transform=test_transform,
                             image_size=image_size)
    return test_data


def train_data_loader_baseline(list_file, num_segments, duration, image_size, args):
    



    if args.dataset == 'DAiSEE':
            #TODO:这里是否有必要进行？
        train_transforms = torchvision.transforms.Compose([GroupResize(image_size),
                                                Stack(),
                                                ToTorchFormatTensor()])

    train_data = VideoDataset(list_file=list_file,
                              num_segments=num_segments,
                              duration=duration,
                              mode='train',
                              transform=train_transforms,
                              image_size=image_size)

    return train_data