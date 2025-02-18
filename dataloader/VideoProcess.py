#读取一个视频，获取视频的帧
import time
from sys import stdout

import cv2
import os
import pandas as pd
from tqdm import tqdm

import subprocess
import debugpy
# try:
#     debugpy.listen(('localhost', 12323))
#     print('Waiting for debugger attach')
#     debugpy.wait_for_client()
# except Exception as e:
#     print(e)

    
base_output_folder = "data/DFEW/frames"
#修改成 "data/DAISEE/DataSet/Train"  or "Test" or "Validation"
video_base_folder = "data/DFEW/videos"
video_format = ".mp4"

'''
extract videos from a video
'''
def process_video(video_path, video_format=video_format):
    video_path = video_path + video_format
    cap = cv2.VideoCapture(os.path.join(video_base_folder, video_path))
    frame_count = 0
    output_folder = video_path.split('.')[0].zfill(5)

    output_folder = os.path.join(base_output_folder, output_folder)

    os.makedirs(output_folder, exist_ok=True)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        picture = os.path.join(output_folder,f"{frame_count}.jpg")
        cv2.imwrite(picture, frame)
        frame_count += 1
    cap.release()
    return frame_count

def process(set_csv, annotation_file):
    df = pd.read_csv(set_csv)

    annotations = []
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        video_name = str(row['video_name'])
        label = row['label']
        frame_cnt = process_video(video_name)
        video_name = base_output_folder + video_name
        annotation = video_name + " " +str(frame_cnt) + " " + str(label)
        annotations.append(annotation)

    with open(annotation_file, "w") as f:
        for ann in annotations:
            f.write(ann + "\n")

#
# train_csv = "data/DFEW/EmoLabel_DataSplit/train(single-labeled)/set_1.csv"
# test_csv = "data/DFEW/EmoLabel_DataSplit/test(single-labeled)/set_1.csv"
# train_annotation = "annotation/DFEW_train_set1.txt"
# test_annotation = "annotation/DFEW_test_set1.txt"
# process(train_csv, train_annotation)
# process(test_csv,test_annotation)



train_csv = "data/DAiSEE/Labels/TrainLabels.csv"
test_csv = "data/DAiSEE/Labels/TestLabels.csv"
valid_csv = "data/DAiSEE/Labels/ValidationLabels.csv"
all_csv = "data/DAiSEE/Labels/AllLabels.csv"

train_anno = "/annotation/DAiSEE_train.txt"
test_anno = "/annotation/DAiSEE_test.txt"
val_anno = "/annotation/DAiSEE_val.txt"


'''
0.处理原始数据集
'''
# 4599990171.avi
# 459999021.avi
#同一个数据集可能有多种形式的视频文件
formats = ['.avi', '.mp4']

txt_base = "data/DAiSEE/DataSet"

modes = [ 'Train',"Test", "Validation"]
def _video_to_mode():
    video2mode = {}

    for mode in modes:
        txt_path = os.path.join(txt_base, mode)
        txt_path = txt_path + ".txt"

        with open(txt_path, 'r') as f:
            datas = f.readlines()
            for data in datas:
                _data = data
                _data = _data.strip()
                tdata, format  = _data.split('.')
                obj = {}
                obj['mode'] = mode
                obj['format'] = format
                video2mode[tdata] = obj

    return video2mode
def process_DAISEE(set_csv, annotation_file, mode="all"):
    df = pd.read_csv(set_csv)
    video2mode = _video_to_mode()
    for index, row in tqdm(df.iterrows(), total=len(df), desc=f"Step:0 Processing {mode}"):
        video_name = str(row['ClipID']).split('.')[0]
        label = row['Engagement']  # [0, 1, 2, 3]
        prefix = video_name[:6]
        subfix = video_name[-4:]
        _mode = video2mode[video_name]['mode']
        _format = video2mode[video_name]['format']
        # 构建视频文件路径
        video_path = f"data/DAiSEE/DataSet/{_mode}/{prefix}/{video_name}/{video_name}.{_format}"

        #部分内容没有处理，特殊处理，用后注释
        _output_path = f"data/DAiSEE/{_mode}/{video_name}"
        if os.path.exists(_output_path):
            continue


        command = [r"openface.bat", video_path, _mode]
        process = subprocess.run(command, text=True)

        # 打印输出和错误
        print("输出:", process.stdout)
        print("错误:", process.stderr)
# process_DAISEE(all_csv, None, 'all')
# process_DAISEE(train_csv, None, 'Train')
# process_DAISEE(test_csv, None, 'Test')
# process_DAISEE(valid_csv, None, 'Validation')
#

'''
1.重命名文件夹
'''
import os
from tqdm import tqdm
import shutil


# for mode in modes:
#     path = f"data/DAiSEE/{mode}"
#     for folder in tqdm(os.listdir(path), desc=mode):
#         if folder.endswith("_aligned"):
#
#             new_name = folder[:-8]
#             try:
#                 os.rename(os.path.join(path, folder), os.path.join(path, new_name))
#             except FileExistsError:
#                 shutil.rmtree(os.path.join(path, folder))



'''
2.重命名图片文件
'''
# for mode in modes:
#     path = f"data/DAiSEE/{mode}"
#     # 列出所有文件夹
#     folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
#     for folder in tqdm(folders, total=len(folders), desc=mode):
#         for img in os.listdir(os.path.join(path,folder)):
#             if img.startswith("frame_det_00_") and img.endswith(".bmp"):
#                 number = int(img.split("_")[-1].split('.')[0])
#                 new_name = f"{number}.bmp"
#
#
#                 new_name = os.rename(os.path.join(path, folder, img), os.path.join(path, folder, new_name))
#                 # print(f'{os.path.join(path,folder,img)} {os.path.join(path,folder,new_name)}')


'''
3.1删除置信度为0的文件并且重新排序
'''

# modes = ['Train', 'Validation']
# for mode in modes:
#     path = f"data/EmotiW/{mode}_csvs"
#     _path = f'data/EmotiW/{mode}'
#     csv_files = [file for file in os.listdir(path) if file.endswith(".csv")]

#     for csv_file in tqdm(csv_files, total=len(csv_files), desc=mode):
#         df = pd.read_csv(os.path.join(path, csv_file))

#         # df['frame'] = df['frame'].astype(int)
#         bmp_files_path = os.path.join(_path, csv_file.split('.')[0])
#         bmp_files = os.listdir(bmp_files_path)
#         _bmp_files = [f for f in bmp_files if f.endswith(".bmp")]

#         for index, row in df.iterrows():
#             if row['confidence']==0:
#                 old_file_name = f"{int(row['frame'])}.bmp"
#                 old_path = os.path.join(bmp_files_path, old_file_name)
#                 if old_file_name in _bmp_files:
#                     os.remove(old_path)

#         # zero_confidence = df[df["confidence"]==0].reset_index(drop=True)
#         non_zero_confidence = df[df["confidence"] != 0].reset_index(drop=True)
#         new_file_names = []
#         for index, row in non_zero_confidence.iterrows():
#             old_file_name = f"{int(row['frame'])}.bmp"
#             new_file_name = f"{index + 1}.bmp"
#             if old_file_name in _bmp_files:
#                 new_file_names.append((old_file_name, new_file_name))


#         for old_file_name, new_file_name in new_file_names:
#             old_path = os.path.join(bmp_files_path, old_file_name)
#             new_path = os.path.join(bmp_files_path, new_file_name)
#             os.rename(old_path, new_path)


#         df = df[df['confidence']!=0].reset_index(drop=True)

#         df['frame'] = df.index + 1
#         df.to_csv(os.path.join(path, csv_file), index=False)

'''
3.2重命名图片文件，形如0001.bmp ...
'''
# img_format = '.bmp'
# for mode in modes:
#     path = f"data/EmotiW/{mode}"
#     # 列出所有文件夹
#     folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
#     for folder in tqdm(folders, total=len(folders), desc=mode):
#         folder_path = os.path.join(path, folder)
#         img_files = os.listdir(folder_path)

#         img_files = natsorted(img_files)
#         for index, img in enumerate(img_files, start=1):
#             new_name = f"{str(index).zfill(4)}{img_format}"
#             old_path = os.path.join(folder_path, img)
#             new_path = os.path.join(folder_path, new_name)
#             os.rename(old_path, new_path)
#             ##重命名为0001.bmp的形式


'''
[option]4.将csv文件中的部分列取出来
'''

# features = [
#     "gaze_0_x", "gaze_0_y", "gaze_0_z",
#     "gaze_1_x", "gaze_1_y", "gaze_1_z",
#     "gaze_angle_x", "gaze_angle_y",
#     "pose_Tx", "pose_Ty", "pose_Tz",
#     "pose_Rx", "pose_Ry", "pose_Rz",
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
#     "AU28_c", "AU45_c"
# ]

# for mode in modes:
#     path = f"data/DAiSEE/{mode}"
#     csv_files = [file for file in os.listdir(path) if file.endswith(".csv")]

#     for csv_file in tqdm(csv_files, total=len(csv_files), desc=mode):
#         df = pd.read_csv(os.path.join(path, csv_file))
#         df = df[features]
#         csv_file = 'selected_' + csv_file
#         df.to_csv(os.path.join(path, csv_file), index=False)

# '''
# 5.构建annotation文件
# '''

DATASET = "EmotiW"
modes = ['Train', 'Validation']
# video2mode = _video_to_mode()
# for mode in modes:

    # df = pd.read_csv(f'./data/EmotiW/{mode}Labels.csv')
    # df.columns = ['xx','chunk', 'label']
    # df = df[['chunk', 'label']]
    # df['chunk'] = df['chunk'].apply(lambda x: x.split('.')[0])

# df.to_csv('./data/DAiSEE/Labels/AllLabels.csv', index=False)
# annotation_file = f'annotation/{DATASET}_{mode}_set.txt'

# base_path = "data/EmotiW"
# train_cnt = 0
# def process_label(x):

#     if x == 'Highly-Engaged':
#         return 3
#     elif x == 'Engaged':
#         return 2
#     elif x == 'Barely-engaged':
#         return 1
#     elif x == 'Not-Engaged':
#         return 0

#     print(x)
#     print('xxxxx')
#     return 1 / 0
# for mode in modes:
#     df = pd.read_csv(f'./data/EmotiW/{mode}Labels.csv')
#     df['label'] = df['label'].apply(process_label)
#     for index, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {mode}"):

#         # video_name = str(row['ClipID']).split('.')[0]
#         # label = row['Engagement']  # [0, 1, 2, 3]
#         # _mode = video2mode[video_name]['mode']
#         # _format = video2mode[video_name]['format']
#         video_name = str(row['chunk']).split('.')[0]
#         label = row['label']
#         _mode = mode
#         _format = '.mp4'
#         if os.path.exists(os.path.join(base_path,f'{_mode}_csvs', video_name + '.csv')): #有的csv文件已经没有了
#             data_csv = os.path.join(base_path,f'{_mode}_csvs', video_name + '.csv')
#             df_data = pd.read_csv(data_csv)
#             frame_cnt = len(df_data)
#             frame_cnt_frames = len(os.listdir(os.path.join(base_path, mode, video_name)))
#             f_cnt = min(frame_cnt, frame_cnt_frames)
#             if f_cnt >= 16:
#                 data_path = os.path.join(base_path, _mode, video_name)
#                 csv_path = data_csv
#                 with open(f'./annotation/{DATASET}_{_mode}_set.txt', 'a') as f:
#                     f.write(data_path + " " +str(f_cnt) + " " + str(label)+ " " + csv_path + '\n')



# for mode in modes:
#     data_path = os.path.join('data/DAiSEE', mode)
#     for folder in tqdm(os.listdir(data_path), total=len(os.listdir(data_path))):
#         if folder.endswith('.avi') or folder.endswith('.mp4') or folder.endswith('.hog') or folder.endswith('.txt'):
#             os.remove(os.path.join(data_path, folder))


# v2m = _video_to_mode() 
# for mode in modes:
#     data_path = f"data/DAiSEE/DataSet/{mode}"
#     for file in tqdm(os.listdir(data_path), total=len(os.listdir(data_path)), desc=f"mode"):
#         for f_file in os.listdir(os.path.join(data_path, file)):
#             ff_file = os.path.join(data_path, file, f_file)
#             for t_path in os.listdir(ff_file):
#                 name, format = t_path.split('.') #名称 + 后缀
#                 cap = cv2.VideoCapture(os.path.join(ff_file, t_path))
#                 frame_count = 0
#                 _mode = v2m[name]['mode']
#                 output_folder = f'data/DAiSEE/frames/{_mode}/{name}'
#                 os.makedirs(output_folder, exist_ok=True)
#                 while True:
#                     ret, frame = cap.read()
#                     if not ret:
#                         break
#                     picture = os.path.join(output_folder,f"{frame_count}.jpg")
#                     cv2.imwrite(picture, frame)
#                     frame_count += 1
#                 cap.release()

# s = {}
# for mode in modes:
#     data_path = f'data/DAiSEE/frames/{mode}'
#     for f in os.listdir(data_path)
#         print(f)
#         cnt = len(os.listdir(os.path.join(data_path, f)))
#         if cnt not in s.keys():
#             s[cnt] = 1
#         else:
#             s[cnt] += 1
# print(s)

# import torch
# import numpy as np
# from torchvision.transforms import transforms
# from PIL import Image
# import torch.nn as nn
# from efficientnet_pytorch import EfficientNet



# v2m = _video_to_mode() 
# df = pd.read_csv('./data/DAiSEE/Labels/AllLabels.csv')
# df = df[['ClipID', 'Engagement']]
# df['ClipID'] = df['ClipID'].apply(lambda x: x.split('.')[0])

# base_path = "data/DAiSEE"


# base_line = EfficientNet.from_pretrained('efficientnet-b7')
# base_line._fc = nn.Identity()
# model = base_line.to('cuda')
# # EfficientNet 预处理
# transform = transforms.Compose([
#     transforms.Resize(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])
# #获取图片数量
# train_embeddings = []
# test_embeddings = []
# valid_embeddings = []

# train_embeddings = np.empty((0, 2560))  # 初始化为空的二维数组，形状 [0, 2560]
# test_embeddings = np.empty((0, 2560))   # 同样初始化测试集嵌入
# valid_embeddings = np.empty((0, 2560))  # 同样初始化验证集嵌入
# for index, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing All"):

#     video_name = str(row['ClipID']).split('.')[0]
#     label = row['Engagement']  # [0, 1, 2, 3]
#     _mode = v2m[video_name]['mode']
#     _format = v2m[video_name]['format']
#     img_path = f'data/DAiSEE/frames/{_mode}/{video_name}'


#     img_files = sorted(os.listdir(img_path), key=lambda x: int(x.split('.')[0]))
#     cnt = len(os.listdir(img_path))
#     embeddings = []
#     if cnt < 300:

#         batch_files = img_files[:]
#     elif cnt >= 300:
#         batch_files = img_files[:300]
#     batch_embedding = []
#     tmp = []
#     for img in batch_files:
#         i_p = os.path.join(img_path ,img)
#         i = Image.open(i_p).convert("RGB")

#         img_tensor = transform(i).unsqueeze(0).to('cuda')

#         with torch.no_grad():
#             features = model(img_tensor) #[2560]

#         batch_embedding.append(features.cpu().numpy())
        
#     batch_embedding = np.concatenate(batch_embedding, axis=0) #[300, 2560]
#     last_row = batch_embedding[:-1, :]
#     batch_embedding = np.vstack([batch_embedding, np.tile(last_row, (300 - cnt, 1))])

#     if _mode == "Train":
#         train_embeddings = np.concatenate([train_embeddings, batch_embedding], axis=0) #[60 *k, 2560]
#     if _mode == "Test":
#         test_embeddings = np.concatenate([test_embeddings, batch_embedding], axis=0)
#     if _mode == "Validation":
#         valid_embeddings = np.concatenate([valid_embeddings, batch_embedding], axis=0)

    
# np.save('data/DAiSEE/frames/Train.npy',train_embeddings)
# np.save('data/DAiSEE/frames/Validation.npy', valid_embeddings)
# np.save('data/DAiSEE/frames/Test.npy', test_embeddings)
    
        
    



# for mode in modes:
#     data_path = f"data/DAiSEE/DataSet/{mode}"
#     for file in tqdm(os.listdir(data_path), total=len(os.listdir(data_path)), desc=f"{mode}"):
#         for f_file in os.listdir(os.path.join(data_path, file)):
#             ff_file = os.path.join(data_path, file, f_file)
#             for t_path in os.listdir(ff_file):
#                 name, format = t_path.split('.') #名称 + 后缀
#                 cap = cv2.VideoCapture(os.path.join(ff_file, t_path))
#                 frame_count = 0
#                 _mode = v2m[name]['mode']
#                 output_folder = f'data/DAiSEE/frames/{_mode}/{name}'
#                 os.makedirs(output_folder, exist_ok=True)
#                 while True:
#                     ret, frame = cap.read()
#                     if not ret:
#                         break
#                     picture = os.path.join(output_folder,f"{frame_count}.jpg")
#                     cv2.imwrite(picture, frame)
#                     frame_count += 1
#                 cap.release()


#Test里面的每一行重新命名了，同时将数据移动到对应的位置
test_datas = []
modes = ['Train', 'Validation', 'Test']
for mode in modes:
    datas = []
    with open(f'annotation/EmotiW_{mode}_set.txt', 'r') as f:
        for data in f.readlines():
            arrs = data.strip().split(' ')
            arrs[1] = '280'
            datas.append(arrs[0] + ' '+ arrs[1] + ' ' + arrs[2] + ' ' +arrs[3]+ '\n')
            # ori_img_path = arrs[0]
            # ori_csv_path = arrs[-1]
            # new_img_path = ori_img_path.replace('Train', 'Test')
            # new_csv_path = ori_csv_path.replace('Train', 'Test')
            # print(f'ori,new: {ori_img_path}, {new_img_path}, {ori_csv_path}, {new_csv_path}')
            # test_datas.append(data.replace('Train', 'Test'))

            # os.rename(ori_img_path, new_img_path)
            # os.rename(ori_csv_path, new_csv_path)
    
    with open(f'annotation/EmotiW_{mode}_set.txt', 'w') as f:
        f.writelines(datas)
# with open('annotation/EmotiW_Test_set.txt', 'w') as f:
#     f.writelines(test_datas)
