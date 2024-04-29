import os
import sys
import numpy as np
import torch
import scipy.io as scio
import matplotlib.pyplot as plt
import random
from utils.keyframes import *  # 导入自定义的函数或类
from utils.utils import *       # 导入自定义的工具函数

# 相对变换后的位置边界
crop_size = 256

# 需要的关节位置
need_index = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 42, 43, 44, 45, 46, 47, 48, 49]#一共24个 对应mediapipe的[11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22] x y

mat_data_dir = './data/SLR_dataset/xf500_body_depth_mat'
save_dir = './data/SLR_dataset/processed'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

data = []
label = []
index = 0
# 起止编号
start_idx = 45
end_idx = 95

# 关键帧数量
keyframe_num = 36
for index in range(start_idx, end_idx):
    label_dir = "%06d" % index
    print(label_dir)
    one_label_dir = os.path.join(mat_data_dir, label_dir)#形成一个完整路径
    one_label_data = []
    for one_mat in os.listdir(one_label_dir):
        # 读取mat数据
        one_data = scio.loadmat(os.path.join(one_label_dir, one_mat))['data'].astype(np.float32)
        print("len(one_data)",len(one_data))
        # 提取关键帧
        key_indexes = extract_keyframes_indexes(one_data, keyframe_num)  # 调用提取关键帧的函数
        # 剔除小于keyframe_num帧数据,不能出现这种情况 否则会报错
        if len(key_indexes) < keyframe_num:
            print(label_dir + one_mat + "length is too short")
            continue
        key_frames = one_data[key_indexes]
        one_label_data.append(key_frames)
        pass

    print("len(one_label_data)", len(one_label_data))
    one_label_data_array = np.array(one_label_data, dtype=np.float32)

    data.append(one_label_data_array)
    one_label = int(index - start_idx)
    label.append(np.ones(len(one_label_data)) * one_label)

print("one_label_data_array一维", len(one_label_data_array))
print("one_label_data_array二维", len(one_label_data_array[0]))
print("one_label_data_array三维", len(one_label_data_array[0][0]))
print("data一维长度",len(data))
print("data二维长度",len(data[0]))
print("data三维长度",len(data[0][0]))
print("need_index的长度",len(need_index))
# 截取部分需要的关节点
print("Cutting data...")
data_array = np.array(data, dtype=np.float32).reshape(-1, keyframe_num, 50)[:, :, need_index] #这里的50和len(data)得一样长
label_array = np.array(label, dtype=np.int16).reshape(-1, 1)
# 转换为相对坐标
print("Convert to relative coordinates...")
for i in range(len(data_array)):
    for j in range(len(data_array[i])):
        data_array[i][j] = abs2rel(data_array[i][j], crop_size)  # 调用转换为相对坐标的函数
# 打乱数据
# 这个操作的目的是为了在训练过程中，打乱数据集中样本的顺序，以增加模型的泛化能力，避免过拟合。
# 通常情况下，机器学习模型在处理数据时需要随机打乱数据，这可以防止模型过度依赖数据的顺序和排列方式。
print("Shuffle data...")
#这行代码创建了一个长度为 len(label_array) 的列表 shuffle_index，
# 并将其初始化为从 0 到 len(label_array)-1 的连续整数。这个列表被用来打乱数据集中样本的顺序。
shuffle_index = list(range(len(label_array)))
random.shuffle(shuffle_index)
data_array_shuffled = data_array[shuffle_index]
label_array_shuffled = label_array[shuffle_index]

# 保存文件
data_npy_name = "SLR_S" + str(start_idx) + "_E" + str(end_idx) + "_K" + str(keyframe_num) + "_body_data.npy"
label_npy_name = "SLR_S" + str(start_idx) + "_E" + str(end_idx) + "_K" + str(keyframe_num) + "_body_label.npy"

np.save(os.path.join(save_dir, data_npy_name), data_array_shuffled, allow_pickle=True)
np.save(os.path.join(save_dir, label_npy_name), label_array_shuffled, allow_pickle=True)
