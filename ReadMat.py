import cv2
import scipy.io

# 读取.mat文件
# mat_data = scipy.io.loadmat('data/SLR_dataset/xf500_body_depth_mat/000000/1.mat')
#
#
# # 创建一个空的DataFrame
# df = pd.DataFrame()
#
# # 将.mat文件中的每个变量存储到DataFrame的不同列中
# for key in mat_data:
#     if not key.startswith('__'):  # 跳过系统生成的元数据
#         df[key] = mat_data[key].flatten()  # 将数据存储到DataFrame的列中
#
# # 将DataFrame写入Excel文件
# df.to_excel('data/SLR_dataset/matToXlsx/output.xlsx', index=False)
import numpy as np
import scipy.io

from utils.keyframes import extract_keyframes_indexes

# # 找到最接近4900且能整除24的数
# number = 4900
# nearest_number = (number // 24) * 24  # 找到4900以下最大的能整除24的数
# print("离4900最近且能整除24的数是:", nearest_number)

import scipy.io
import numpy as np

from utils.plot_data import plot_data

# 从.mat文件读取数据
data = scipy.io.loadmat('data/SLR_dataset/xf500_body_depth_mat/000071/1.mat')
np.set_printoptions(threshold=np.inf)
# 获取需要存储的数据
data_array = data['data']



shape = data_array.shape
print("一维大小：", shape[0])
print("二维大小：", shape[1])

# 转换数组格式
resized_array = data_array.tolist()

print(resized_array)

# skeleton_data_array = np.array(resized_array, dtype=np.float32).reshape(-1, 24)
# print(skeleton_data_array)


# plot_data(resized_array)



# print(data_array)

# 设置NumPy的打印选项
# np.set_printoptions(threshold=np.inf,delimiter=',')

# key_indexes = extract_keyframes_indexes(data_array , 36)

# 输出完整的数组
# print(key_indexes)
