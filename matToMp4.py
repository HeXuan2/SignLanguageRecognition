import cv2
import numpy as np
import scipy.io as scio

# 需要的关节位置的索引（从1开始，我们需要转换为从0开始的索引）
need_index = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 42, 43, 44, 45, 46, 47, 48, 49] #八组

# 加载.mat文件
mat_file_path = './data/SLR_dataset/xf500_body_depth_mat/000064/1.mat'  # 替换为您的.mat文件路径
mat_data = scio.loadmat(mat_file_path)

# 提取数据
data = mat_data['data'].astype(np.float32)

# 只保留需要的关节位置
data = data[:, need_index]

# 设置视频的尺寸和帧率
frame_width, frame_height = 640, 480
frame_rate = 30
# 定义一组颜色
colors = [
    (255, 0, 0),   # 红
    (255, 255, 0), # 黄
    (0, 0, 255),   # 蓝
    (0, 255, 0),   # 绿
    (128, 0, 128), # 紫
    (255, 255, 255), # 白
    (0, 255, 255),   # 青
    (255, 0, 255),   # 品红色
    (255, 192, 203), # 粉色
    (0, 100, 0)    ,  # 深绿色
    (255,120,24) ,#橙色
    (35,106,107) #土
]



# 为每个点选择一个颜色
color_index = 0
# 初始化视频编写器
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_path = 'data/SLR_dataset/mp4/tuanjieMat.mp4'  # 输出视频文件的路径
video_writer = cv2.VideoWriter(video_path, fourcc, frame_rate, (frame_width, frame_height))

# 生成视频
for frame_data in data:
    # 创建一个空白图像
    image = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

    # 假设数据中每两个数值代表一个点的x和y坐标
    for i in range(0,len(frame_data), 2):
        x = int(frame_data[i])
        y = int(frame_data[i + 1])

        # 确保坐标在图像尺寸范围内
        x = np.clip(x, 0, frame_width - 1)
        y = np.clip(y, 0, frame_height - 1)

        # 选择一个颜色
        color = colors[color_index]

        # 绘制点
        cv2.circle(image, (x, y), 5, color, -1)

        # 更新颜色索引
        color_index = (color_index + 1) % len(colors)
    # 将图像写入视频
    video_writer.write(image)

# 释放资源
video_writer.release()

print(f'Video saved to {video_path}')