import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

import numpy as np

# 设置字体为宋体
plt.rcParams['font.sans-serif'] = ['SimSun']
# 生成日期范围
# start_date = datetime(2023, 12, 1)
# end_date = datetime(2023, 12, 7)
#
# date_range = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]
#
# # 模拟学习记录数据
# study_records = [1, 3, 6, 4, 5, 3, 4]
# # 设置 y 轴范围
# plt.ylim(1, 7)
# # 绘制折线图
# plt.plot(date_range, study_records, marker='o')
#
# # 格式化 x 轴日期显示
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
# plt.gca().xaxis.set_major_locator(mdates.DayLocator())
#
# # 设置标题和标签
# plt.title('学习记录')
# plt.xlabel('日期')
# plt.ylabel('学习闯关数')
#
# # 自动调整日期标签的角度，以避免重叠
# plt.gcf().autofmt_xdate()
#
# # 显示图表
# plt.show()


# import matplotlib.pyplot as plt
#
# # 学科名称
# subjects = ['你好', '集中', '丰富', '团结', '幸福']
#
# # 学习得分
# scores = [85, 92, 78, 88, 90]
#
# # 绘制柱状图
# plt.bar(subjects, scores)
#
# # 设置标题和标签
# plt.title('学习得分')
# plt.xlabel('手语单词')
# plt.ylabel('手语单词学习得分')
#
# # 显示图表
# plt.show()


# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# import numpy as np
#
# # 学生综合能力指标
# indicators = ['语言能力', '数学能力', '科学能力', '艺术能力', '体育能力', '社交能力']
#
# # 学生能力评分
# scores = [80, 90, 70, 85, 75, 95]
#
# # 计算六边形各个顶点的位置
# center_x = 0
# center_y = 0
# radius = 1
# angles = [i * 2 * np.pi / 6 for i in range(6)]
# vertices = [(center_x + radius * np.cos(angle), center_y + radius * np.sin(angle), score / 100) for angle, score in zip(angles, scores)]
#
# # 创建画布和子图
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# # 创建六边形面
# hexagon = Poly3DCollection([vertices], alpha=0.5, linewidths=1, edgecolors='r')
# hexagon.set_facecolor('b')
#
# # 添加六边形面到子图
# ax.add_collection3d(hexagon)
#
# # 绘制能力评分线
# lines = [((center_x, center_y, 0), vertex) for vertex in vertices]
# for line in lines:
#     ax.plot([line[0][0], line[1][0]], [line[0][1], line[1][1]], zs=[line[0][2], line[1][2]], linestyle='--', color='r')
#
# # 设置坐标轴范围
# ax.set_xlim(-1.2, 1.2)
# ax.set_ylim(-1.2, 1.2)
# ax.set_zlim(0, 1)
#
# # 设置坐标轴标签
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('能力评分')
#
# # 添加能力指标标签
# for i, indicator in enumerate(indicators):
#     angle = angles[i]
#     x = center_x + (radius + 0.2) * np.cos(angle)
#     y = center_y + (radius + 0.2) * np.sin(angle)
#     ax.text(x, y, 1, indicator, ha='center', va='center')
#
# # 设置标题
# plt.title('学生综合能力')
#
# # 显示图表
# plt.show()


import numpy as np
import matplotlib.pyplot as plt

# 学习记录数据
data = np.array([
    [80, 90, 70, 85],
    [75, 95, 80, 90],
    [85, 70, 75, 80],
    [90, 85, 90, 95]
])

# 用户名称和关卡名称
user_names = ['用户1', '用户2', '用户3', '用户4']
level_names = ['关卡1', '关卡2', '关卡3', '关卡4']

# 绘制折线图
fig, ax = plt.subplots()

for i in range(len(user_names)):
    ax.plot(level_names, data[i, :], marker='o', label=user_names[i])

# 设置图表标题和轴标签
ax.set_title('学习记录')
ax.set_xlabel('关卡名称')
ax.set_ylabel('得分')

# 设置刻度和标签
ax.set_xticks(np.arange(len(level_names)))
ax.set_xticklabels(level_names)
ax.legend()

# 显示图表
plt.show()