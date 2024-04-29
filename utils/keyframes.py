import numpy as np
import operator


"""
这段代码定义了一个名为 Frame 的类，用于存储每一帧的信息，然后还有一个函数 extract_keyframes_indexes(frames, keyframe_num) 用于从视频帧中提取关键帧的索引。

Frame 类包含了一个构造方法 __init__ 用来初始化每一帧的 id 和 diff（差异值），以及一些比较方法（__lt__, __gt__, __eq__, __ne__）用于实现对象之间的比较操作。

extract_keyframes_indexes(frames, keyframe_num) 函数接收一个帧列表 frames 和关键帧数量 keyframe_num 作为输入，然后遍历帧列表计算相邻帧之间的差异，并将差异值存储到 frame_diffs 中。接着对 frame_diffs 进行排序，选取差异值最大的前 keyframe_num 个帧作为关键帧，并将它们的索引存储到 keyframe_id_set 中，最终返回关键帧的索引列表。

这段代码主要是用于视频处理中提取关键帧的功能，通过计算相邻帧之间的差异值来判断哪些帧可以作为关键帧。如果你对这段代码还有其他问题或需要进一步解释，请随时告诉我。
"""

class Frame:
    """class to hold information about each frame
    
    """
    def __init__(self, id, diff):
        self.id = id
        self.diff = diff
 
    def __lt__(self, other):
        if self.id == other.id:
            return self.id < other.id
        return self.id < other.id
 
    def __gt__(self, other):
        return other.__lt__(self)
 
    def __eq__(self, other):
        return self.id == other.id and self.id == other.id
 
    def __ne__(self, other):
        return not self.__eq__(other)


def extract_keyframes_indexes(frames, keyframe_num):
    if len(frames) <= keyframe_num:
        # 如果帧的数量小于等于关键帧数量，返回所有帧的索引
        print("帧的数量小于等于关键帧数量，返回所有帧的索引")
        return range(len(frames))
    curr_frame = None# 当前帧
    prev_frame = None# 前一帧
    frame_diffs = []# 存储帧间像素差异均值的列表
    new_frames = []# 存储帧对象的列表

    # 计算相邻帧间的像素差异均值
    for i in range(len(frames)):
        curr_frame = frames[i]
        if (curr_frame is not None and \
            prev_frame is not None):
            diff = np.asarray(abs(curr_frame - prev_frame))  # 计算像素差异
            diff_sum = np.sum(diff)  # 计算像素差异的总和
            diff_sum_mean = diff_sum / len(diff)  # 计算像素差异的均值
            frame_diffs.append(diff_sum_mean)  # 将像素差异均值添加到列表中
            frame = Frame(i, diff_sum_mean)  # 创建帧对象
            new_frames.append(frame)  # 将帧对象添加到列表中
        prev_frame = curr_frame

    # 计算关键帧的索引
    keyframe_id_set = set()  # 用于存储关键帧索引的集合

    # 根据像素差异均值对帧对象列表进行排序
    new_frames.sort(key=operator.attrgetter("diff"), reverse=True)

    # 选择像素差异最大的前N帧作为关键帧
    for keyframe in new_frames[:keyframe_num]:
        keyframe_id_set.add(keyframe.id)  # 将关键帧的索引添加到集合中

    return list(keyframe_id_set)  # 将关键帧索引转换为列表并返回

