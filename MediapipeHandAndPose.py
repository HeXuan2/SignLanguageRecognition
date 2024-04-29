import cv2
import mediapipe as mp
import numpy as np
import time
from PIL import Image
# 初始化 MediaPipe 解决方案
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
# 导入手部检测模块
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

# 视频路径，若要打开摄像头，设为 0
# filePath = 'examples/media/test4.mp4'
filePath=0

# 打开视频
cap = cv2.VideoCapture(filePath)

# 初始化 Pose 和 Hands
with (
    mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose,
    mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands):
    while cap.isOpened():
        # flag, frame = cap.read()
        # if not flag:
        #     break
        success, image = cap.read()  # cap.read()返回两个值，一个是bool值，表示是否成功读取帧，第二个则是帧本身
        if not success:  # 没有捕捉到帧，忽略这一帧
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # 将帧转换为 RGB
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 处理帧以获取姿势和手部关键点
        pose_results = pose.process(image)
        hands_results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 绘制身体和手部关键点轨迹
        # 绘制姿势关键点在图像上。
        mp_drawing.draw_landmarks(
            image,
            pose_results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        if not hands_results.multi_hand_landmarks:  # 如果没有检测到手部关键点，则继续处理下一张图像
            continue
        image_height, image_width, _ = image.shape  # 获取图像的高度和宽度

        for hand_landmarks in hands_results.multi_hand_landmarks:
            # 在图像上绘制手部关键点和连接线
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
            # 保存带有标注的图像

        # 显示实时帧
        cv2.imshow('MediaPipe Pose and Hands', cv2.flip(image,1))

        if cv2.waitKey(10) & 0xFF == 27:
            break

# 释放资源
cap.release()
cv2.destroyAllWindows()
