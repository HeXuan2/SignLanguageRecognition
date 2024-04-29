import cv2
import mediapipe as mp
import numpy as np

# 初始化 MediaPipe 解决方案
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 视频路径或打开摄像头
# filePath = 'input.mp4'
filePath = 0

# 打开视频
cap = cv2.VideoCapture(filePath)

# 设置保存视频的格式
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('sorry.mp4', fourcc, 20.0, (640, 480))

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # 将帧转换为 RGB，并进行镜像处理
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.flip(image, 1)  # 水平翻转图像

    # 处理帧以获取姿势和手部关键点
    pose_results = pose.process(image)
    hand_results = hands.process(image)

    # 清除轨迹图并重置为黑色背景
    track_img = np.zeros((480, 640, 3), dtype=np.uint8)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 绘制身体和手部关键点轨迹
    if pose_results.pose_landmarks:
        for landmark in pose_results.pose_landmarks.landmark:
            x = int(landmark.x * image.shape[1])
            y = int(landmark.y * image.shape[0])
            cv2.circle(track_img, (x, y), 1, (0, 255, 0), 5)

    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                x = int(landmark.x * image.shape[1])
                y = int(landmark.y * image.shape[0])
                cv2.circle(track_img, (x, y), 1, (255, 0, 0), 5)  # 使用不同颜色表示手部关键点

    # 镜像处理轨迹图
    track_img = cv2.flip(track_img, 1)

    # 显示实时帧
    cv2.imshow('MediaPipe Pose and Hands', image)

    # 显示轨迹图
    cv2.imshow('Track Image', track_img)

    # 将轨迹图写入视频文件
    out.write(track_img)

    if cv2.waitKey(10) & 0xFF == 27:
        break

# 释放资源
cap.release()
out.release()
hands.close()
pose.close()
cv2.destroyAllWindows()