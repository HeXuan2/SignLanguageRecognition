import cv2
import mediapipe as mp
import time

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
# 导入手部检测模块
mp_hands = mp.solutions.hands
# 对于静态图片:
IMAGE_FILES = []
BG_COLOR = (192, 192, 192)  # 灰色背景


# 对于摄像头输入:接收来自前端的视频流



cap = cv2.VideoCapture("data/SLR_dataset/mp4/jizhongme.mp4")
# cap = cv2.VideoCapture("data/SLR_dataset/mp4/fengfuai.mp4")
# cap = cv2.VideoCapture(0)

# 获取视频的帧率
fps = cap.get(cv2.CAP_PROP_FPS)

# 设置运行时间为4秒
countdown_time = 10  # 单位：秒



width = 400
height = 300
landmark_points = []  # 存储关键点坐标的列表

pose=mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

hands=mp_hands.Hands(
    model_complexity=0, # 模型复杂度，0表示最简单
    min_detection_confidence=0.5, # 最小检测置信度
    min_tracking_confidence=0.5) # 最小跟踪置信度
# 设置中文字体和大小
font = cv2.FONT_HERSHEY_SIMPLEX  # 设置为宋体
font_size = 1
color = (0, 255, 0)

# 记录开始时间
start_time = time.time()
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("忽略空相机帧.")
        break
    image = cv2.resize(image, (width, height))
    # 在当前帧上添加时间戳
    current_time = time.time()
    elapsed_time = current_time - start_time
    if elapsed_time >= countdown_time:
        break


    # 调整图像大小

    # 沿水平方向翻转图像
    # image = cv2.flip(image, 1)


    # 在当前帧上添加倒计时
    current_time = time.time()
    elapsed_time = current_time - start_time
    remaining_time = max(countdown_time - elapsed_time, 0)
    countdown = int(remaining_time) + 1
    cv2.putText(image, f"Countdown: {countdown}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # # 在当前帧上添加时间戳
    # timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # cv2.putText(image, timestamp, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # 为了提高性能，可选择将图像标记为不可写，以通过引用传递。
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resultsPose = pose.process(image)
    resultsHand=hands.process(image)

    # 在图像上绘制姿势。
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    frame_landmark_points = []  # 存储当前帧关键点坐标的列表

    h, w, c = image.shape
    # 输出特定关键点的坐标并绘制
    index=0;

    if resultsPose.pose_landmarks and resultsHand.multi_hand_landmarks :

        for landmark in [11, 13, 12, 14]:
            landmark_px = resultsPose.pose_landmarks.landmark[landmark]
            cx, cy = int(landmark_px.x * w), int(landmark_px.y * h)
            # print("landmark当前值", landmark)
            frame_landmark_points.append(cx)
            frame_landmark_points.append(cy)
            # print(f"关键点 {landmark} 的坐标为 ({cx}, {cy})")
            cv2.circle(image, (cx, cy), 5, (0, 255, 0), -1)

        for i, hand_landmarks in enumerate(resultsHand.multi_hand_landmarks):
            handedness = resultsHand.multi_handedness[i].classification[0].label
            # print(handedness)
            if len(resultsHand.multi_hand_landmarks) == 2:  # 如果检测到两只手
               #当两只手为不同的手时
               if len(resultsHand.multi_hand_landmarks) == 2:  # 如果检测到两只手
                   for landmark in [0,5,8,4]:
                       landmark_px = hand_landmarks.landmark[landmark]
                       cx, cy = int(landmark_px.x * w), int(landmark_px.y * h)
                       frame_landmark_points.append(cx)
                       frame_landmark_points.append(cy)
                       cv2.circle(image, (cx, cy), 5, (0, 255, 0), -1)


            elif len(resultsHand.multi_hand_landmarks) == 1:
                for landmark in [0, 5, 8, 4]:
                    landmark_px = hand_landmarks.landmark[landmark]
                    cx, cy = int(landmark_px.x * w), int(landmark_px.y * h)
                    if handedness == 'Left':
                        # print("一只left",landmark)
                        frame_landmark_points.append(cx)
                        frame_landmark_points.append(cy)
                        frame_landmark_points.append(0)
                        frame_landmark_points.append(0)
                        # left_hand_landmarks.append(cx)
                        # left_hand_landmarks.append(cy)
                        # right_hand_landmarks.append(0)
                        # right_hand_landmarks.append(0)
                        cv2.circle(image, (cx, cy), 5, (0, 255, 0), -1)
                    elif handedness=='Right':
                        # print("一只right", landmark)
                        frame_landmark_points.append(cx)
                        frame_landmark_points.append(cy)
                        frame_landmark_points.append(0)
                        frame_landmark_points.append(0)
                        # right_hand_landmarks.append(cx)
                        # right_hand_landmarks.append(cy)
                        # left_hand_landmarks.append(0)
                        # left_hand_landmarks.append(0)
                        cv2.circle(image, (cx, cy), 5, (255, 0, 0), -1)
            else:
                # print("识别多只手的情况")
                for i in range(0, 4):
                    frame_landmark_points.append(0)
                    frame_landmark_points.append(0)
                    frame_landmark_points.append(0)
                    frame_landmark_points.append(0)
                break
    else:
        continue

    # 交换索引值为4、5、6、7和8、9、10、11的位置的数值
    frame_landmark_points[4:8], frame_landmark_points[8:12] = frame_landmark_points[8:12], frame_landmark_points[4:8]
    # 交换索引值为12、13、14、15和16、17、18、19的位置的数值
    frame_landmark_points[12:16], frame_landmark_points[16:20] = frame_landmark_points[16:20], frame_landmark_points[12:16]
    landmark_points.append(frame_landmark_points)  # 将当前帧的关键点坐标添加到列表中



    # 水平翻转图像以进行自拍视图显示。
    cv2.imshow('MediaPipe Pose', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break
    # 将图像写入视频
    # output_video.write(image)

cap.release()


print(landmark_points)
num_elements = sum(len(row) for row in landmark_points)
print(num_elements)

for row in landmark_points:
    row_length = len(row)
    if(row_length!=24):
        print("一维数组的长度为:", row_length)
        print(row)
