import os
import signal
from tkinter import Image
import numpy as np
from flask_cors import CORS
from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import time
import json
import requests
from PIL import Image, ImageDraw, ImageFont


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
# 导入手部检测模块
mp_hands = mp.solutions.hands
# 对于静态图片:
IMAGE_FILES = []
BG_COLOR = (192, 192, 192)  # 灰色背景

# 设置运行时间为4秒
countdown_time = 5  # 单位：秒



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

app = Flask(__name__)
CORS(app)

CORS(app, origins='http://localhost:63343/frontweb', methods=['*'], allow_headers=['Content-Type'])

#电脑自带摄像头
# camera0 = cv2.VideoCapture("data/SLR_dataset/mp4/jizhongme.mp4")
camera0 = cv2.VideoCapture(0)

# 获取视频的帧率
fps =camera0.get(cv2.CAP_PROP_FPS)

#增加一个usb摄像头
camera1 = cv2.VideoCapture(1)


@app.route("/")
def index():
    return render_template("index.html")

def cv2AddChineseText(img, text, position, textColor=(0, 255, 0), textSize=30):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "SimHei.ttf", textSize, encoding="utf-8")
    # 绘制文本
    draw.text(position, text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


#获得本地摄像头图像字节流传输
def gen_frames0():
    result_dict = {}
    # 记录开始时间
    start_time = time.time()
    while 1:
        ret,frame= camera0.read()
        if not ret:break
        image = cv2.resize(frame, (width, height))
        # 在当前帧上添加时间戳
        current_time = time.time()
        elapsed_time = current_time - start_time
        if elapsed_time >= countdown_time:
            print(landmark_points)
            url = "http://127.0.0.1:60504/predict"
            data = {
                "frame_len": 24,  # 设置 frame_len 参数
                "skeleton_data": landmark_points}
            # 发送 POST 请求
            response = requests.post(url, data=json.dumps(data))
            print(response)
            # 解析响应数据
            result_dict = response.json()
            print(result_dict)
            # 将结果值写入到图像的右上角
            # 更新显示文字的持续时间
            # 重新开始倒计时
            start_time = time.time()
            landmark_points.clear()  # 清空关键点坐标列表
            continue


        if result_dict and result_dict['sucess'] == True:
                image = cv2AddChineseText(image, f"{result_dict['prediction']}", (30, 120), (255, 0, 0), 20)

        # 在当前帧上添加倒计时
        current_time = time.time()
        elapsed_time = current_time - start_time
        remaining_time = max(countdown_time - elapsed_time, 0)
        countdown = int(remaining_time) + 1
        cv2.putText(image, f"Countdown: {countdown}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)

        # 为了提高性能，可选择将图像标记为不可写，以通过引用传递。
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resultsPose = pose.process(image)
        resultsHand = hands.process(image)

        # 在图像上绘制姿势。
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        frame_landmark_points = []  # 存储当前帧关键点坐标的列表

        h, w, c = image.shape
        # 输出特定关键点的坐标并绘制
        index = 0;

        if resultsPose.pose_landmarks and resultsHand.multi_hand_landmarks:

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
                    # 当两只手为不同的手时
                    if len(resultsHand.multi_hand_landmarks) == 2:  # 如果检测到两只手
                        for landmark in [0, 5, 8, 4]:
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
                        elif handedness == 'Right':
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
        frame_landmark_points[4:8], frame_landmark_points[8:12] = frame_landmark_points[8:12], frame_landmark_points[
                                                                                               4:8]
        # 交换索引值为12、13、14、15和16、17、18、19的位置的数值
        frame_landmark_points[12:16], frame_landmark_points[16:20] = frame_landmark_points[
                                                                     16:20], frame_landmark_points[12:16]
        landmark_points.append(frame_landmark_points)  # 将当前帧的关键点坐标添加到列表中

        #把获取到的图像格式转换(编码)成流数据，赋值到内存缓存中;
        #主要用于图像数据格式的压缩，方便网络传输
        ret1,buffer = cv2.imencode('.jpg',image)
        #将缓存里的流数据转成字节流

        image = buffer.tobytes()
        #指定字节流类型image/jpeg
        yield  (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')

#获得第二个摄像头图像字节流传输
def gen_frames1():
    while 1:
        ret,frame= camera1.read()
        if not ret:break
        ret1,buffer = cv2.imencode('.jpg',frame)
        frame = buffer.tobytes()
        yield  (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

#网页端请求地址响应，注意那个mimetype资源媒体类型
#服务器推送，使用multipart/mixed混合类型的变种--multipart/x-mixed-replace。
#这里，“x-”表示属于实验类型。“replace”表示每一个新数据块都会代替前一个数据块。
#也就是说，新数据不是附加到旧数据之后，而是替代它，boundary为指定边界
@app.route('/video_feed0')
def video_feed0():
    return Response(gen_frames0(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed1')
def video_feed1():
    return Response(gen_frames1(),mimetype = 'multipart/x-mixed-replace; boundary=frame')

@app.route("/stop_app", methods=["GET"])
def stop_app():
    # 停止应用程序的运行
    os.kill(os.getpid(), signal.SIGINT)
    return jsonify(status="fail", message="App stopped")

@app.route("/getlandmark_points", methods=["GET"])
def getlandmark_points():
    return landmark_points

if __name__=='__main__':
    app.run(port=60500)
    # print(landmark_points)
