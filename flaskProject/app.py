from flask import Flask, render_template, Response
import cv2 as cv
import numpy

app = Flask(__name__)
#电脑自带摄像头
camera0 = cv.VideoCapture(0)
#增加一个usb摄像头
camera1 = cv.VideoCapture(1)

@app.route("/")
def index():
    return render_template("index.html")

#获得本地摄像头图像字节流传输
def gen_frames0():
    while 1:
        ret,frame= camera0.read()
        if not ret:break
        #把获取到的图像格式转换(编码)成流数据，赋值到内存缓存中;
        #主要用于图像数据格式的压缩，方便网络传输
        ret1,buffer = cv.imencode('.jpg',frame)
        #将缓存里的流数据转成字节流
        frame = buffer.tobytes()
        #指定字节流类型image/jpeg
        yield  (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

#获得第二个摄像头图像字节流传输
def gen_frames1():
    while 1:
        ret,frame= camera1.read()
        if not ret:break
        ret1,buffer = cv.imencode('.jpg',frame)
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

if __name__=='__main__':
    app.run()