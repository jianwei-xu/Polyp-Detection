import colorsys
from timeit import default_timer as timer
import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw
from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
from keras.utils import multi_gpu_model
from xlutils.copy import copy
import glob
import xlrd  # 导入模块
from calcIOU import cal
import xml.dom.minidom
import os, shutil
import pandas as pd
import skimage
from skimage import io,measure,transform
import cv2

class YOLO(object):
    _defaults = {
        "model_path": 'model_data/darknet_198.h5',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'model_data/coco_classes.txt',
        "score" : 0.1,
        "iou" : 0.1,
        "model_image_size" : (416,416),
        "gpu_num" : 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image, filename,line,img_num,xlspath):
        start = timer()

        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')


        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })
        print("")
        print('Found {} boxes for {}'.format(len(out_boxes), 'img'),filename)

        coor_result=[]#coordinate list
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print("Confidence Score:",label, "(xmin,ymin,xmax,ymax)", (left, top,right, bottom))

            coor_result.append([left,top,right,bottom])

        end = timer()
        processing_time=end-start
        print("YOLO Processing time:",processing_time)
        return image,processing_time,coor_result

    def close_session(self):
        self.sess.close()

def detect_video(yolo, video_path="Data/Testing Data/videos/video1.avi",
                 output_path="Data/Testing Data/videos/video1_result.avi"):
    xlspath=""
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    totalFrameNumber = vid.get(cv2.CAP_PROP_FRAME_COUNT)
    video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)


    #建立一个图片数组，存7张图片。一个坐标列表，存7张图片的坐标。frame t-3, t-2, t-1, t, t+1, t+2, t+3
    frame_arr=np.zeros((((7,video_size[1],video_size[0],3))))
    coor_arr=[0 for x in range(0,7)]
    #frame[3]当前帧t，frame[0] t-3帧，frame[1] t-2帧，frame[2] t-1帧，frame[4] t+1帧，frame[5] t+2帧，frame[6] t+3帧
    #每一个循环更新一次列表

    #detectimage函数需要改动，返回所有坐标
    #detectimage中使用坐标列表，每一个循环append一次。ws.wirte之后append

    frame_num = 0
    while True:#循环处理每一帧
        line=0
        filename="frame "+str(frame_num)
        return_value, frame = vid.read()

        frame=frame[:,:,::-1]#BGR to RGB

        #detector network检测
        image = Image.fromarray(frame)
        image,pt,coor_result = yolo.detect_image(image,filename,line,frame_num,xlspath)
        # result = np.asarray(image)
        # result = result[...,::-1]  # RGB to BGR
        # result = result.astype(np.uint8)

        #更新图片列表，坐标列表
        if frame_num<=6:
            frame_arr[frame_num] = frame
            coor_arr[frame_num]=coor_result
        if frame_num>=7:
            for i in range(0,6):
                frame_arr[i]=frame_arr[i+1]
                coor_arr[i]=coor_arr[i+1]
            frame_arr[6]=frame
            coor_arr[6]=coor_result
        #print(coor_arr)
        #Inter-frame Similarity Corrlation Unit
        if frame_num>=3:
            start=timer()
            final_result=ISCU(frame_arr=frame_arr,coor_arr=coor_arr,frame_num=frame_num)
            end=timer()
            print("ISCU Processing Time:",end-start)

        else:
            final_result=coor_arr[frame_num]
        #print("final result",final_result)


        #可视化输出
        if frame_num<3:
            fps="FPS: "+str(int(1/pt))
            frame = frame[:, :, ::-1]  # BGR to RGB
            frame=frame.astype(np.uint8)
            cv2.putText(frame, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.50, color=(0, 0, 255), thickness=2)
            cv2.namedWindow("result", cv2.WINDOW_NORMAL)
            cv2.imshow("result", frame)
            if isOutput:
                out.write(frame)
        else:
            fps="FPS: "+str(int(1/pt))
            outframe=frame_arr[3]
            outframe = outframe[:, :, ::-1]  # BGR to RGB
            outframe = outframe.astype(np.uint8)
            cv2.putText(outframe, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.50, color=(0, 0, 255), thickness=2)
            cv2.namedWindow("result", cv2.WINDOW_NORMAL)
            for box_num in range(0,len(final_result)):
                cv2.rectangle(outframe,
                              (final_result[box_num][0], final_result[box_num][1]),
                              (final_result[box_num][2], final_result[box_num][3]),
                              (0, 255, 0), 3)
            cv2.imshow("result", outframe)
            if isOutput:
                out.write(outframe)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



        frame_num=frame_num+1
        if frame_num==totalFrameNumber:
            break
    yolo.close_session()


def ISCU(frame_arr,coor_arr,frame_num):
    iou_th=0.4#IOU阈值
    IOU_T1=0.4
    IOU_T2=0.3
    IOU_T3=0.2
    ssim_th=0.85 #ssim阈值
    SSIM=np.zeros(7)


    final_result=[]

    # for i in range(0,7):
    #     SSIM[i]=skimage.measure.compare_ssim(frame_arr[3],frame_arr[i],multichannel=True)
    SSIM[3]=0   #忽略与自己的相似度
    if (SSIM < ssim_th).all():  # 如果都不相似
        for i in range(0,len(coor_arr[3])):#循环当前帧的框
            tp_flag = 0
            x1=(coor_arr[3][i][0]+coor_arr[3][i][2])/2
            y1=(coor_arr[3][i][1]+coor_arr[3][i][3])/2
            w1=coor_arr[3][i][2]-coor_arr[3][i][0]
            h1=coor_arr[3][i][3]-coor_arr[3][i][1]

            if w1*h1<12689:
                IOU_T=IOU_T3
            elif w1*h1>72392:
                IOU_T=IOU_T1
            else:
                IOU_T=IOU_T2

            for ii in range(0,7):#循环相邻帧
                if ii==3:#忽略与当前帧的比较
                    continue
                for iii in range(0,len(coor_arr[ii])):
                    x2 = (coor_arr[ii][iii][0] + coor_arr[ii][iii][2]) / 2
                    y2 = (coor_arr[ii][iii][1] + coor_arr[ii][iii][3]) / 2
                    w2 = coor_arr[ii][iii][2] - coor_arr[ii][iii][0]
                    h2 = coor_arr[ii][iii][3] - coor_arr[ii][iii][1]
                    IOU=cal(x1, y1, w1, h1, x2, y2, w2, h2)
                    if IOU>IOU_T:
                        tp_flag=tp_flag+1

            if tp_flag>=3:
                final_result.append([int(coor_arr[3][i][0]),int(coor_arr[3][i][1]),
                                     int(coor_arr[3][i][2]),int(coor_arr[3][i][3])])

    else:
        for i in range(0,len(coor_arr[3])):#循环当前帧的框
            tp_flag = 0
            x1=(coor_arr[3][i][0]+coor_arr[3][i][2])/2
            y1=(coor_arr[3][i][1]+coor_arr[3][i][3])/2
            w1=coor_arr[3][i][2]-coor_arr[3][i][0]
            h1=coor_arr[3][i][3]-coor_arr[3][i][1]

            if w1*h1<12689:
                IOU_T=IOU_T3
            elif w1*h1>72392:
                IOU_T=IOU_T1
            else:
                IOU_T=IOU_T2

            for ii in range(0,7):#循环相邻帧
                if ii==3:#忽略与当前帧的比较
                    continue
                for iii in range(0,len(coor_arr[ii])):
                    x2 = (coor_arr[ii][iii][0] + coor_arr[ii][iii][2]) / 2
                    y2 = (coor_arr[ii][iii][1] + coor_arr[ii][iii][3]) / 2
                    w2 = coor_arr[ii][iii][2] - coor_arr[ii][iii][0]
                    h2 = coor_arr[ii][iii][3] - coor_arr[ii][iii][1]
                    IOU=cal(x1, y1, w1, h1, x2, y2, w2, h2)
                    if IOU>IOU_T:
                        tp_flag=tp_flag+1

            if tp_flag>(np.where(SSIM>=ssim_th)[0].shape[0])/2:
                final_result.append([int(coor_arr[3][i][0]), int(coor_arr[3][i][1]),
                                     int(coor_arr[3][i][2]), int(coor_arr[3][i][3])])
    return final_result

