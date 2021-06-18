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
        rb = xlrd.open_workbook(xlspath)
        wb = copy(rb)
        ws = wb.get_sheet(0)
        #ws.write(img_num,line,filename)
        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        #print(image_data.shape)
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

            ws.write(img_num,line+1, str(left))
            ws.write(img_num,line+2, str(top))
            ws.write(img_num,line+3, str(right))
            ws.write(img_num,line+4, str(bottom))
            wb.save(xlspath)
            line=line+4
            coor_result.append([left,top,right,bottom])


        end = timer()
        processing_time=end-start
        print("YOLO Processing time:",processing_time)
        return image,processing_time,coor_result

    def close_session(self):
        self.sess.close()

def detect_video(yolo, video_path="Data/Testing Data/videos/video1/video1.avi",
                 output_path="Data/Testing Data/videos/video1/video1_result.avi"):
    if os.path.exists("results"+".xls"):
        os.remove("results"+".xls")
    xlspath="results"+".xls"
    df=pd.DataFrame()
    df.to_excel(xlspath)

    if os.path.exists("iscu_results"+".xls"):
        os.remove("iscu_results"+".xls")
    xlspath2="iscu_results"+".xls"
    df=pd.DataFrame()
    df.to_excel(xlspath2)

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
    frame_arr_resize = np.zeros((((7, 64, 64, 3))))
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
            frame_arr_resize=transform.resize(frame,(64,64))
            coor_arr[frame_num]=coor_result
        if frame_num>=7:
            for i in range(0,6):
                frame_arr[i]=frame_arr[i+1]
                coor_arr[i]=coor_arr[i+1]
            frame_arr[6]=frame
            frame_arr_resize = transform.resize(frame, (64, 64))
            coor_arr[6]=coor_result
        #print(coor_arr)

        #Inter-frame Similarity Corrlation Unit
        if frame_num>=3:
            start=timer()
            final_result=ISCU(frame_arr=frame_arr_resize,coor_arr=coor_arr,frame_num=frame_num)
            end=timer()
            print("ISCU Processing Time:",end-start)

        else:
            final_result=coor_arr[frame_num]
        #print("final result",final_result)


        #可视化输出
        if frame_num<3:
            fps="FPS: "+str(int(1/pt))
            frame = frame[:, :, ::-1]  # RGB to BGR
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
            outframe = outframe[:, :, ::-1]  # RGB to BGR
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


    print("Waiting metrics calculating")
    print("Result without ISCU")
    rb = xlrd.open_workbook(xlspath)
    wb = copy(rb)
    ws = wb.get_sheet(0)
    for i in range(0,int(totalFrameNumber)):
        frame_num=str(i)
        ws.write(i, 0, frame_num.zfill(6))
    wb.save(xlspath)
    (path,filename)=os.path.split(video_path)
    xmlpath=path+"\\labels\\"
    metrics(xlspath=xlspath,xmlpath=xmlpath,img_num=int(totalFrameNumber))

    print("")
    print("Waiting metrics calculating")
    print("Result with ISCU")
    rb = xlrd.open_workbook(xlspath2)
    wb = copy(rb)
    ws = wb.get_sheet(0)
    for i in range(0,int(totalFrameNumber)):
        frame_num=str(i)
        ws.write(i, 0, frame_num.zfill(6))
    wb.save(xlspath2)
    (path,filename)=os.path.split(video_path)
    xmlpath=path+"\\labels\\"
    metrics(xlspath=xlspath2,xmlpath=xmlpath,img_num=int(totalFrameNumber))

def ISCU(frame_arr,coor_arr,frame_num):
    iou_th=0.4#IOU阈值
    IOU_T1=0.4
    IOU_T2=0.3
    IOU_T3=0.2
    ssim_th=0.85 #ssim阈值
    SSIM=np.zeros(7)

    rb = xlrd.open_workbook('iscu_results.xls')
    wb = copy(rb)
    ws = wb.get_sheet(0)
    line = 0
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
                ws.write(frame_num-3,line+1, str(int(coor_arr[3][i][0])))
                ws.write(frame_num-3,line+2, str(int(coor_arr[3][i][1])))
                ws.write(frame_num-3,line+3, str(int(coor_arr[3][i][2])))
                ws.write(frame_num-3,line+4, str(int(coor_arr[3][i][3])))
                line=line+4
                wb.save('iscu_results.xls')
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
                ws.write(frame_num-3,line+1, str(int(coor_arr[3][i][0])))
                ws.write(frame_num-3,line+2, str(int(coor_arr[3][i][1])))
                ws.write(frame_num-3,line+3, str(int(coor_arr[3][i][2])))
                ws.write(frame_num-3,line+4, str(int(coor_arr[3][i][3])))
                line=line+4
                wb.save('iscu_results.xls')
                final_result.append([int(coor_arr[3][i][0]), int(coor_arr[3][i][1]),
                                     int(coor_arr[3][i][2]), int(coor_arr[3][i][3])])
    return final_result


def write_imagename_to_excel(xlspath,imagepath):
    rb = xlrd.open_workbook(xlspath)
    wb = copy(rb)
    ws = wb.get_sheet(0)
    img_num = 0
    for jpgfile in glob.glob(imagepath):
        line = 0
        (filepath, filename) = os.path.split(jpgfile)
        ws.write(img_num, line, filename)
        img_num = img_num + 1
    wb.save(xlspath)

def metrics(xlspath,xmlpath,img_num):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    tp2 = 0
    cont = 0
    data = xlrd.open_workbook(xlspath)  # 打开电影.xlsx文件读取数据
    table = data.sheets()[0]  # 读取第一个（0）表单

    for i in range(0, img_num):
        a = table.row_values(i)
        while '' in a:
            a.remove('')
        jpgname = a[0]
        (xmlname, extension) = os.path.splitext(jpgname)
        xmlname = xmlpath + xmlname + ".xml"
        y = os.path.exists(xmlname)  # xml是否存在
        box_num = int((len(a) - 1) / 4)

        if (box_num == 0) and (y == False):
            tn = tn + 1
        elif y == True:
            dom = xml.dom.minidom.parse(xmlname)
            root2 = dom.documentElement

            bb1 = root2.getElementsByTagName('xmin')
            bb2 = root2.getElementsByTagName('ymin')
            bb3 = root2.getElementsByTagName('xmax')
            bb4 = root2.getElementsByTagName('ymax')
            object_num = bb1.length
            cont = cont + object_num
            xmin0 = np.zeros(object_num)
            ymin0 = np.zeros(object_num)
            xmax0 = np.zeros(object_num)
            ymax0 = np.zeros(object_num)
            x1 = np.zeros(object_num)
            y1 = np.zeros(object_num)
            w1 = np.zeros(object_num)
            h1 = np.zeros(object_num)
            for m in range(0, object_num):
                xmin0[m] = bb1[m].firstChild.data
                ymin0[m] = bb2[m].firstChild.data
                xmax0[m] = bb3[m].firstChild.data
                ymax0[m] = bb4[m].firstChild.data
                xmin0[m] = int(xmin0[m])
                ymin0[m] = int(ymin0[m])
                ymax0[m] = int(ymax0[m])
                xmax0[m] = int(xmax0[m])
                x1[m] = (xmin0[m] + xmax0[m]) / 2
                y1[m] = (ymin0[m] + ymax0[m]) / 2
                w1[m] = xmax0[m] - xmin0[m]
                h1[m] = ymax0[m] - ymin0[m]

        if (box_num != 0) and (y == False):
            fp = fp + box_num

        if y == True:
            for j in range(0, box_num):  # 循环检测出的框
                fp_flag = False
                xmin = int(table.cell(i, j * 4 + 1).value)
                ymin = int(table.cell(i, j * 4 + 2).value)
                xmax = int(table.cell(i, j * 4 + 3).value)
                ymax = int(table.cell(i, j * 4 + 4).value)
                x2 = (xmin + xmax) / 2
                y2 = (ymin + ymax) / 2
                w2 = xmax - xmin
                h2 = ymax - ymin
                for n in range(0, object_num):  # 循环ground truth

                    IOU = cal(x1[n], y1[n], w1[n], h1[n], x2, y2, w2, h2)
                    # if IOU>0.3:
                    if (x2 < (x1[n] + w1[n] / 2) and x2 > (x1[n] - w1[n] / 2) and y2 > (y1[n] - h1[n] / 2) and y2 < (
                            y1[n] + h1[n] / 2)) or IOU > 0.1:
                        tp = tp + 1
                        fp_flag = True
                        break

                if (fp_flag == False):
                    fp = fp + 1

            for n in range(0, object_num):#循环labels
                fn_flag = False
                for j in range(0, box_num):
                    xmin = int(table.cell(i, j * 4 + 1).value)
                    ymin = int(table.cell(i, j * 4 + 2).value)
                    xmax = int(table.cell(i, j * 4 + 3).value)
                    ymax = int(table.cell(i, j * 4 + 4).value)
                    x2 = (xmin + xmax) / 2
                    y2 = (ymin + ymax) / 2
                    w2 = xmax - xmin
                    h2 = ymax - ymin
                    IOU = cal(x1[n], y1[n], w1[n], h1[n], x2, y2, w2, h2)
                    # if IOU>0.3:
                    if (x2 < (x1[n] + w1[n] / 2) and x2 > (x1[n] - w1[n] / 2) and y2 > (y1[n] - h1[n] / 2) and y2 < (
                            y1[n] + h1[n] / 2)) or IOU > 0.1:
                        tp2 = tp2 + 1
                        fn_flag = True
                if (fn_flag == False):
                    fn = fn + 1

    Sen=tp/cont
    Pre=tp/(tp+fp)
    F1=2*Sen*Pre/(Sen+Pre)
    F2=5*Sen*Pre/(Sen+4*Pre)
    print("True Positive    =   ",tp)
    print("True Positive    =   ",tp2)
    print("False Positive   =   ",fp)
    print("False Negetive   =   ",fn)
    print("True Negative    =   ",tn)
    print("Total Polyps     =   ",cont)
    print("Sensitive        =   ",Sen)
    print("Precision        =   ", Pre)
    print("F1 Score         =   ", F1)
    print("F2 Score         =   ", F2)
