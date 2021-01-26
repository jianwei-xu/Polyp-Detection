from yolo_metrics_image import YOLO, detect_video,write_imagename_to_excel,metrics
from PIL import Image
import pandas as pd
import glob
import xlrd              #导入模块
import xlwt
import os, shutil
from calcIOU import cal
import  xml.dom.minidom
import os, shutil
from xml.dom.minidom import Document
import numpy as np
import argparse

def detect_img(yolo):
    path = "Data\\training data\\images\\*.jpg"
    #outdir = "Data\\testing data\\Image Set\\prediction\\"
    xmlpath="Data\\training data\\labels\\"
    if os.path.exists("FPRM_generate"+".xls"):
        os.remove("FPRM_generate"+".xls")
    xlspath="FPRM_generate"+".xls"
    df=pd.DataFrame()
    df.to_excel(xlspath)
    img_num=0
    for jpgfile in glob.glob(path):
        line=0
        (filepath, filename) = os.path.split(jpgfile)
        img = Image.open(jpgfile)
        img,pt = yolo.detect_image(img,filename,line,img_num,xlspath)
        #img.save(os.path.join(outdir, os.path.basename(jpgfile)))
        img_num=img_num+1
    write_imagename_to_excel(xlspath=xlspath,imagepath=path)
    print("Total number of images",img_num)
    metrics(xlspath=xlspath, xmlpath=xmlpath, img_num=img_num)
    yolo.close_session()
    return img_num

def FPRM(img_num):
    doc = Document()
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    tp2=0
    cont=0
    xml_path="Data\\training data\\labels\\"
    fpxml_path = "Data\\training data\\fplabels\\"


    data = xlrd.open_workbook('FPRM_generate.xls')  # 打开电影.xlsx文件读取数据
    table = data.sheets()[0]  # 读取第一个（0）表单

    for i in range(0, img_num):
        a = table.row_values(i)
        while '' in a:
            a.remove('')
        jpgname = a[0]
        (xmlname, extension) = os.path.splitext(jpgname)
        fpxmlname = fpxml_path +xmlname + ".xml"
        xmlname = xml_path + xmlname + ".xml"

        y = os.path.exists(xmlname)  # xml是否存在
        box_num = int((len(a) - 1) / 4)

        if (box_num == 0) and (y == False):
            tn = tn + 1
        elif y == True:
            shutil.copy(xmlname, fpxmlname)

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
                xmin1 = int(table.cell(i, j * 4 + 1).value)
                ymin1 = int(table.cell(i, j * 4 + 2).value)
                xmax1 = int(table.cell(i, j * 4 + 3).value)
                ymax1 = int(table.cell(i, j * 4 + 4).value)
                x2 = (xmin1 + xmax1) / 2
                y2 = (ymin1 + ymax1) / 2
                w2 = xmax1 - xmin1
                h2 = ymax1 - ymin1
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

                    #写入FP labels
                    domm = xml.dom.minidom.parse(fpxmlname)
                    root = domm.documentElement
                    obje = doc.createElement("object")
                    root.appendChild(obje)
                    name = doc.createElement("name")
                    name_txt = doc.createTextNode("fp")
                    name.appendChild(name_txt)
                    pose = doc.createElement("pose")
                    pose_txt = doc.createTextNode("Unspecified")
                    pose.appendChild(pose_txt)
                    truncated = doc.createElement("truncated")
                    truncated_txt = doc.createTextNode("0")
                    truncated.appendChild(truncated_txt)
                    difficult = doc.createElement("difficult")
                    difficult_txt = doc.createTextNode("0")
                    difficult.appendChild(difficult_txt)

                    bndbox = doc.createElement("bndbox")
                    xmin = doc.createElement("xmin")
                    xmin_txt = doc.createTextNode(str(xmin1))
                    xmin.appendChild(xmin_txt)
                    ymin = doc.createElement("ymin")
                    ymin_txt = doc.createTextNode(str(ymin1))
                    ymin.appendChild(ymin_txt)
                    xmax = doc.createElement("xmax")
                    xmax_txt = doc.createTextNode(str(xmax1))
                    xmax.appendChild(xmax_txt)
                    ymax = doc.createElement("ymax")
                    ymax_txt = doc.createTextNode(str(ymax1))
                    ymax.appendChild(ymax_txt)
                    bndbox.appendChild(xmin)
                    bndbox.appendChild(ymin)
                    bndbox.appendChild(xmax)
                    bndbox.appendChild(ymax)

                    obje.appendChild(name)
                    obje.appendChild(pose)
                    obje.appendChild(truncated)
                    obje.appendChild(difficult)
                    obje.appendChild(bndbox)
                    xmlname_fp = fpxmlname
                    f = open(xmlname_fp, "w")
                    f.write(domm.toprettyxml(indent="  "))
                    f.close()

            for n in range(0, object_num):  # 循环labels
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

FLAGS = None
if __name__ == '__main__':
    img_num=detect_img(YOLO())
    FPRM(img_num)

