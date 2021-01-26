import sys
import argparse
from yolo_visualization_video import YOLO, detect_video
from PIL import Image
import os
import pandas as pd
import glob
import xlrd
import xlwt
from xlutils.copy import copy
def detect_img(yolo):
    path = "Data\\testing data\\Image Set\\images\\*.jpg"
    outdir = "Data\\testing data\\Image Set\\prediction\\"
    xmlpath="Data\\testing data\\Image Set\\labels\\"
    if os.path.exists("results"+".xls"):
        os.remove("results"+".xls")
    xlspath="results"+".xls"
    df=pd.DataFrame()
    df.to_excel(xlspath)
    img_num=0
    for jpgfile in glob.glob(path):
        line=0
        (filepath, filename) = os.path.split(jpgfile)
        img = Image.open(jpgfile)
        img,pt = yolo.detect_image(img,filename,line,img_num,xlspath)
        img.save(os.path.join(outdir, os.path.basename(jpgfile)))
        img_num=img_num+1
    write_imagename_to_excel(xlspath=xlspath,imagepath=path)
    print("Total number of images",img_num)
    metrics(xlspath=xlspath,xmlpath=xmlpath,img_num=img_num)
    yolo.close_session()

FLAGS = None

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )

    parser.add_argument(
        '--image', default=False, action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    )
    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", nargs='?', type=str,required=False,default="./Data/Testing Data/videos/video1.avi",
        help = "Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="./Data/Testing Data/videos/video1_result.avi",
        help = "[Optional] Video output path"
    )

    FLAGS = parser.parse_args()

    if FLAGS.image:
        """
        Image detection mode, disregard any remaining command line arguments
        """
        print("Image detection mode")
        if "input" in FLAGS:
            print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)
        detect_img(YOLO(**vars(FLAGS)))
    elif "input" in FLAGS:
        detect_video(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output)
    else:
        print("Must specify at least video_input_path.  See usage with --help.")
