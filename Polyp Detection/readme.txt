This code is based on the work of qqwweee/keras-yolo3. I'm really grateful to the [original implementation](https://github.com/qqwweee/keras-yolo3) by the authors, which is very useful.
This project only includes main source code. For the complete model and data, please refer to http://202.120.40.153:5000/sharing/3CJbMyYMd.
If you refer to or use this code, please cite the paper —— Xu J, Zhao R, Yu Y, et al. Real-time automatic polyp detection in colonoscopy using feature enhancement module and spatiotemporal similarity correlation unit[J]. Biomedical Signal Processing and Control, 2021, 66: 102503.
If you have any questions, please contact xujw96@foxmail.com. 

1. Training:
python train.py
Please put the image in "VOCdevkit\VOC2007\JPEGImages", and put their labels in "VOCdevkit\VOC2007\Annotations".
Then generate two txt file—"train.txt" and "test.txt", which include the image paths and the corresponding coordinates.

2. Image sets testing and calculating metrics: 
python test_metrics_image.py --image
Please put the images in "Data\Testing Data\Image Set\images", and put their labels in "Data\Testing Data\Image Set\labels".

3. Video sets testing and calculating metrics: 
python test_metrics_video.py --input "Data\Testing Data\videos\video1\video1.mp4" --output "Data\Testing Data\videos\video1\video1_result.mp4"

4. Image sets testing and visualization: 
python test_visualization_image.py --image
Please put the images in "Data\Testing Data\Image Set\images", and put their labels in "Data\Testing Data\Image Set\labels".
The visualization results are shown in "Data\Testing Data\Image Set\prediction".

5. Video sets testing and visualization: 
python test_visualization_video.py --input "Data\Testing Data\videos\video1\video1.mp4" --output "Data\Testing Data\videos\video1\video1_result.mp4"

6. False Positive Relearning Module
python FPRM.py
The new generated labels are in "Data\Training Data\labels_FPRM".
When training, please replace the corresponding labels in "train.txt" with the new generated labels.

7. Image Style Transfer Module
Please refer to the "README.md" in "/ISTM".
When training, please replace the corresponding image names in "train.txt" with the new image style transferred images.


Environment: 
Python 		3.6.9
Tensorflow 	1.14.0
Keras		2.2.4
