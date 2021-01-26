"""
Retrain the YOLO model for your own dataset.
"""
import numpy as np
import keras
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping,LearningRateScheduler
import tensorflow as tf
from yolo3.model import preprocess_true_boxes, yolo_body, yolo_loss
from yolo3.utils import get_random_data
import os
import pandas as pd
from keras.optimizers import Adam
import threading

class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):  # Py3
        with self.lock:
            return next(self.it)


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

def get_classes(classes_path):
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)

def create_model(input_shape, anchors, num_classes, load_pretrained=False, freeze_body=False,
            weights_path='model_data/yolo_weights.h5'):
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)
    y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], \
        num_anchors//3, num_classes+5)) for l in range(3)]

    model_body = yolo_body(image_input, num_anchors//3, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body:
            # Do not freeze 3 output layers.
            num = len(model_body.layers)-7
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)
    return model

@threadsafe_generator
def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes, random):
    n = len(annotation_lines)
    np.random.shuffle(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            i %= n
            image, box = get_random_data(annotation_lines[i], input_shape, random=random)
            image_data.append(image)
            box_data.append(box)
            i += 1
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)

def data_generator_wrap(annotation_lines, batch_size, input_shape, anchors, num_classes, random):
    n = len(annotation_lines)
    if n==0 or batch_size<=0: return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes, random)

def scheduler(epoch):
    # 每隔200个epoch，学习率减小为原来的1/10
    if epoch % 200 == 0 and epoch != 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.1)
        print("lr changed to {}".format(lr * 0.1))
    else:
        lr = K.get_value(model.optimizer.lr)
        print("lr changed to {}".format(lr))
    return K.get_value(model.optimizer.lr)

class Save_Model(keras.callbacks.Callback):
    def on_epoch_end(self,epoch, logs={}):
        epoch_num = str(epoch)
        loss = logs.get('val_loss')
        #val_loss = logs.get('val_loss')
        #print(epoch_num, val_loss)
        if loss<15:
            self.model.save_weights('model_data/darknet_' + epoch_num + '.h5')

def train(model, annotation_path, val_path, input_shape, anchors, num_classes, log_dir='logs/'):
    model.compile(optimizer=Adam(lr=1e-3), loss={
        'yolo_loss': lambda y_true, y_pred: y_pred})
    checkpoint = ModelCheckpoint(log_dir + "best_loss.h5",
        monitor='val_loss', save_weights_only=True, save_best_only=True, period=1,verbose=1)
    reduce_lr = LearningRateScheduler(scheduler)
    save_model=Save_Model()
    callbacks_list=[checkpoint,save_model,reduce_lr]
    batch_size = 10
    #val_split = 0.02
    with open(annotation_path) as f:
        lines = f.readlines()
    with open(val_path) as f:
        val_lines=f.readlines()
    np.random.shuffle(lines)
    np.random.shuffle(val_lines)
    num_val = len(val_lines)
    num_train = len(lines)
    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
    #model.load_weights('model_data/darknet_1199.h5')
    loss_log=model.fit_generator(data_generator_wrap(lines, batch_size, input_shape, anchors, num_classes,random=True),
            steps_per_epoch=max(1, num_train//batch_size),
            validation_data=data_generator_wrap(val_lines, batch_size, input_shape, anchors, num_classes, random=False),
            validation_steps=max(1, num_val//batch_size),
            epochs=600,
            callbacks=callbacks_list)

    pd.DataFrame(loss_log.history).to_csv(log_dir+'log.csv')
    model.save_weights(log_dir + 'trained.h5')



if __name__ == '__main__':
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    # tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,allow_soft_placement=True)
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # session = tf.Session(config = config)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'#屏蔽tf信息
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
    sess = tf.Session(config=config)
    K.set_session(sess)


    annotation_path = 'train.txt'
    val_path = 'test.txt'
    log_dir = 'model_data/'
    classes_path = 'model_data/voc_classes.txt'
    anchors_path = 'model_data/yolo_anchors.txt'
    class_names = get_classes(classes_path)
    anchors = get_anchors(anchors_path)
    input_shape = (416, 416)  # multiple of 32, hw


    model = create_model(input_shape, anchors, len(class_names))
    train(model, annotation_path, val_path, input_shape, anchors, len(class_names), log_dir=log_dir)