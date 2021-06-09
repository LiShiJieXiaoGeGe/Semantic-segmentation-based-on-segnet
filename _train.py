import _model
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.utils import get_file
import cv2 as cv
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt


# 设置gpu显存需要按需申请  处理异常 Function call stack
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)
# # tf.config.experimental.set_memory_growth(physical_devices[1], True)


CLASS_NUMBERS = 3 # 分几类
HEIGHT = 416  # 图片的长
WIDHT = 416  # 图片的宽
batch_size = 2  # 一次处理的图片数
img_path = r'D:/ALL_CODES/code_learn/python/seg_vgg/dataset_stone/img_stone/'  # 记得末尾再加一个  /
label_path = r'D:/ALL_CODES/code_learn/python/seg_vgg/dataset_stone/label_stone/'  # 记得末尾再加一个  /
train_txt_path = r'D:\ALL_CODES\code_learn\python\seg_vgg\dataset_stone\train.txt'


def get_model():
    '''
    获取模型，并加载官方预训练的模型参数 keras.utils.getfile
    :return:
    '''
    model = _model.main(HEIGHT,WIDHT)
    # 下载模型参数  这里之前已经下载好故注释掉
    # url = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
    # file_name = 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'  #下载后保存的文件名
    # weight_path = get_file(file_name,origin=url,cache_dir='models')
    # print(weight_path)

    # 根据参数路径加载参数
    weight_path = r'C:/Users/M S I/.keras/models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
    model.load_weights(weight_path,by_name=True)

    # 编译模型
    model.compile(optimizer='adam',loss = binary_crossentropy,metrics=['accuracy'])
    print("预加载模型成功")
    # model.summary()
    return model


def split_dataset(SEED):
    '''
    切分数据集为训练集和验证集和测试集
    :return:
    '''
    with open(train_txt_path,'r') as f:
        lines = f.readlines()
        np.random.seed(SEED)
        np.random.shuffle(lines)
        train_num = int(len(lines) * 0.8)
        val_num = int(len(lines) * 0.1)
        train_data_lines = lines[:train_num]
        val_data_lines = lines[train_num:train_num+val_num]

        test_data_lines = lines[train_num+val_num:]
        # print(test_data_lines)
    print("切分数据集成功...")
    return train_data_lines,val_data_lines,test_data_lines


def generate_arrays_from_file(lines,batch_size):
    '''
    生成器，将图片读入，并且预处理，之后喂入神经网络
    :param lines:训练集或测试集列表  [jpg;png/n...]
    :param batch_size: 每次处理的图片张数
    :return:
    '''

    readline = 0  # 当前所读行
    counts = len(lines)  # 数据个数
    while True:
        x_train = []
        y_train = []
        for i in range(batch_size):
            if readline == 0:  # 读完所有数据一次后，将数据打乱，以待下次读
                np.random.shuffle(lines)
            line = lines[readline]
            x_name = line.split(';')[0]
            x_img = cv.imread(img_path+x_name)
            x_img = np.array(x_img)
            x_img = cv.resize(x_img,dsize=(HEIGHT,WIDHT))
            # print('x_img.shape',x_img.shape)
            x_img = x_img / 255.  # 归一化
            x_train.append(x_img)

            y_name = line.split(';')[1].replace('\n', '')
            y_img = cv.imread(label_path+y_name)
            y_img = np.array(y_img)
            y_img = cv.resize(y_img,dsize=(int(HEIGHT/2),int(WIDHT/2)))  # resize到(208,208,3)
            # 三个通道灰度值相同，无法做交叉熵运算，故需进行图片分层
            # 图层变换过程：最终需要的是class_number个列表，代表每个像素的分类(208,208,3)-->(208,208,class_numbers)-->(208*208,class_numbers)
            labels = np.zeros(shape=(int(HEIGHT/2),int(WIDHT/2),CLASS_NUMBERS))

            for cn in range(CLASS_NUMBERS):
                labels[:,:,cn] = (y_img[:,:,0] == cn).astype(int)
            # print('labels.shape',labels.shape)
            labels = np.reshape(labels,(-1,CLASS_NUMBERS))  # （208，208，2）-->(208*208,2)
            # print('labels.shape',labels.shape)
            y_train.append(labels)

            readline = (readline+1)%counts

        yield np.array(x_train),np.array(y_train)  # 必须转换成numpy类型,tensorflow要求输入的就是numpy类型array
    print("加载图像成功...")


def set_callbacks():
    # 1. 有关回调函数的设置（callbacks)
    logdir = os.path.join("callbacks")
    print(logdir)
    if not os.path.exists(logdir):  # 如果没有文件夹
        os.mkdir(logdir)
    output_model_file = os.path.join(logdir, 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5')
    callbacks = [
        ModelCheckpoint(output_model_file, save_best_only=True, save_freq='epoch'),
        ReduceLROnPlateau(factor=0.5, patience=3),
        EarlyStopping(min_delta=1e-4, patience=10)
    ]
    return callbacks, logdir


def main():
    # 获取已建立的模型，并加载官方与训练参数，模型编译
    model = get_model()
    # 打印模型摘要
    # model.summary()

    # 设置回调函数 并返回保存的路径
    callbacks, logdir = set_callbacks()

    # 生成样本和标签

    # 训练
    split_dataset_times = 1
    SEED = 1024  # 用于切分数据集的种子
    for i in range(split_dataset_times):
        print('当前第',str(i+1),'次切分数据集')
        train_data_lines, val_data_lines, test_data_lines = split_dataset(SEED)
        train_nums = len(train_data_lines)
        print('train_num',train_nums)
        val_nums = len(val_data_lines)
        generate_arrays_from_file(train_data_lines, batch_size=batch_size)
        history = model.fit_generator(generate_arrays_from_file(train_data_lines, batch_size),
                            steps_per_epoch=max(1, train_nums // batch_size),
                            epochs=500, callbacks=callbacks,
                            validation_data=generate_arrays_from_file(val_data_lines, batch_size),
                            validation_steps=max(1, val_nums // batch_size),
                            initial_epoch=0)

        save_weight_path = os.path.join(logdir,str(i)+'_last.h5') # 保存模型参数的路径

        model.save_weights(save_weight_path)
        SEED += 1  # 不同的切分种子
        # ---------绘制---------------------------
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        plt.subplot(1, 2, 1)
        plt.plot(acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()