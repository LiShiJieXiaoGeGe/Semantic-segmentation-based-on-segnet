import numpy as np
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import tensorflow as tf
import sys

class_number = 3 # 分类数


def encoder(input_height, input_width):
    """
    语义分割的第一部分，特征提取，主要用到VGG网络，模型输入要求(416,416,3)
    """

    # 输入层
    img_input = Input(shape=(input_height, input_width, 3))
    print('img_input',img_input)
    # 三行为一个结构单元，size减半
    # 416,416,3 -> 208,208,64,
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPool2D((2, 2), strides=(2, 2))(x)
    f1 = x  # 暂存提取的特征

    # 208,208,64 -> 104,104,128
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPool2D((2, 2), strides=(2, 2))(x)
    f2 = x  # 暂存提取的特征

    # 104,104,128 -> 52,52,256
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = MaxPool2D((2, 2), strides=(2, 2))(x)
    f3 = x  # 暂存提取的特征

    # 52,52,256 -> 26,26,512
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = MaxPool2D((2, 2), strides=(2, 2))(x)
    f4 = x  # 暂存提取的特征

    # 26,26,512 -> 13,13,512
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = MaxPool2D((2, 2), strides=(2, 2))(x)
    f5 = x  # 暂存提取的特征

    return img_input, [f1, f2, f3, f4, f5]


def decoder(feature_map_list, class_number, input_height=416, input_width=416, encoder_level=3):
    """
    """
    # 获取一个特征图，特征图来源encoder里面的f1,f2,f3,f4,f5; 这里获取f4
    feature_map = feature_map_list[encoder_level]
    # print('feature_map.shape',feature_map.shape)  # （26，26，512）

    # 解码过程 ，以下 （26,26,512） -> (208,208,64)
    # f4.shape=(26,26,512) -> 26,26,512
    x = ZeroPadding2D((1, 1))(feature_map)
    x = Conv2D(512, (3, 3), padding='valid')(x)
    x = BatchNormalization()(x)

    # 上采样，图像长宽扩大2倍，(26,26,512) -> (52,52,256)
    x = UpSampling2D((2, 2))(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(256, (3, 3), padding='valid')(x)
    x = BatchNormalization()(x)

    # 上采样，图像长宽扩大2倍 (52,52,512) -> (104,104,128)
    x = UpSampling2D((2, 2))(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(128, (3, 3), padding='valid')(x)
    x = BatchNormalization()(x)

    # 上采样，图像长宽扩大2倍，(104,104,128) -> (208,208,64)
    x = UpSampling2D((2, 2))(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(64, (3, 3), padding='valid')(x)
    x = BatchNormalization()(x)

    # 再进行一次卷积，将通道数变为2（要分类的数目） (208,208,64) -> (208,208,2)
    x = Conv2D(class_number, (3, 3), padding='same')(x)
    # reshape: (208,208,2) -> (208*208,2)
    x = Reshape((int(input_height / 2) * int(input_width / 2), -1))(x)

    # 求取概率
    output = Softmax()(x)
    # print('output.shape', output.shape)
    return output


def main(Height = 416,Width = 416):
    img_input,feature_map_list = encoder(Height,Width)
    output = decoder(feature_map_list,class_number=class_number,input_width=Width,input_height=Height,encoder_level=3)

    model = Model(img_input,output)
    model.summary()
    return model


if __name__ == '__main__':
    main(Height = 416,Width = 416)

