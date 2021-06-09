
import os,sys
from skimage import data_dir
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

def data_augment(dir_path):#数据增强

    datagen = ImageDataGenerator(
        rotation_range=40,  # 旋转角度
        width_shift_range=0.2,  # 平移
        height_shift_range=0.2,
        shear_range=0.,
        zoom_range=0.,
        horizontal_flip=False  # 翻转
    )
    dirs = os.listdir(dir_path)
    x_all = []
    for file in dirs:
        img = load_img(dir_path + "\\" + file)
        print("load path:", dir_path + "\\" + file)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)  # 这是一个numpy数组，形状为 (1, 3, 150, 150)
        x_all.append(x)
        i = 0
        for batch in datagen.flow(x, batch_size=1,
                                  save_to_dir=r'E:\LiShijie\aug_images', save_prefix=file):
            i += 1
            if i > 50:  # 数据扩充倍数，此处为数据扩充40倍
                break  # 否则生成器会退出循环


#--------
if __name__ == "__main__":
    data_augment(r'E:\LiShijie\images')
    pass
