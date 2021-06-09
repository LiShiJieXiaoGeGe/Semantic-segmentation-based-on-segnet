import cv2 as cv
import os
import shutil
import numpy as np
from matplotlib import pyplot as plt
'''
os.path.splitext(file)分离文件名和扩展名
os.path.split(file) 将path分割成目录和文件名二元组返回
'''


def mkdir(path):
    if os.path.exists(path):  # 如果path路径存在
        # 递归删除文件夹下的所有子文件夹和子文件
        shutil.rmtree(path)
    # 创建目录
    os.mkdir(path)


def get_file_path(file_dir):  # 游走遍历文件目录
    L = []
    for root, dirs, files in os.walk(file_dir):  # 遍历所有文件
        # print("root:",root) 当前文件路径
        # print("dirs:",dirs) 当前路径下的目录
        # print("files",files) 当前路径下所有非目录文件
        for file in files:  # 遍历所有文件名
            if os.path.splitext(file)[1] == '.bmp':   # 指定尾缀  os.path.splitext(file)分离文件名和扩展名
                L.append(os.path.join(root, file))  # 拼接处绝对路径并放入列表
    print('总文件数目：', len(L))
    return L


def resize_and_save(path_list):

    save_file_dir = os.path.join('..','images')
    mkdir(save_file_dir)
    # 遍历所有bmp文件路径
    for i,item in enumerate(path_list):
        # 读取图片
        img_raw = cv.imread(item)
        # 截取roi
        img_roi = img_raw[768:1750,1100:1750]
        print(img_roi.shape)  # 打印resize大小
        # 修改图片名字并另存
        photo_name = os.path.split(item)[1]
        photo_name = str(i)+'.bmp'
        # saved_file_dir = os.path.join('..', 'images', class_name)
        saved_file_path = os.path.join('..', 'images', photo_name)
        print("saved_find_path",saved_file_path)

        # 保存
        cv.imwrite(saved_file_path, img_roi)
        # cv.imshow("roi",img_roi)


def test(img):
    img = img[768:1800,1100:1750]
    cv.imshow("img",img)

if __name__ ==  "__main__":

    list_dir = get_file_path(r'E:\capture\data')
    resize_and_save(list_dir)
    # img = cv.imread(r"E:\capture\data\1.bmp")
    # test(img)


    cv.waitKey(0)
    cv.destroyAllWindows()