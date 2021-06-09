
import cv2
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import os
import shutil


def test():

    global c
    c = c+1


def transfer_data(dir_path,img_path,label_path):  # 搬运数据 dir_path必须是绝对路径

    files = os.listdir(dir_path)
    # print(files)
    # c = 0  # 新生成的文件序号，修改图片以及标签文件名,必须有，要不然会造成文件的覆盖
    global c  # 声明c为全局变量
    for file in files:
        if os.path.isdir(dir_path + '/' + file):  # os.path.isdir()要求传入的是绝对路径，需要进行路径拼接
            # print(file,'是文件夹')
            new_path = dir_path + '/' + file
            # print('new path:',new_path)  D:/MY_ALL_CODES/Project_LSJ/test/34_bmp_0_224_json
            new_files = os.listdir(new_path)
            # print('new files',new_files)
            shutil.copyfile(new_path + '/' + new_files[0],
                            img_path + '/' +str(c)+'_'+new_files[0])
            shutil.copyfile(new_path + '/' + new_files[1],
                            label_path+ '/'+str(c)+'_'+new_files[1])
            print('saved: ',new_path + '/' + new_files[0],img_path + '/' +str(c)+'_'+new_files[0])
            c = c+1


def pre_handle(img):  # 只处理灰度图  仅仅是做了灰度值的处理

    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    (h,w) = img.shape
    l = []
    flag_l = False
    print('img.shape',img.shape)

    # 产生l
    for j in range(h):
        for k in range(w):

            for c in l:
                if c == img[j][k]:
                    flag_l = True
                    break
            if flag_l == False:  # l中不存在该像素
                l.append(img[j][k])
            else:
                flag_l = False
    l = np.array(l)
    l = np.sort(l)
    # cv2.imshow('img_', img)
    print(l)

    for j in range(h):
        for k in range(w):
            for i in range(len(l)):  # [0,6)
                if img[j][k] == l[i] :
                    img[j][k] = int(256/6 * i)

    img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    # cv2.imshow('img_', img)
    return img


if __name__ == '__main__':

    c = 0   # 全局变量
    transfer_data(r'D:\MY_ALL_CODES\Project_LSJ\jsons\14_24_jsons',img_path=r'C:\Users\M S I\Desktop\code_learn\python\seg_vgg\dataset\img_ham_other',
                  label_path=r'C:\Users\M S I\Desktop\code_learn\python\seg_vgg\dataset\label_ham_other')  # 搬运数据
    transfer_data(r'D:\MY_ALL_CODES\Project_LSJ\jsons\25_33_jsons',img_path=r'C:\Users\M S I\Desktop\code_learn\python\seg_vgg\dataset\img_ham_other',
                  label_path=r'C:\Users\M S I\Desktop\code_learn\python\seg_vgg\dataset\label_ham_other')  # 搬运数据
    transfer_data(r'D:\MY_ALL_CODES\Project_LSJ\jsons\34_40_jsons',img_path=r'C:\Users\M S I\Desktop\code_learn\python\seg_vgg\dataset\img_ham_other',
                  label_path=r'C:\Users\M S I\Desktop\code_learn\python\seg_vgg\dataset\label_ham_other')  # 搬运数据
    dir_path = r'C:/Users/M S I/Desktop/code_learn/python/seg_vgg/dataset/label_ham_other'
    path_list = os.listdir(dir_path)
    path_list = sorted(path_list,key=lambda x: int(x.split('_')[0]))
    for img_path in path_list:
        img = cv2.imread(dir_path+'/'+img_path)
        img = pre_handle(img)
        cv2.imwrite(dir_path+'/'+img_path,img)

    # 统一处理
    path_list = os.listdir(dir_path)
    path_list = sorted(path_list, key=lambda x: int(x.split('_')[0]))
    img = cv2.imread(dir_path+'/'+path_list[0],0)

    # print(path_list)
    l = [0,42,85,128,170,213]
    for i in range(1069):

        img = cv2.imread(dir_path + '/' + path_list[i], 0)

        if (i>=153 and i<=203) or (i>=306 and i<=406):
            h = img.shape[0]
            w = img.shape[1]
            for m in range(h):
                for n in range(w):
                    if img[m][n] != 0:
                        if img[m][n] == l[1]:
                            img[m][n] = 2
                        else:
                            img[m][n] = 1


        if (i>=0 and i<=50) or (i>=967 and i<=1017):
            h = img.shape[0]
            w = img.shape[1]
            for m in range(h):
                for n in range(w):
                    if img[m][n] != 0:
                        if img[m][n] == l[2]:
                            img[m][n] = 2
                        else:
                            img[m][n] = 1

        if (i>=407 and i<=457) or (i>=509 and i<=559) or (i>=1018 and i<=1068):
            h = img.shape[0]
            w = img.shape[1]
            for m in range(h):
                for n in range(w):
                    if img[m][n] != 0:
                        if img[m][n] == l[3]:
                            img[m][n] = 2
                        else:
                            img[m][n] = 1

        if i>=560 and i<=610:
            h = img.shape[0]
            w = img.shape[1]
            for m in range(h):
                for n in range(w):
                    if img[m][n] != 0:
                        if img[m][n] == l[4]:
                            img[m][n] = 2
                        else:
                            img[m][n] = 1


        if i>=51 and i<=101 :
            h = img.shape[0]
            w = img.shape[1]
            for m in range(h):
                for n in range(w):
                    if img[m][n] != 0:
                        if img[m][n] == l[5]:
                            img[m][n] = 2
                        else:
                            img[m][n] = 1

        if (i>=102 and i<=152) or (i>=204 and i<=305) or (i>=916 and i<=966) or (i>=458 and i<=508):
            h = img.shape[0]
            w = img.shape[1]
            for m in range(h):
                for n in range(w):
                    if img[m][n] != 0:
                        img[m][n] = 1

        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(dir_path+'/'+path_list[i],img)