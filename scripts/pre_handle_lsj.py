import cv2
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import os
import shutil


def transfer_data(dir_path):  # 搬运数据 dir_path必须是绝对路径

    files = os.listdir(dir_path)
    # print(files)
    c = 0  # 新生成的文件序号，修改图片以及标签文件名,必须有，要不然会造成文件的覆盖
    for file in files:
        if os.path.isdir(dir_path + '/' + file):  # os.path.isdir()要求传入的是绝对路径，需要进行路径拼接
            # print(file,'是文件夹')
            new_path = dir_path + '/' + file
            # print('new path:',new_path)  D:/MY_ALL_CODES/Project_LSJ/test/34_bmp_0_224_json
            new_files = os.listdir(new_path)
            # print('new files',new_files)
            shutil.copyfile(new_path + '/' + new_files[0],
                            'C:/Users/M S I/Desktop/code_learn/python/seg_vgg/dataset/img_ham_lsj' + '/' +str(c)+'_'+new_files[0])
            shutil.copyfile(new_path + '/' + new_files[1],
                            'C:/Users/M S I/Desktop/code_learn/python/seg_vgg/dataset/label_ham_lsj' + '/'+str(c)+'_'+new_files[1])
            c = c+1


def pre_handle_1(img):  # 只处理灰度图  仅仅是做了灰度值的处理

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


def batch_handle(dir_path):
    '''
    处理一个文件夹里的图片
    :param dir_path: 文件夹路径
    :return:
    '''
    for file in os.listdir(dir_path):
        full_path = os.path.join(dir_path,file)
        img = cv2.imread(full_path)
        img = pre_handle_1(img)
        cv2.imwrite(full_path,img)


if __name__ == "__main__":

    transfer_data(r'D:\MY_ALL_CODES\Project_LSJ\jsons\lsj_jsons')  # 搬运数据
    # 处理数据
    imgs = []
    base_path = 'C:/Users/M S I/Desktop/code_learn/python/seg_vgg/dataset/label_ham_lsj'
    img_paths = os.listdir(base_path)
    # sorted(img_paths)
    img_paths = sorted(img_paths,key=lambda x:int(x.split("_")[0]) )
    # print(img_paths)
    # print(len(img_paths))
    # [0,49]
    for img_path in img_paths[:50]:
        print("当前处理",img_path)
        full_path = base_path+'/'+img_path
        img = cv2.imread(full_path)
        img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        h = img_gray.shape[0]
        w = img_gray.shape[1]
        for i in range(h):
            for j in range(w):
                if img_gray[i][j] != 0:  # 红色ham
                    img_gray[i][j] = 2
        img = cv2.cvtColor(img_gray,cv2.COLOR_GRAY2BGR)
        cv2.imwrite(full_path,img)

    # [50,101)
    for img_path in img_paths[50:101]:
        # print("当前处理",img_path)
        full_path = base_path+'/'+img_path
        img = cv2.imread(full_path)
        img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        h = img_gray.shape[0]
        w = img_gray.shape[1]
        # 产生l
        l = []
        flag_l = False
        for j in range(h):
            for k in range(w):

                for c in l:
                    if c == img_gray[j][k]:
                        flag_l = True
                        break

                if flag_l == False:  # l中不存在该像素
                    l.append(img_gray[j][k])
                else:
                    flag_l = False

        l.sort()
        # cv2.imshow('img_', img)
        print(l)

        for j in range(h):
            for k in range(w):
                # for i in range(len(l)):  # [0,6)
                if img_gray[j][k] == 15 or img_gray[j][k] == 38 or img_gray[j][k] == 113:
                     img_gray[j][k] = 1
                elif img_gray[j][k] == 75:
                     img_gray[j][k] = 2

        img = cv2.cvtColor(img_gray,cv2.COLOR_GRAY2BGR)
        # cv2.imshow('img_',img)
        cv2.imwrite(full_path,img)

    # [101,152)
    for img_path in img_paths[101:152]:
        print("当前处理",img_path)
        full_path = base_path+'/'+img_path
        img = cv2.imread(full_path)
        img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        h = img_gray.shape[0]
        w = img_gray.shape[1]
        for i in range(h):
            for j in range(w):
                if img_gray[i][j] != 0:  #
                    img_gray[i][j] = 1
        img = cv2.cvtColor(img_gray,cv2.COLOR_GRAY2BGR)
        cv2.imwrite(full_path,img)

    for img_path in img_paths[152:203]:
        print("当前处理", img_path)
        full_path = base_path + '/' + img_path
        img = cv2.imread(full_path)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h = img_gray.shape[0]
        w = img_gray.shape[1]
        for i in range(h):
            for j in range(w):
                if img_gray[i][j] != 0:  #三角形
                    img_gray[i][j] = 1
        img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(full_path, img)

    for img_path in img_paths[203:254]:
        print("当前处理", img_path)
        full_path = base_path + '/' + img_path
        img = cv2.imread(full_path)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h = img_gray.shape[0]
        w = img_gray.shape[1]
        for i in range(h):
            for j in range(w):
                if img_gray[i][j] != 0:  #圆
                    img_gray[i][j] = 1
        img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(full_path, img)

    for img_path in img_paths[254:299]:
        print("当前处理", img_path)
        full_path = base_path + '/' + img_path
        img = cv2.imread(full_path)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h = img_gray.shape[0]
        w = img_gray.shape[1]
        for i in range(h):
            for j in range(w):
                if img_gray[i][j] != 0:  #正方形
                    img_gray[i][j] = 1
        img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(full_path, img)





