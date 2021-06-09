import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil

path_2 = r"C:\Users\Administrator\Pictures\02_stone.bmp"
path = r"C:\Users\Administrator\Pictures\01_stone.bmp"
dir = r"E:/LiShijie/images/ash"
mask_dir = r"E:/LiShijie/images/masks"
dst_dir = r"E:/LiShijie/images/dsts"
ash_low = 50
ash_high = 150
# H: 26 ~ 114


def get_roi(dir,y_l,y_h,x_l,x_h):
    """
    批量提取dir文件夹中的图片roi
    :param dir:
    """
    filelist = os.listdir(dir)
    for path in filelist:
        full_path = dir + '/' + path
        src = cv.imread(full_path)
        src = src[y_l:y_h,x_l:x_h]
        cv.imwrite(full_path,src)


def change_file_name(dir):
    """
        批量修改文件夹名称
    :param dir:
    """
    fileList = os.listdir(dir)
    n = 0
    for i in fileList:
        # 设置旧文件名（就是路径+文件名）
        oldname = dir + '/' + i
        # 设置新文件名
        newname = dir + '/' + str(n + 1) + '_stone_with_ash.bmp'
        os.rename(oldname, newname)  # 用os模块中的rename方法对文件改名
        print(oldname, '======>', newname)
        n += 1


def nothing(a):
    """
    回调函数，不做任何事
    """
    pass


def hist(path):
    """
    显示直方图
    :param img:
    """
    src = cv.imread(path,0)
    plt.hist(src.ravel(),bins=256,rwidth=0.8,range=(0,255))
    plt.show()


def color_tracking():
    """
    颜色追踪
    """
    cv.namedWindow("bar")
    cv.createTrackbar('h_low','bar',0,180,nothing)
    cv.createTrackbar('h_high','bar',180,180,nothing)
    cv.createTrackbar('v_low','bar',0,255,nothing)
    cv.createTrackbar('v_high','bar',255,255,nothing)
    cv.createTrackbar('s_low','bar',0,255,nothing)
    cv.createTrackbar('s_high','bar',255,255,nothing)
    while True:
        src = cv.imread(path)
        height,width = src.shape[:2]
        frame = cv.resize(src,dsize=(round(width/2),round(height/2)))
        hsv = cv.cvtColor(frame,cv.COLOR_BGR2HSV)

        # 读取滑动条信息
        h_low = cv.getTrackbarPos('h_low','bar')
        h_high = cv.getTrackbarPos('h_high','bar')
        v_low = cv.getTrackbarPos('v_low','bar')
        v_high = cv.getTrackbarPos('v_high','bar')
        s_low = cv.getTrackbarPos('s_low','bar')
        s_high = cv.getTrackbarPos('s_high','bar')

        #生成掩膜
        hsv_low = np.array([h_low,s_low,v_low])
        hsv_high = np.array([h_high,s_high,v_high])
        mask = cv.inRange(hsv,hsv_low,hsv_high)  # 判断图像是否在[hsv_low,hsv_high]之间，输出的是二值化图像
        res = cv.bitwise_and(frame,frame,mask=mask)

        cv.imshow("src",frame)
        cv.imshow("mask",mask)
        cv.imshow("res",res)

        key = cv.waitKey(1)
        if key == 113 or key == 81 or key == 27:
            break;


def origin_LBP(img):
    """
    原始LBP特征
    """
    dst = np.zeros(img.shape,dtype=img.dtype)
    h,w=img.shape
    start_index=1
    for i in range(start_index,h-1):
        for j in range(start_index,w-1):
            center = img[i][j]
            code = 0
#             顺时针，左上角开始的8个像素点与中心点比较，大于等于的为1，小于的为0，最后组成8位2进制
            code |= (img[i-1][j-1] >= center) << (np.uint8)(7)
            code |= (img[i-1][j  ] >= center) << (np.uint8)(6)
            code |= (img[i-1][j+1] >= center) << (np.uint8)(5)
            code |= (img[i  ][j+1] >= center) << (np.uint8)(4)
            code |= (img[i+1][j+1] >= center) << (np.uint8)(3)
            code |= (img[i+1][j  ] >= center) << (np.uint8)(2)
            code |= (img[i+1][j-1] >= center) << (np.uint8)(1)
            code |= (img[i  ][j-1] >= center) << (np.uint8)(0)
            dst[i-start_index][j-start_index]= code
    return dst


def stone_detect(dir):
    """
        批处理dir中的文件
    :param dir:
    """
    file_list = os.listdir(dir)
    for path in file_list:
        full_path = dir+'/'+path  # 每张图片的完整路径
        src = cv.imread(full_path,0)
        mask = np.ones(src.shape, dtype='uint8')*255  # 掩膜，灰尘区域标黑
        dst = src.copy()
        h,w = src.shape[:2]
        for y in range(h):
            for x in range(w):
                if src[y][x] > ash_low and src[y][x] < ash_high:
                    dst[y][x] = 0
                    mask[y][x] = 0
        cv.imwrite(dst_dir+'/'+path,dst)
        cv.imwrite(mask_dir+'/'+path,mask)


def test():
   src = cv.imread(r"D:\ALL_CODES\code_learn\python\stone_with_ash\1_stone_with_ash_json\label.png")
   h, w, c = src.shape
   for y in range(h):
        for x in range(w):
            if src[y][x][0] != 0 :
                print(src[y][x][0],"  ",src[y][x][1],"  ",src[y][x][2])


def batch_handle_json(path):  # 批处理json文件:  path为json文件存放的路径
    json_file = os.listdir(path)
    os.system("activate base")  # 在当前进程打开子进程执行command命令,相当于在命令行下敲命令
    for file in json_file:
        os.system("labelme_json_to_dataset.exe %s"%(path + '/' + file))  # 格式化输出


def get_label(dir):  # 处理dir中的json中的label,ash = 1,stone = 2,background = 0

    json_list = os.listdir(dir)
    for json in json_list:
        img_path = dir+"/"+json+"/"+"label.png"
        src = cv.imread(img_path)
        # print("处理图像:  ",img_path)
        h,w,c = src.shape
        l = json.split("_")
        print(l)
        if len(l) == 5 :  # stone_with_ash
            for y in range(h):
                for x in range(w):
                    if src[y][x][2] != 0:  # 通道2是灰
                        src[y][x][0] = 1
                        src[y][x][1] = 1
                        src[y][x][2] = 1
                        continue
                    if src[y][x][1] != 0:  # 通道1是石头
                        src[y][x][0] = 2
                        src[y][x][1] = 2
                        src[y][x][2] = 2

        if len(l) == 3 and l[1] == "ash":
            for y in range(h):
                for x in range(w):
                    for t in range(c):
                        if src[y][x][t] != 0:
                            src[y][x][0] = 1
                            src[y][x][1] = 1
                            src[y][x][2] = 1

        if len(l) == 3 and l[1] == "stone":
            for y in range(h):
                for x in range(w):
                    for t in range(c):
                        if src[y][x][t] != 0:
                            src[y][x][0] = 2
                            src[y][x][1] = 2
                            src[y][x][2] = 2
        cv.imwrite(img_path,src)
        print("处理完毕： ",img_path)


def transfer_data(dir):  # 将jsons中的数据提取并且形成数据集
    file_list = os.listdir(dir)
    c = 0
    for file in file_list:
        img_path = dir+'/'+ file + "/img.png"
        label_path = dir + '/' + file + "/label.png"
        shutil.copyfile(img_path,r'D:/ALL_CODES/code_learn/python/seg_vgg/dataset_stone/img_stone'+'/'+str(c)+"_"+"img.png")
        shutil.copyfile(label_path,r'D:/ALL_CODES/code_learn/python/seg_vgg/dataset_stone/label_stone'+'/'+str(c)+"_"+"label.png")
        with open(r'D:\ALL_CODES\code_learn\python\seg_vgg\dataset_stone\train.txt', mode='a+') as f:
            f.write(str(c) + '_img.png;' + str(c) + '_label.png\n')
        print("复制完毕：",img_path,"    ",label_path)
        c += 1


if __name__ == '__main__':
    # batch_handle_json(r"D:\ALL_CODES\code_learn\python\stone_with_ash")
    # get_label(r"D:\ALL_CODES\code_learn\python\stone_with_ash")
    # test()
    transfer_data(r"D:/ALL_CODES/code_learn/python/jsons")


