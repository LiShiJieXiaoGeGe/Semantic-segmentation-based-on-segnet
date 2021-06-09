import os
import shutil


def copy_file(src_path,obj_path):
    '''
    # 将文件合并到 img_ham_all  label_ham_all
    :param src_path: 
    :param obj_path: 
    :return: 
    '''
    global c  # 函数里用到外面定义的全局变量要声明一下

    path_lists = os.listdir(src_path)
    for path in path_lists:
        shutil.copyfile(src_path+'/'+path,  obj_path+'/'+str(c)+'_'+path.split('_')[1])
        print(obj_path+'/'+str(c)+'_'+path.split('_')[1])

        c = c+1


def generate_txt(lenth):

    with open(r'C:\Users\M S I\Desktop\code_learn\python\seg_vgg\dataset\train_ham.txt', mode='a+') as f:
        for i in range(lenth):
            f.write(str(i)+'_img.png;'+str(i)+'_label.png' + '\n')


if __name__ == '__main__':
    c = 0  # 全局变量
    copy_file(r'C:/Users/M S I/Desktop/code_learn/python/seg_vgg/dataset/lsj_dataset/img_ham_lsj',r'C:/Users/M S I/Desktop/code_learn/python/seg_vgg/dataset/img_ham_all')
    copy_file(r'C:/Users/M S I/Desktop/code_learn/python/seg_vgg/dataset/img_ham_other',r'C:/Users/M S I/Desktop/code_learn/python/seg_vgg/dataset/img_ham_all')
    c = 0
    copy_file(r'C:/Users/M S I/Desktop/code_learn/python/seg_vgg/dataset/lsj_dataset/label_ham_lsj',r'C:/Users/M S I/Desktop/code_learn/python/seg_vgg/dataset/label_ham_all')
    copy_file(r'C:/Users/M S I/Desktop/code_learn/python/seg_vgg/dataset/label_ham_other',r'C:/Users/M S I/Desktop/code_learn/python/seg_vgg/dataset/label_ham_all')
    lenth = len(os.listdir(r'C:/Users/M S I/Desktop/code_learn/python/seg_vgg/dataset/img_ham_all'))
    if len(os.listdir(r'C:/Users/M S I/Desktop/code_learn/python/seg_vgg/dataset/img_ham_all')) == len(os.listdir(r'C:/Users/M S I/Desktop/code_learn/python/seg_vgg/dataset/label_ham_all')):
        print('0k')
    else:
        print('fuck')
    generate_txt(lenth)


