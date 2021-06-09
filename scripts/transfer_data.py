import os
import shutil

if __name__ == '__main__':

    dir_path = r'D:\MY_ALL_CODES\Project_LSJ\jsons'  # 复制 改名 改格式
    files = os.listdir(dir_path)
    # print(files)
    c = 0  # 新生成的文件序号，修改图片以及标签文件名

    for file in files:
        if os.path.isdir(dir_path+'/'+file):  # os.path.isdir()要求传入的是绝对路径，需要进行路径拼接
            # print(file,'是文件夹')
            new_path = dir_path+'/'+file
            # print('new path:',new_path)  D:/MY_ALL_CODES/Project_LSJ/test/34_bmp_0_224_json
            new_files = os.listdir(new_path)
            # print('new files',new_files)
            shutil.copyfile(new_path+'/'+new_files[0],'C:/Users/M S I/Desktop/code_learn/python/seg_vgg/dataset/img_ham'+'/'+
                            str(c)+'_'+new_files[0])
            shutil.copyfile(new_path + '/' + new_files[1],
                            'C:/Users/M S I/Desktop/code_learn/python/seg_vgg/dataset/label_ham' + '/' +
                            str(c) + '_' + new_files[1])
            with open(r'C:\Users\M S I\Desktop\code_learn\python\seg_vgg\dataset\train_ham.txt',mode='a+') as f:
                f.write(str(c)+'_'+new_files[0]+';'+str(c)+'_'+new_files[1]+'\n')

            c = c+1