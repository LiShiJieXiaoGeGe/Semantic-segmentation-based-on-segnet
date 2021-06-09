import _model
import numpy as np
import os, copy
from PIL import Image
import _train

HEIGHT = 416  # 图像的长
WIDHT = 416  # 图像的宽
CLASS_NUMBERS = 3 # 分类数
class_colors = [[0, 0, 0], [255, 0, 0],[0,255,0]]


def get_model():
    """ 加载模型和参数"""
    # 获取模型
    model = _model.main()

    # 加载参数
    # model.load_weights("callbacks/ep044-loss0.030-val_loss0.028.h5")
    model.load_weights('callbacks/0_last.h5')
    return model


def precess_img(img):
    """
     对图像进行预处理，使其满足输入数据的要求
     1， 改变大小，2. 数据归一化 3. 改变shape
    :param img: 要预处理的图像
    :return: 返回预处理后的数组
    """

    # 对测试图像进行预处理，以适应模型的输入
    test_img = img.resize((HEIGHT, WIDHT))  # 改变图像大小-> (416,416)
    test_img_array = np.array(test_img)  # 图像变成数组
    test_img_array = test_img_array / 255.  # 归一化
    # print(test_img_array.shape)
    # (416,416,30 -> (1,416,416,3)
    test_img_array = test_img_array.reshape(-1, HEIGHT, WIDHT, 3)
    # print(test_img_array.shape)
    return test_img_array


def predicting(model):
    """ 预测"""
    # 获取测试图片
    test_data_lines = _train.split_dataset(SEED=1024)[0]
    dir_name = 'dataset_stone/img_stone'
    for line in test_data_lines:
        test_line_name = line.split(';')[0]
        print(test_line_name)
        test_img = Image.open(dir_name+'/'+test_line_name)

        old_test_img = copy.deepcopy(test_img)  # 复制图像 ， 备份
        test_img_array = np.array(test_img)  # 图片转成数组
        # print(test_img_array.shape)
        original_height = test_img_array.shape[0]
        original_width = test_img_array.shape[1]

        test_img_array = precess_img(test_img)  # 图片预处理
        predict_picture = model.predict(test_img_array)[0]  # 预测
        predict_picture = predict_picture.reshape((int(HEIGHT / 2), int(WIDHT / 2), CLASS_NUMBERS))
        # (208,208,2) -> (208,208)两个图层对应位置比较，保存最大的索引
        # 通道数为2，所以里面保存的是 0或 1
        predict_picture = predict_picture.argmax(axis=-1)
        # print(predict_picture.shape)

        # 下面要做的是对合并的图层，分层
        seg_img_array = np.zeros((int(HEIGHT / 2), int(WIDHT / 2), 3))  # (208,208,3) 值全为0
        colors = class_colors

        # 根据一张合并的图层（predict_picture) 将不同的类赋予不同的颜色
        for cn in range(CLASS_NUMBERS):
            # seg_img 0通道
            seg_img_array[:, :, 0] += ((predict_picture[:, :] == cn) * colors[cn][0]).astype('uint8')
            # seg_img 1 通道
            seg_img_array[:, :, 1] += ((predict_picture[:, :] == cn) * colors[cn][1]).astype('uint8')
            # segm_img 2 通道
            seg_img_array[:, :, 2] + ((predict_picture[:, :] == cn) * colors[cn][2]).astype('uint8')

        # 数组 转换成 图片
        seg_img = Image.fromarray(np.uint8(seg_img_array))  # (208,208,3)
        # seg_img.show()
        # print(seg_img.size)
        # 恢复图像大小 (208,208,3) ->(600,800) 方便和原图叠加
        seg_img = seg_img.resize((original_width, original_height))

        # 合并图像
        print(old_test_img.size)
        print(seg_img.size)
        image = Image.blend(old_test_img, seg_img, 0.4)

        # 保存图片
        save_path = 'dataset_stone'
        save_path = os.path.join(save_path, 'imgout_train')
        save_path_seg = os.path.join(save_path,'imgout_seg')
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        if not os.path.exists(save_path_seg):
            os.mkdir(save_path_seg)

        image.save(save_path + '/' + test_line_name)
        seg_img.save(save_path_seg+'/'+test_line_name)


model = get_model()
predicting(model)



