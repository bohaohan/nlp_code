# -*- coding: UTF-8 -*-
__author__ = 'bohaohan'
import os
# from passage.models import RNN
from PIL import Image
import numpy as np
# from detect import *
from pre_pro import *
logos = ['Acura讴歌','Armani阿玛尼','AstonMartin阿斯顿马丁','Audi奥迪','Balenciaga巴黎世家',
         'Bally巴利','Bentley宾利','Benz奔驰','BMW宝马','CK卡文克莱','Coach蔻驰','Ferrari法拉利',
         'GUCCI古驰','LV路易威登','Piaget伯爵','Porsche保时捷','Rollsroyce劳斯莱斯','Titoni梅花',
         'Volvo沃尔沃','YSL圣罗兰']


def get_logo_index(name):
    i = 0
    for j in logos:
        if j == name:
            return i
        i+=1


def get_name(name):
    if '_' in name:
        return name[:name.index('_')]
    else:
        return name


def load_data():
    names = []
    str = ''
    data = np.empty((103, 1, 50, 50), dtype="float32")
    label = np.empty((103,), dtype="uint8")
    imgs = os.listdir("./collect_bi_32")
    for i in range(104):
        if i > 0:
            # img = Image.open("./img/" + imgs[i])
            # arr = np.asarray(img, dtype="float32")
            arr = get_data_new("./collect_bi_32/" + imgs[i])
            # arr1 = np.asarray(Image.open('./collect_bi_32/' + imgs[i]))
            # arr = [[1 for x in range(50)] for y in range(50)]
            # for k in range(len(arr1)):
            #     for j in range(len(arr1[0])):
            #         if arr1[k][j] == 0:
            #             arr[k][j] = 0
            #         else:
            #             arr[k][j] = 255
            # print imgs[i], arr
            name = imgs[i][:imgs[i].index(".")]
            # print arr
            # print name
            name = get_name(name)
            # if name not in names:
            #     names.append(name)
            #     str += '\''+ name + '\','
            index = get_logo_index(name)
            data[i-1, :, :, :] = arr
            label[i-1] = index
            # print label
            # print label[i-1]
    # print label
    # print str
    print 'get data success'
    # print data.shape(0)
    # for i in data[0]:
    #     for j in i:
    #         print j
    return data, label

def load_n():
    data = []
    label = []
    imgs = os.listdir("./collect_bi_32")
    for i in range(len(imgs)):
        if i > 0:
            # print imgs[i]
            # img = Image.open("./img/" + imgs[i])
            # arr = np.asarray(img, dtype="float32")
            arr1 = np.asarray(Image.open('./collect_bi_32/' + imgs[i]))
            arr = [1 for x in range(50) for y in range(50)]
            for k in range(len(arr1)):
                for j in range(len(arr1[0])):
                    if arr1[k][j] == 0:
                        arr[k*50 + j] = -1
                    else:
                        arr[k*50 + j] = 1
            # print imgs[i], arr
            name = imgs[i][:imgs[i].index(".")]
            # print arr
            name = get_name(name)
            # print name
            index = get_logo_index(name)
            data.append(arr)
            # label[i-1] = index
            label.append(index)
            # print label
            # print label[i-1]
    # print label
    print 'get data success'
    # print data.shape(0)
    # for i in data[0]:
    #     for j in i:
    #         print j
    # print label
    print data[1][1]
    return data, label

if __name__ == '__main__':
    # data, l = load_data()
    # # data = data/255
    # for i in data[0]:
    #     for j in i:
    #         print j
    arr = get_data_new("./collect_bi_32/Acura讴歌_0.jpg")
    # print arr
    for i in arr:
        print i
    # arr = [[0 for x in range(50)] for y in range(50)]
    # for i in range(len(arr)):
    #     for j in range(len(arr[0])):
    #         print arr[i][j]
    # img = np.asarray(Image.open('./collect_bi/Acura讴歌_1.jpg'))
    # for i in range(len(img)):
    #     for j in range(len(img[0])):
    #         if img[i][j] == 255:
    #             img[i][j] = 1
    #         else:
    #             img[i][j] = -1
    # print img
    # width = img.size[0]
    # height = img.size[1]
    # print width, height
    # for h in range(0, height):
    #     for w in range(0, width):
    #         pixel = img.getpixel((w, h))
            # print pixel
    # print img