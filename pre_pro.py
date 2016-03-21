# -*- coding: UTF-8 -*-
__author__ = 'bohaohan'
import cv2

# img = cv2.imread('img/Acura讴歌_2.jpg',0)
# ret,thresh1 = cv2.threshold(img, 125, 255, cv2.THRESH_BINARY)
# cv2.imwrite('aaa.jpg', thresh1)
# from detect import *
import glob
import sys
import numpy as np
import codecs
reload(sys)
sys.setdefaultencoding('utf-8')
from skimage.filters import threshold_otsu, threshold_adaptive
# l = -1
# a = []
# file = codecs.open('array.json', 'wb', encoding='utf-8')
# str = '['
# for imagePath in glob.glob("./img/*.*"):
#     if '_0' in imagePath:
#         l+=1
#         imagePath.encode('utf-8')
#         imagename = imagePath[imagePath.rfind("/") + 1:imagePath.rfind(".")]
#         print l, get_name(imagename)
#         str += "'"+get_name(imagename) + "',"
#         a.append(get_name(imagename))
# str += ']'
# # print str
#
logos = ['Acura讴歌','Armani阿玛尼','AstonMartin阿斯顿马丁','Audi奥迪','Balenciaga巴黎世家',
         'Bally巴利','Bentley宾利','Benz奔驰','BMW宝马','CK卡文克莱','Coach蔻驰','Ferrari法拉利',
         'GUCCI古驰','LV路易威登','Piaget伯爵','Porsche保时捷','Rollsroyce劳斯莱斯','Titoni梅花',
         'Volvo沃尔沃','YSL圣罗兰']


def get_logo_index(name):
    i = 0
    for j in logos:
        if j == name:
            return i
        i += 1

# print logos.index('高缇耶')


def get_data_new(path):
    img = cv2.imread(path, 0)
    arrary = np.asarray(img, dtype="float32")
    a = np.empty((50, 50), dtype="float32")
    height = len(arrary)
    width = len(arrary[0])
    for i in range(height):
        for j in range(width):
            a[i][j] = arrary[i][j]
    return a


def get_data(path):
    img = cv2.imread(path, 0)
    arrary = np.asarray(img, dtype="float32")
    height = len(arrary)
    width = len(arrary[0])
    if height > width:
        ratio = 50/float(height)
    else:
        ratio = 50/float(width)
    # print ratio
    img = cv2.resize(img, None, fx=ratio, fy=ratio)
    arrary = np.array(img)
    height = len(arrary)
    width = len(arrary[0])

    a = np.empty((50, 50), dtype="float32")
    # a = [1 for x in range(100) for y in range(100)]
    for i in range(50):
        for j in range(50):
            a[i][j] = 255
    # print a
    for i in range(height):
        for j in range(width):
            a[i][j] = arrary[i][j]
    for i in range(50):
        for j in range(50):
            if a[i][j] >= 125:
                a[i][j] = 1
            else:
                a[i][j] = 0
    return a
    # for i in range(100):
    #     for j in range(100):
    #         if j == 99 and i == 99:
    #             result += str(a[i][j])
    #         else:
    #             result += str(a[i][j]) + " "
    # # print height, width
    #
    # print result
    # return result

    # l +=1
    # print l
    # imageName = imagePath[imagePath.rfind("/") + 1:imagePath.rfind(".")]
    # img = cv2.imread(imagePath,0)
    # ret, thresh1 = cv2.threshold(img, 125, 255, cv2.THRESH_BINARY)
    # cv2.imwrite(imagePath, thresh1)

def get_data_test(path):
    img = cv2.imread(path, 0)
    arrary = np.asarray(img, dtype="float32")
    height = len(arrary)
    width = len(arrary[0])
    if height > width:
        ratio = 50/float(height)
    else:
        ratio = 50/float(width)
    # print ratio
    img = cv2.resize(img, None, fx=ratio, fy=ratio)
    arrary = np.array(img)
    height = len(arrary)
    width = len(arrary[0])

    # a = np.empty((100, 100), dtype="float32")
    a = [1 for x in range(50) for y in range(50)]
    for i in range(50):
        for j in range(50):
            a[i*50+j] = 255
    # print a
    for i in range(height):
        for j in range(width):
            a[i*50+j] = arrary[i][j]
    for i in range(50):
        for j in range(50):
            if a[i*50+j] >= 125:
                a[i*50+j] = 1
            else:
                a[i*50+j] = -1
    a = np.asfarray(a)
    a[a == 0] = -1
    return a

def get_data_n(path):
    img = cv2.imread(path, 0)
    arrary = np.asarray(img, dtype="float32")
    height = len(arrary)
    width = len(arrary[0])
    if height > width:
        ratio = 100/float(height)
    else:
        ratio = 100/float(width)
    # print ratio
    img = cv2.resize(img, None, fx=ratio, fy=ratio)
    arrary = np.array(img)
    height = len(arrary)
    width = len(arrary[0])

    # a = np.empty((100, 100), dtype="float32")
    a = [1 for x in range(100) for y in range(100)]
    for i in range(100):
        for j in range(100):
            a[i*100+j] = 255
    # print a
    for i in range(height):
        for j in range(width):
            a[i*100+j] = arrary[i][j]
    for i in range(100):
        for j in range(100):
            if a[i*100+j] >= 125:
                a[i*100+j] = 1
            else:
                a[i*100+j] = -1
    # a = np.asfarray(img)
    # a[a == 0] = -1
    return a


def get_t_index(index):
    a = [0 for i in range(102)]
    a[index] = 1
    res = ""
    for i in range(102):
        if i == 101:
            res += str(a[i])
        else:
            res += str(a[i]) + " "
    return res
#
# if __name__ == '__main__':
#     k = 0
#     file = codecs.open('logo1.data', 'wb', encoding='utf-8')
#     indexs = []
#     for imagePath in glob.glob("./img/*.*"):
#         k+=1
#         print imagePath
#         imageName = imagePath[imagePath.rfind("/") + 1:imagePath.rfind(".")]
#         index = get_logo_index(get_name(imageName))
#         if index not in indexs:
#             indexs.append(index)
#         result = get_data(imagePath)
#         if index != None:
#             index_s = get_t_index(index)
#             line = str(result) + '\n'
#             file.write(line.decode("unicode_escape"))
#             line = str(index_s) + '\n'
#             file.write(line.decode("unicode_escape"))
#
#     print k, "total"
#     print len(indexs)
#     print indexs
# img = cv2.imread("./img/Acura讴歌_2.jpg", 0)
# print img
# arrary = numpy.array(img)
# k = 0
# height = len(arrary)
def pre_pro():
    k =0
    for imagePath in glob.glob("./img/*.*"):
        k+=1
        print imagePath
        img = cv2.imread(imagePath,0)
        ret, thresh1 = cv2.threshold(img, 135, 255, cv2.THRESH_BINARY)
        cv2.imwrite(imagePath, thresh1)
        # imageName = imagePath[imagePath.rfind("/") + 1:imagePath.rfind(".")]

# pre_pro()

def resize_img():
    for imagePath in glob.glob("./test_img3_bi/*.*"):
        imagePath.encode('utf-8')
        imagename = imagePath[imagePath.rfind("/") + 1:]
        # print imagename
        img = cv2.imread(imagePath, 0)
        arrary = np.asarray(img, dtype="float32")
        height = len(arrary)
        width = len(arrary[0])
        if height > width:
            ratio = 50/float(height)
        else:
            ratio = 50/float(width)
        # print ratio
        img = cv2.resize(img, None, fx=ratio, fy=ratio)
        arrary = np.asarray(img, dtype="float32")
        height = len(arrary)
        width = len(arrary[0])
        a = np.empty((50, 50), dtype="float32")
        for i in range(50):
            for j in range(50):
                a[i][j] = 255
        # print a
        for i in range(height):
            for j in range(width):
                a[i][j] = arrary[i][j]

        cv2.imwrite('./test_img3_bi/'+imagename, a)


def threshold():
    for imagePath in glob.glob("./test_img3_bi/*.*"):
        imagePath.encode('utf-8')
        imagename = imagePath[imagePath.rfind("/") + 1:]
        img = cv2.imread(imagePath, 0)
        binary_adaptive = threshold_adaptive(img, 40, offset=10)
        arrary = np.asarray(binary_adaptive, dtype="int")
        for i in range(len(arrary)):
            for j in range(len(arrary[0])):
                if arrary[i][j] == 1:
                    arrary[i][j] = 255
                else:
                    arrary[i][j] = 0
        # print arrary
        cv2.imwrite('./test_img3_bi/'+imagename, arrary)

if __name__ == '__main__':
    print 'test'
    threshold()
    # img = get_data_test("./test_img/Audi奥迪_3.jpg")
    # # for i in img:
    # #     print i
    # print [img]
    # resize_img()

