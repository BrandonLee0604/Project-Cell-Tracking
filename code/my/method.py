import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


def contrast_stretching(img):
    """
    do contrast stretching on a image
    :param img:
    :return: matrix
    """
    a = 0
    b = 255
    c = img.min()
    d = img.max()
    result = img.copy()

    H = img.shape[0]
    W = img.shape[1]
    for i in range(H):
        for j in range(W):
            result[i][j] = (img[i][j] - c) * (b - a) / (d - c) + a

    return result

def img_sharpen(img_I):
    stardard_deviation = 1.0
    a = 1.25
    img_L = cv2.GaussianBlur(img_I, (3, 3), stardard_deviation)
    img_H = img_I.astype(np.int16) - img_L.astype(np.int16)
    img_O = img_I + img_H * a
    img_O = cv2.normalize(img_O, None, 0, 255, norm_type=cv2.NORM_MINMAX)
    img_O = img_O.astype(np.uint8)

    return img_O

def show(img, name=None):
    """
    display all the imgs and destroy all the windows.
    param img: a list with all imgs
    param name: a list with corresponding names
    """
    if name is None:
        name = []
    if not name:
        for i in img:
            cv2.imshow('img', i)
    else:
        for i in range(len(img)):
            cv2.imshow(name[i], img[i])
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_all_images(filedir):
    """
    get all images in a folder
    :param filedir:
    :return:
    """
    result = []
    for files in os.listdir(filedir):
        result.append(filedir + files)
    return result


def max_filter(img, kernel_size):
    """
    The implementation of max filter (with zero padding).
    :param img: gary value of an image
    :param kernel_size: the size of kernel (N*N)
    :return: max filtered matrix
    """
    H = img.shape[0]
    W = img.shape[1]

    padding = kernel_size // 2

    # bulit-in method in openCV to realize padding
    padding_img = cv2.copyMakeBorder(img, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=0)
    # print(padding_img.shape)

    # hand written padding method
    # padding_img = np.zeros((H+padding*2, W+padding*2), dtype=np.uint8)
    # padding_img[padding:padding+H, padding:padding+W] = img.copy().astype(np.uint8)

    result = padding_img.copy()

    # traverse the img and do the transformation
    for i in range(H):
        for j in range(W):
            # find max element in neighbours
            result[padding + i][padding + j] = np.max(padding_img[i: i + kernel_size, j: j + kernel_size])

    return result[padding: padding + H, padding: padding + W]


def min_filter(img, N):
    """
    The implementation of min filter (with 255 padding).
    :param img: image after max filtered
    :param N: the size of kernel used in max filter
    :return: min filtered matrix
    """
    H = img.shape[0]
    W = img.shape[1]

    padding = N // 2

    # bulit-in method in openCV to realize padding
    # ***** a change here is we must padding with constant 255
    padding_img = cv2.copyMakeBorder(img, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=255)
    # print(padding_img.shape)

    # hand written padding method
    # padding_img = np.ones((H+padding*2, W+padding*2), dtype=np.uint8) *\
    #         np.array([255]*(H+padding*2)*(W+padding*2), dtype=np.uint8).reshape(H+padding*2, W+padding*2)
    #  padding_img[padding:padding+H, padding:padding+W] = img.copy().astype(np.uint8)

    result = padding_img.copy()

    for i in range(H):
        for j in range(W):
            result[padding + i][padding + j] = np.min(padding_img[i: i + N, j: j + N])

    return result[padding: padding + H, padding: padding + W]


def sub_bound(image):
    """
    remove sells which partly in the image
    :param image:
    :return:
    """
    shape = np.shape(image)
    bound = []
    for i in range(shape[0]):
        for j in range(shape[1]):
            if i == 0 or j == 0 or i == shape[0] - 1 or j == shape[1] - 1:
                if image[i][j] == 255:
                    bound.append([i, j])
                    image[i][j] = 0
    flag = 0
    while flag == 0:
        len_0 = len(bound)
        for i in range(shape[0]):
            for j in range(shape[1]):
                if image[i][j] == 255 and i != 0 and i != shape[0] - 1 and j != 0 and j != shape[1] - 1:
                    if [i + 1, j] in bound or [i - 1, j] in bound or [i, j + 1] in bound or [i, j - 1] in bound:
                        image[i][j] = 0
                        bound.append([i, j])
        if len(bound) == len_0:
            flag = 1
    return image


def frequency_list(img):
    """
    find the frequency list of gray image
    :param img:
    :return:
    """
    frequency_list = [0 for _ in range(256)]

    img_1d = img.reshape(-1, 1)
    for i in range(img_1d.shape[0]):
        pixel_value = img_1d[i][0]
        frequency_list[pixel_value] += 1
    return frequency_list


def get_right_most(frequency):
    """
    get right most index of the frequency list
    :param frequency:
    :return:
    """
    index = 0
    f_sum = 0
    for i in range(len(frequency)):
        if frequency[i] != 0 and (f_sum / sum(frequency) <= 0.98):
            index = i
            f_sum += frequency[i]
    return index


if __name__ == '__main__':
    file_path = './datasets/Fluo-N2DL-HeLa/'
    print(get_all_images(file_path))