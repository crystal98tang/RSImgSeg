import csv
import numpy as np
import pandas as pd
import imageio
import os


def read_csv(path):
    """
    读取csv
    :param path: 标签csv文件
    :return: 返回list
    """
    lines = []
    with open(path, "r") as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            lines.append(row[0].split()[0])
    return lines

# 将图片编码为rle格式
def rle_encode(im):
    '''
    im: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = im.flatten(order = 'F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

# 将rle格式进行解码为图片
def rle_decode(mask_rle, shape=(512, 512)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    '''
    if not isinstance(mask_rle, float):
        s = mask_rle.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        starts -= 1
        ends = starts + lengths
        img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
        for lo, hi in zip(starts, ends):
            img[lo:hi] = 1
        return img.reshape(shape, order='F')
    else:
        img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
        return img.reshape(shape, order='F')


def testGenerator(imagelist, start_num, test_path, num_image):
    for i in range(start_num, start_num + num_image):
        img = np.array(imageio.imread(os.path.join(test_path, imagelist[i])))
        img = img / 255
        img = np.reshape(img, (1,) + img.shape)  # (1,256,256,3)
        yield img

def vary(img, th):
    img[img > th] = 1
    img[img < th] = 0
    return img