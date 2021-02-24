from utils.utils import read_csv, rle_decode, rle_encode
from utils.config import cfg

import pandas as pd
import cv2


train_mask = pd.read_csv(cfg.PATH.TrainCSV, sep='\t', names=['name', 'mask'])
# 读取第一张图，并将对于的rle解码为mask矩阵
for i, itr in train_mask.iterrows():
    img = cv2.imread('DataSet/train/'+ itr['name'])
    mask = rle_decode(itr['mask'])
    mask = mask * 255
    cv2.imwrite('DataSet/train_mark/'+ itr['name'], mask)
    # 结果为True


