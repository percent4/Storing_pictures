import cv2
import numpy as np
from TF_CNN import CNN

# 模型保存地址
MODEL_SAVE_PATH = 'E://logs/cnn_digits.ckpt'
# CNN初始化
cnn = CNN(2000, 0.0001, MODEL_SAVE_PATH)

def reg(number):

    image_path = "E://CNN_DIGITS/cnn_%d.png"%number
    img = cv2.imread(image_path, 0)

    vec = np.array([1-int(x/255) for x in img.ravel()]).reshape(1, 256)
    # print(vec)
    pred = cnn.predict(vec)
    reg_result = list(pred[0]).index(max(pred[0]))
    #print(reg_result)

    return reg_result

def main():

    res = []
    for i in range(10):
        res.append(reg(i))

    print(res)

main()