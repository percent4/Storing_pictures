import cv2
import numpy as np

def transform(number):
    image_path = "E://CNN_DIGITS/digit-%s.png"%number
    img = cv2.imread(image_path, 0)
    img = 255-img

    ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    image, contours, hierarchy = cv2.findContours(thresh, 3, 2)
    cnt = contours[0]

    x, y, w, h = cv2.boundingRect(cnt)  # 外接矩形
    print((x,y,w,h))
    img = img[y:y+h, x:x+w]

    img = 255-img

    img = cv2.resize(img,(16,16),interpolation=cv2.INTER_AREA)

    img_new = []
    for row in img:
        for x in row:
            if x <= 250:
                img_new.append(0)
            else:
                img_new.append(255)

    img_new = np.array(img_new).reshape(16,16)

    cv2.imwrite("E://CNN_DIGITS/cnn_%s.png"%number, img_new)

def main():
    for i in range(10):
        transform(i)

main()