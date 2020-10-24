import cv2
from config import DefaultConfig
import numpy as np

cfg = DefaultConfig()
mode = 'Fluo-N2DL-HeLa'

org = cv2.imread('./datasets/Fluo-N2DL-HeLa/Sequence 1/t013.tif', 0)
print(np.mean(org))

ret, th = cv2.threshold(org, np.mean(org) + cfg.grayDict[mode], 255, cv2.THRESH_TOZERO)
# find all cell contours in this image
contours, hierarchy = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

tn, fp = 0, 0

img2 = cv2.imread('./datasets/Fluo-N2DL-HeLa/Sequence 1 Masks/t013mask.tif', -1)
img1_array = np.array(org)
img2_array = np.array(img2)
image = img1_array * img2_array
img_O = cv2.normalize(image, None, 1, 255, norm_type=cv2.NORM_MINMAX)
img_O = img_O.astype(np.uint8)
ret2, th2 = cv2.threshold(img_O, np.mean(img_O) + cfg.grayDict[mode], 255, cv2.THRESH_TOZERO)
# cv2.imshow('th2', th2)
# find all cell contours in this image
contours2, hierarchy2 = cv2.findContours(th2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(len(contours) / len(contours2))


def iou(a, b):
    area_a = a[2] * a[3]
    area_b = b[2] * b[3]

    w = min(b[0] + b[2], a[0] + a[2]) - max(a[0], b[0])
    h = min(b[1] + b[3], a[1] + a[3]) - max(a[1], b[1])

    if w <= 0 or h <= 0:
        return 0

    area_c = w * h

    return area_c / (area_a + area_b - area_c)


bbox_1, bbox_2 = [], []
for i in range(len(contours)):
    x, y, w, h = cv2.boundingRect(contours[i])
    bbox_1.append([x, y, w, h, i])

for i in range(len(contours2)):
    x, y, w, h = cv2.boundingRect(contours2[i])
    bbox_2.append([x, y, w, h, i])

threshold = 0.9
cnt = 0

for i in range(len(bbox_2)):
    for j in range(len(bbox_1)):
        if iou(bbox_2[i], bbox_1[j]) > threshold:
            cv2.rectangle(org, (bbox_1[j][0], bbox_1[j][1]), 
                          (bbox_1[j][0] + bbox_1[j][2], bbox_1[j][1] + bbox_1[j][3]), (0, 0, 255), 1)
            cv2.rectangle(img_O, (bbox_2[i][0], bbox_2[i][1]),
                          (bbox_2[i][0] + bbox_2[i][2], bbox_2[i][1] + bbox_2[i][3]), (255, 255, 255), 1)
            cnt += 1
            break

print(cnt / len(contours2))

cv2.imshow('org', org)
cv2.imshow('gt', img_O)
cv2.waitKey(0)
cv2.destroyAllWindows()
