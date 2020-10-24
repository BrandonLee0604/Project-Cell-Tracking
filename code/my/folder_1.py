import cv2
import numpy as np
from skimage import data, filters
from skimage import img_as_uint, img_as_float
import matplotlib.pyplot as plt

import numpy as np


def q2(img):

    frequency_list = [0 for _ in range(256)]

    img_1d = img.reshape(-1, 1)
    print(img_1d.shape)
    for i in range(img_1d.shape[0]):
        pixel_value = img_1d[i][0]
        frequency_list[pixel_value] += 1
    print(frequency_list)
    x_list = [i for _ in range(256)]
    plt.bar(x_list, frequency_list, width=1)
    plt.show()

img = cv2.imread('./datasets/DIC-C2DH-HeLa/t000.tif', cv2.IMREAD_GRAYSCALE)
blur = cv2.medianBlur(img, 15)
img2 = cv2.fastNlMeansDenoising(blur, None, 10, 7, 21)

#q2(img)
#q2(blur)
# hist,bins = np.histogram(img.flatten(),256,[0,256])
# cdf = hist.cumsum()
# cdf_normalized = cdf * hist.max()/ cdf.max()
# plt.plot(cdf_normalized, color = 'b')
# plt.hist(img.flatten(),256,[0,256], color = 'r')
# plt.xlim([0,256])
# plt.legend(('cdf','histogram'), loc = 'upper left')
# plt.show()
# cdf_m = np.ma.masked_equal(cdf,0)
# cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
# cdf = np.ma.filled(cdf_m,0).astype('uint8')
# img2 = cdf[img]
#
# cv2.imshow("ccc", img2)


equ = cv2.equalizeHist(img2)
q2(equ)
# cv2.imshow("bbb", equ)
img3 = filters.meijering(equ, sigmas=range(1, 10, 2), alpha=None, black_ridges=True)
cv2.imshow("aaa", img3)

img4 = np.array(img3) * 255
img4 = img4.astype(np.uint8)
#cv2.imwrite("aaa.png", img4)

ret3, th3 = cv2.threshold(img4, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# cv2.imshow("ddd", th3)
# morphological opening
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
img_binary = cv2.morphologyEx(th3, cv2.MORPH_OPEN, kernel)
cv2.imshow("eee", img_binary)
cv2.waitKey(0)
B = th3.copy()
for i in range(B.shape[0]):
    for j in range(B.shape[1]):
        if i == 0 or i == B.shape[0]-1 or j == 0 or j == B.shape[1]-1:
            continue
        else:
            B[i][j] = 255

new = B
cur = B.copy()
a = cv2.dilate(cur, (3, 3), 1)
cur = cv2.bitwise_and(a, th3)
print(cur, new)

while True:
    if (cur == new).all():
        break
    x = cv2.dilate(cur, (3, 3), 1)
    new = cv2.bitwise_and(x, th3)

cv2.imshow('complement', th3)
cv2.waitKey(0)
cv2.destroyAllWindows()
