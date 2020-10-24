import cv2
from config import DefaultConfig
import numpy as np

cfg = DefaultConfig()


def getDetections(frame, mode):
    """

    :param frame:
    :param mode:
    :return:
    dets: records the bounding box of each cell, [x, y, w, h]
    x1: records all x axis values
    y1: records all y axis values
    wList: records all width
    hList: records all height
    divList: records all elements, if it is dividing append 1 else 0
    """
    # convert the image into gray values
    ret, gray_frame = cv2.threshold(frame, np.mean(frame) + cfg.grayDict[mode], 255, cv2.THRESH_TOZERO)

    # find all cell contours in this image
    cnts, hierarchy = cv2.findContours(gray_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dets, x1, y1, wList, hList, divList, division_num = [], [], [], [], [], [], 0

    for c in cnts:
        # if area in [10, 4500]
        if cfg.CONTOUR_AREA_MIN < cv2.contourArea(c) < cfg.CONTOUR_AREA_MAX:
            # get coordinates
            (x, y, w, h) = cv2.boundingRect(c)
            # get the region of interests
            RoI = frame[y:y + h, x:x + w]

            if mode == 'PhC-C2DL-PSC':
                maxPix = np.mean(RoI)
            elif mode == 'Fluo-N2DL-HeLa':
                maxPix = RoI.max()
            else:
                maxPix = RoI.max()

            # if max pixel value > the pre-defined lightness threshold it is considered in mitosis
            if maxPix >= cfg.divDict[mode]:
                divList.append(1)
                division_num += 1
            else:
                divList.append(0)
            x1.append(x)
            y1.append(y)
            wList.append(w)
            hList.append(h)
            dets.append([x, y, w, h])
    if cfg.debug:
        print('dets num: %d, div num: %d' % (len(dets), division_num))
    return dets, x1, y1, wList, hList, divList


if __name__ == '__main__':
    frame = cv2.imread('datasets/Fluo-N2DL-HeLa/Sequence 2/t000.tif', 0)

    dets, a, b, c, d, mitosis = getDetections(frame, 'Fluo-N2DL-HeLa')
    print(len(dets))
    for i, det in enumerate(dets):
        print(i, det)
        pt1 = (int(det[0]), int(det[1]))
        pt2 = (int(det[0] + det[2]), int(det[1] + det[3]))
        pt3 = (int(det[0] + det[2] / 2), int(det[1] + det[3] / 2))
        # if is in dividing process, draw circle
        # else bounding box
        if mitosis[i] == 1:
            cv2.circle(frame, pt3, det[2], (255, 0, 0), 0)
        else:
            cv2.rectangle(frame, pt1, pt2, (255, 0, 0))
        cv2.putText(frame, str(i), pt1, 0, 5e-3 * 200, (255, 0, 0), 2)
    cv2.imshow('frame', frame)
    cv2.waitKey(0)
