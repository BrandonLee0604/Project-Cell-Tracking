# 我们需要降噪+除去只有部分在img里的cells
import cv2
import numpy as np
import my.method as method


def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        print(x, y)
        # cv2.circle(img, (x, y), 1, (0, 0, 255), thickness=-1)
        # cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
        #             1.0, (0, 0, 0), thickness=1)
        # cv2.imshow("image", img)


if __name__ == '__main__':
    # get all images
    file_dir = 'E:/proj/datasets/Fluo-N2DL-HeLa/Sequence 1/'
    imgs = method.get_all_images(file_dir)

    for file in imgs:
        # read the image
        img = cv2.imread(file, 1)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # applied contrast stretching
        O = method.contrast_stretching(gray)
        # get frequency list of intensity histrogram
        frequency = method.frequency_list(O)
        # get peak value's index
        index_of_max = frequency.index(max(frequency))
        index_of_right_most_in_Gau = method.get_right_most(frequency)

        threshold = (index_of_max + index_of_right_most_in_Gau) // 2

        # Gaussian blur, then thresholding
        blur = cv2.GaussianBlur(O, (5, 5), 0)
        ret, th = cv2.threshold(blur, threshold + 1, 255, cv2.THRESH_BINARY)

        # remove cells partly in the image
        # after_remove_partly = method.sub_bound(th)
        # cv2.imshow('before', th)

        # get contours
        contours, hierarchy = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # compute area
        area = np.array([cv2.contourArea(i) for i in contours])
        # assume the area follows Gaussian distribution
        mean = np.mean(area)
        std = np.std(area, ddof=1)

        print(sorted(area), '\d', mean, '\d', std)
        shave = []
        for i in range(len(area)):
            if area[i] >= mean - std:
                shave.append(contours[i])
        print(len(contours), len(shave))
        cv2.drawContours(img, shave, -1, (255, 255, 255), 1)

        # draw bbox
        for i in range(0, len(shave)):
            x, y, w, h = cv2.boundingRect(shave[i])
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)

        cv2.namedWindow("contours")
        cv2.setMouseCallback("contours", on_EVENT_LBUTTONDOWN)
        cv2.imshow('contours', img)
        if cv2.waitKey(0) == ord('c'):
            continue
        elif cv2.waitKey(0) == ord('q'):
            break

    cv2.destroyAllWindows()
