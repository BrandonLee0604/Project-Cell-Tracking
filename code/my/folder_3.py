import cv2
import numpy as np
import method


if __name__ == '__main__':
    file_dir = './datasets/PhC-C2DL-PSC/'
    imgs = method.get_all_images(file_dir)
    for file in imgs:
        img = cv2.imread(file, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # for N in range(1, 51, 2):
        #     A = min_filter(gray, N)
        #     B = max_filter(A, N)
        #     tmp = gray.astype(np.int16) - B.astype(np.int16)
        #     O = cv2.normalize(tmp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        #     cv2.imshow(f'N={N}', O)
        #     if cv2.waitKey(0) == ord('c'):
        #         continue
        #     elif cv2.waitKey(0) == ord('q'):
        #         break

        N = 11
        A = method.min_filter(gray, N)
        B = method.max_filter(A, N)
        tmp = gray.astype(np.int16) - B.astype(np.int16)
        O = cv2.normalize(tmp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        blur = cv2.GaussianBlur(O, (5, 5), 0)
        ret, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        contours, hierarchy = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, (255, 0, 0), 1)

        for i in range(0, len(contours)):
            x, y, w, h = cv2.boundingRect(contours[i])
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)

        cv2.imshow('contours', img)
        if cv2.waitKey(0) == ord('c'):
            continue
        elif cv2.waitKey(0) == ord('q'):
            break
    cv2.destroyAllWindows()
