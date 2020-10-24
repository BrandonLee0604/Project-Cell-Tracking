from config import DefaultConfig
import cv2
import os
import numpy as np
import random

# ---------------------------------------------------------
# cellAnalysis
# This script analyses the cell tracking results and output a video/ all images to visualize the result.
# Other information of the cells are printed in console.
# ---------------------------------------------------------

cfg = DefaultConfig()

axis_x, axis_y = -1, -1


class Cell:
    """This class is designed for recording the information of cells. A new cell is initialized with a bbox.
     In new frames, a cell is updated when a new bbox is associated to it. The association algorithm is based on IoU."""

    def __init__(self, id, bbox):
        self.historyPos = []
        self.historyPos.append(bbox)
        self.id = id
        self.speed = 0
        self.netDis = 0
        self.totalDis = 0
        self.confRatio = 0
        self.color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        self.speed = 0

    def update(self, bbox):
        self.historyPos.append(bbox)
        self.update_speed()
        self.update_totalDis()
        self.update_netDis()
        self.update_confRatio()

    def get_historyPos(self):
        return self.historyPos

    def get_color(self):
        return self.color

    def update_speed(self):
        if len(self.historyPos) >= 2:
            x1, y1 = np.asarray(self.historyPos[-1][0:2])
            x2, y2 = np.asarray(self.historyPos[-2][0:2])
            dis = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            self.speed = dis
        else:
            self.speed = 0

    def update_totalDis(self):
        if len(self.historyPos) >= 2:
            x1, y1 = np.asarray(self.historyPos[-1][0:2])
            x2, y2 = np.asarray(self.historyPos[-2][0:2])
            dis = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            self.totalDis += dis
        else:
            self.totalDis = 0

    def update_netDis(self):
        if len(self.historyPos) > 2:
            x1, y1 = np.asarray(self.historyPos[-1][0:2])
            x0, y0 = np.asarray(self.historyPos[0][0:2])
            dis = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
            self.netDis = dis
        else:
            self.netDis = 0

    def update_confRatio(self):
        """
        if the denominator equals to zero, we will add 1 for both numerator and denominator
        :return:
        """
        if len(self.historyPos) > 2:
            if self.netDis == 0:
                self.confRatio = (self.totalDis + 1) / (self.netDis + 1)
            else:
                self.confRatio = self.totalDis / self.netDis
        else:
            self.confRatio = 0

    def __str__(self):
        return 'cell No.%d ->  speed: %.2f pixels/frame, totalDis: %.2f, netDis: %.2f, confinementRatio: %.2f' \
               % (self.id, self.speed, self.totalDis, self.netDis, self.confRatio)


def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    global axis_x, axis_y
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        axis_x, axis_y = x, y


def drawTrajectory(track, frame):
    historyPos = track.get_historyPos()
    # print(len(historyPos))
    if len(historyPos) >= 3:
        for d in range(1, len(historyPos)):
            pt1 = (int(historyPos[-d][0]), int(historyPos[-d][1]))
            pt2 = (int(historyPos[-d - 1][0]), int(historyPos[-d - 1][1]))
            cv2.line(frame, pt1, pt2, track.get_color(), 2, cv2.LINE_AA)
            if d + 2 == len(historyPos):
                break
    return frame


def task_3(x, y, cell):
    global axis_x, axis_y
    bbox_x, bbox_y, bbox_w, bbox_h = cell.historyPos[-1]
    if bbox_x <= x <= bbox_x + bbox_w and bbox_y <= y <= bbox_y + bbox_h:
        print(str(cell))
        axis_x, axis_y = -1, -1


def vis(datasetPath):
    # visualize the cell tracking results
    if 'Fluo-N2DL-HeLa' in datasetPath:
        mode = 'Fluo-N2DL-HeLa'
    elif 'PhC-C2DL-PSC' in datasetPath:
        mode = 'PhC-C2DL-PSC'
    detPath = datasetPath.replace(r'/', '_') + '.txt'
    detAbsPath = os.path.join('output', detPath)
    imgPaths = os.listdir(datasetPath)
    allDetections = np.loadtxt(detAbsPath, delimiter=',')
    print('All detections %d are loaded successfully!' % allDetections.shape[0])
    cellID = {}
    if cfg.writeVideo_flag:
        # Define the parameters of output video
        timg = cv2.imread(os.path.join(datasetPath, imgPaths[0]))
        shape = timg.shape
        w = shape[1]
        h = shape[0]
        # print(w, h)
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter('videos/' + detPath.replace('txt', 'avi'), fourcc, 15, (w, h))
        outImgDir = datasetPath.replace('datasets', 'output')
        # print(outImgDir)
        if not os.path.exists(outImgDir):
            os.makedirs(outImgDir)
    for i, path in enumerate(imgPaths):
        # process each frame
        if cfg.data_flag:
            print('----------------------frame %d -----------------------' % (i + 1))
        currentDets = allDetections[allDetections[:, 0] == i + 1, 1:7]
        currentDets[:, 3:5] += currentDets[:, 1:3]  # x1, y1, w, h -> x1, y1, x2, y2
        frame = cv2.imread(os.path.join(datasetPath, path), 0)
        ret, gray_frame = cv2.threshold(frame, np.mean(frame) + cfg.grayDict[mode], 255, cv2.THRESH_TOZERO)
        frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
        divCellNum = 0
        for d in currentDets:
            # d: id, x1, y1, x2, y2
            type = d[5]
            id = int(d[0])
            bbox = [int(d[1]), int(d[2]), int(d[3]), int(d[4])]
            if id not in cellID.keys():
                newCell = Cell(id, bbox)
                cellID[id] = newCell
            else:
                cellID[id].update(bbox)
            pt1 = (bbox[0], bbox[1])
            pt2 = (bbox[2], bbox[3])
            pt3 = (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))
            r = int((bbox[2] - bbox[0]))
            if type == 1:
                divCellNum += 1
                if cfg.mitosis_flag:
                    cv2.circle(frame, pt3, r, (0, 0, 255), 2)
            else:
                if cfg.bbox_flag:
                    cv2.rectangle(frame, pt1, pt2, cellID[id].get_color())

        for i in cellID.keys():
            cell = cellID[i]
            if cfg.traj_flag:
                drawTrajectory(cell, frame)
            if cfg.data_flag:
                cv2.namedWindow("image")
                cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
                # print(axis_x, axis_y)
                if axis_x != -1 and axis_y != -1:

                    task_3(axis_x, axis_y, cell)

        # draw the information needed
        # the 'dets' means the number of cells' detections in each frame
        # the 'cells' means the number of tracked cells (this indicator is always bigger than dets
        # , because there are many errors during tracking)
        cv2.putText(frame, 'frame' + str(i), (30, 30), 0, 5e-3 * 100, (0, 255, 0), 2)
        cv2.putText(frame, 'dets:' + str(currentDets.shape[0]), (30, 60), 0, 5e-3 * 100, (0, 255, 0), 2)
        cv2.putText(frame, 'divisions:' + str(divCellNum), (30, 90), 0, 5e-3 * 100, (0, 255, 0), 2)
        # cv2.putText(frame, 'cells:' + str(len(cellID.keys())), (30, 90), 0, 5e-3 * 100, (0, 255, 0), 2)

        cv2.imshow('image', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cfg.writeVideo_flag:
            out.write(frame)
            cv2.imwrite(os.path.join(outImgDir, path.replace('tif', 'jpg')), frame)


if __name__ == '__main__':
    for datasetPath in cfg.datasetPaths:
        vis(datasetPath)
