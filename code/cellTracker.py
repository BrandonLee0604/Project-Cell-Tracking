#!/usr/bin/env python

# ---------------------------------------------------------
# cellTracker
# All settings are in config.py, then run python cellTracker.py
# This script generates the cell tracking result as MOT format, which can be analysed by cellAnalysis.py
# ---------------------------------------------------------


import os
import cv2
import pandas as pd
from util import load_mot, save_to_csv, track_iou
from Detection import getDetections
from config import DefaultConfig

cfg = DefaultConfig()


def genDets(datasetPath, mode):
    detPath = datasetPath.replace(r'/', '_') + '.txt'
    # folder datasets road
    detAbsPath = os.path.join('det', detPath)

    if os.path.exists(detAbsPath) and not cfg.regenAllDets:
        return
    print('making detections...')
    imgPaths = os.listdir(datasetPath)
    x1, w, y1, h, types = [], [], [], [], []
    frs = []

    for i, path in enumerate(imgPaths):
        frame = cv2.imread(os.path.join(datasetPath, path), 0)
        dets, tx1, ty1, tw, th, type = getDetections(frame, mode)
        x1.extend(tx1)
        w.extend(tw)
        y1.extend(ty1)
        h.extend(th)
        types.extend(type)
        frs.extend([i + 1 for _ in range(0, len(tx1))])
    print("detections:", len(frs))
    df = pd.DataFrame({'fr': frs, 'id': frs, 'x1': x1, 'y1': y1, 'w': w, 'h': h, 'cls': types})
    df.to_csv(detAbsPath, index=False, header=False)


def main(datasetPath):
    if 'Fluo-N2DL-HeLa' in datasetPath:
        mode = 'Fluo-N2DL-HeLa'
    elif 'PhC-C2DL-PSC' in datasetPath:
        mode = 'PhC-C2DL-PSC'
    else:
        mode = 'DIC-C2DH-HeLa'
    detPath = datasetPath.replace(r'/', '_') + '.txt'
    detAbsPath = os.path.join('det', detPath)
    genDets(datasetPath, mode)
    print('loading detections...')
    detections = load_mot(detAbsPath, nms_overlap_thresh=cfg.nms, with_classes=False)
    print('tracking...')
    tracks = track_iou(detections, cfg.sigma_l, cfg.sigma_h, cfg.sigma_iou, cfg.t_min, cfg.max_missing)
    save_to_csv(os.path.join(cfg.output_path, detPath), tracks, fmt=cfg.format)


if __name__ == '__main__':
    for datasetPath in cfg.datasetPaths:
        main(datasetPath)
