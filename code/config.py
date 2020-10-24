class DefaultConfig:
    backgroundFilePath = 'background2.tif'
    datasetPaths = ['datasets/Fluo-N2DL-HeLa/Sequence 1/', 'datasets/Fluo-N2DL-HeLa/Sequence 2/',
                    'datasets/Fluo-N2DL-HeLa/Sequence 3/', 'datasets/Fluo-N2DL-HeLa/Sequence 4/']
    # datasetPaths = ['datasets/PhC-C2DL-PSC/Sequence 1/', 'datasets/PhC-C2DL-PSC/Sequence 2/',
    #                 'datasets/PhC-C2DL-PSC/Sequence 3/', 'datasets/PhC-C2DL-PSC/Sequence 4/']

    output_path = 'output'
    grayDict = {'Fluo-N2DL-HeLa': 0,
                'PhC-C2DL-PSC': 40}
    divDict = {'Fluo-N2DL-HeLa': 143,
               'PhC-C2DL-PSC': 221}
    regenAllDets = True
    # adjust the grayThreshold/divThreshold according to corresponding dict
    grayThreshold = 40
    divThreshold = 220
    debug = True
    writeVideo_flag = True
    # the parameters following are set for the tracker
    # sigma_l, the threshold of the lowest confidence of detection (not used)
    # sigma_h, the threshold of the highest confidence of detection (not used)
    # sigma_iou, the threshold of the lowest IoU
    # t_min, the shortest length of cell's trajectory
    # max_missing, the total tolerant missing frames of a cell
    sigma_l = 0
    sigma_h = 0.5
    sigma_iou = 0.3
    t_min = 4
    max_missing = 10
    format = 'default'
    CONTOUR_AREA_MIN = 10
    CONTOUR_AREA_MAX = 4500
    nms = None
    mitosis_flag = True
    bbox_flag = True
    traj_flag = True
    data_flag = True
