import numpy as np
import pandas as pd
from PIL import Image
import scipy.io as sio
import os

def X2Cube(img,B=[4, 4],skip = [4, 4],bandNumber=16):
    # Parameters
    M, N = img.shape
    col_extent = N - B[1] + 1
    row_extent = M - B[0] + 1
    # Get Starting block indices
    start_idx = np.arange(B[0])[:, None] * N + np.arange(B[1])
    # Generate Depth indeces
    didx = M * N * np.arange(1)
    start_idx = (didx[:, None] + start_idx.ravel()).reshape((-1, B[0], B[1]))
    # Get offsetted indices across the height and width of input array
    offset_idx = np.arange(row_extent)[:, None] * N + np.arange(col_extent)
    # Get all actual indices & index into input array for final output
    out = np.take(img, start_idx.ravel()[:, None] + offset_idx[::skip[0], ::skip[1]].ravel())
    out = np.transpose(out)
    DataCube = out.reshape(M//4, N//4,bandNumber )
    return DataCube


def region_to_bbox(region, center=True):
    n = len(region)
    assert n == 4 or n == 8, ('GT region format is invalid, should have 4 or 8 entries.')

    if n == 4:
        return _rect(region, center)
    else:
        return _poly(region, center)


# we assume the grountruth bounding boxes are saved with 0-indexing
def _rect(region, center):
    if center:
        x = region[0]
        y = region[1]
        w = region[2]
        h = region[3]
        cx = x + w / 2
        cy = y + h / 2
        return np.array([cx, cy, w, h])
    else:
        # region[0] -= 1
        # region[1] -= 1
        return region


def _poly(region, center):
    cx = np.mean(region[::2])
    cy = np.mean(region[1::2])
    x1 = np.min(region[::2])
    x2 = np.max(region[::2])
    y1 = np.min(region[1::2])
    y2 = np.max(region[1::2])
    A1 = np.linalg.norm(region[0:2] - region[2:4]) * np.linalg.norm(region[2:4] - region[4:6])
    A2 = (x2 - x1) * (y2 - y1)
    s = np.sqrt(A1 / A2)
    w = s * (x2 - x1) + 1
    h = s * (y2 - y1) + 1

    if center:
        return np.array([cx, cy, w, h])
    else:
        return np.array([cx - w / 2, cy - h / 2, w, h])

def _compute_distance(boxA, boxB):
    a = np.array((boxA[0]+boxA[2]/2, boxA[1]+boxA[3]/2))
    b = np.array((boxB[0]+boxB[2]/2, boxB[1]+boxB[3]/2))
    dist = np.linalg.norm(a - b)

    assert dist >= 0
    assert dist != float('Inf')

    return dist

def overlap_ratio(rect1, rect2):
    '''Compute overlap ratio between two rects
    Args
        rect:2d array of N x [x,y,w,h]
    Return:
        iou
    '''
    # if rect1.ndim==1:
    #     rect1 = rect1[np.newaxis, :]
    # if rect2.ndim==1:
    #     rect2 = rect2[np.newaxis, :]
    left = np.maximum(rect1[:,0], rect2[:,0])
    right = np.minimum(rect1[:,0]+rect1[:,2], rect2[:,0]+rect2[:,2])
    top = np.maximum(rect1[:,1], rect2[:,1])
    bottom = np.minimum(rect1[:,1]+rect1[:,3], rect2[:,1]+rect2[:,3])

    intersect = np.maximum(0,right - left) * np.maximum(0,bottom - top)
    union = rect1[:,2]*rect1[:,3] + rect2[:,2]*rect2[:,3] - intersect
    iou = intersect / union
    iou = np.maximum(np.minimum(1, iou), 0)
    return iou

def success_overlap(gt_bb, result_bb, n_frame):
    thresholds_overlap = np.arange(0.02, 1.02, 0.02)
    success = np.zeros(len(thresholds_overlap))
    iou = np.ones(len(gt_bb)) * (-1)
    mask = np.sum(gt_bb > 0, axis=1) == 4
    iou[mask] = overlap_ratio(gt_bb[mask], result_bb[mask])
    for i in range(len(thresholds_overlap)):
        success[i] = np.sum(iou >= thresholds_overlap[i]) / float(n_frame)
    return success
def compile_results(gt, bboxes):
    l = np.size(bboxes, 0)
    gt4 = np.zeros((l, 4))
    new_distances = np.zeros(l)
    n_thresholds = 50
    ops = np.zeros(n_thresholds)
    distance_thresholds = np.linspace(1,50,50)
    dp_20 = np.zeros(50)
    precision=dp_20
    for i in range(l):
        gt4[i, :] = region_to_bbox(gt[i, :], center=False)
        new_distances[i] = _compute_distance(bboxes[i, :], gt4[i, :])
    for i in range(50):
        precision[i] =np.float(sum(new_distances < distance_thresholds[i])) / np.size(new_distances)
    dp_20 = precision[19]
    average_center_location_error =new_distances.mean()
    # what's the percentage of frame in which center displacement is inferior to given threshold? (OTB metric)

    success=success_overlap(gt4, bboxes, l)
    # integrate over the thresholds
    auc = np.mean(success)
    return precision,success,average_center_location_error, auc,dp_20

# if __name__ == '__main__':
#     gt = pd.read_table('groundtruth_rect.txt',header=None)
#     gt=gt.dropna(axis=1, how='all').to_numpy()
#     res = pd.read_table('ballMHTNJUST_0.0023.txt', header=None)
#     res = res.dropna(axis=1, how='all').to_numpy()
#     dp, op, cle, auc, dp_20 = compile_results(gt, res)
if __name__ == '__main__':
    # main_path = r'D:\HSI\whispers\test'
    # save_path_main = r'C:\Users\chenlangkun\Desktop\TransT-main\testing_dataset\HSI16'
    main_path = r'/amax/GL/clk/HSI/train'
    save_path_main = r'/amax/GL/clk/HSI16/train'
    dirlist = os.listdir(main_path)
    rem = []
    for dir in dirlist:
        if '.' in dir:
            rem.append(dir)
    for ite in rem:
        dirlist.remove(ite)
    # print(dirlist)
    paths = [os.path.join(main_path, x)+'/HSI' for x in dirlist]
    save_paths = [os.path.join(save_path_main, x) for x in dirlist]
    # print(paths)
    # print(save_paths)
    for i , path in enumerate(paths):
        img_paths = os.listdir(path)
        img_pathx = []
        mat_path = []
        for img_path in img_paths:
            if '.txt' in img_path:
                pass
            else:
                img_pathx.append(img_path)
                mat_path.append(img_path.split('.')[0]+'.mat')
        imgs = [os.path.join(path, x) for x in img_pathx]
        mat_save_paths = [os.path.join(save_paths[i], x) for x in mat_path]
        # print(imgs)
        # print(mat_save_paths)
        for j , image in enumerate(imgs) :
            if not os.path.isdir(save_paths[i]):
                os.makedirs(save_paths[i])
            img = np.array(Image.open(image))
            # print(img)
            matdata = X2Cube(img)
            sio.savemat(mat_save_paths[j], {'image':matdata})
            # print(matdata)
            # break
            # pass
        print(path+' over')
        # if i ==2:
        #     break
    # ------------- testing
    # img_path = '0001.png'
    # img = np.array(Image.open(img_path))
    # print(img)
    # a = X2Cube(img)
    # sio.savemat('0001.mat', {'image':a})
    # print(a)
