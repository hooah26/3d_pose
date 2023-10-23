from __future__ import print_function
import numpy as np
import cv2
import time
from collections import OrderedDict
import json

def open_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as fp:
        json_data = json.load(fp, object_pairs_hook=OrderedDict)

    return json_data

def run_projection_nia2020(cam_path, kpt2d_path, kpt3d_path):
    cam = open_json(cam_path)
    kpt2d_data = open_json(kpt2d_path)
    kpt3d_data = open_json(kpt3d_path)

    intrinsic = cam['intrinsics']
    extrinsic = cam['extrinsics']
    kpt_2d = kpt2d_data['annotations']['2d_pos']
    kpt_3d = kpt3d_data['annotations']['3d_pos']

    # extrinsic = np.linalg.inv(np.append(extrinsic, np.array([[0, 0, 0, 1]]), axis=0))
    # P = np.array(intrinsic) @ np.array(extrinsic[:3, :])

    P = np.array(intrinsic) @ np.array(extrinsic)

    kpt_P = projection_nia2020(P, kpt_3d)
    kpt_P = np.array(kpt_P) * 1920

    print('gt 2d: ', kpt_2d)
    print('projection 2d:', kpt_P)

def run_projection_nia2022(json_path):
    json_data = open_json(json_path)

    intrinsic = json_data['info']['camera']['intrinsic']
    extrinsic = json_data['info']['camera']['extrinsic']
    kpt_2d = json_data['annotation']['actor']['keypoint']['2d']
    kpt_3d = json_data['annotation']['actor']['keypoint']['3d']

    # extrinsic = np.linalg.inv(np.append(extrinsic, np.array([[0, 0, 0, 1]]), axis=0))
    # P = np.array(intrinsic) @ np.array(extrinsic[:3, :])

    P = np.array(intrinsic) @ np.array(extrinsic)

    kpt_P = projection_nia2022(P, kpt_3d)

    print('gt 2d: ', kpt_2d)
    print('projection 2d:', kpt_P)

# project 3D keypoints to camera image plane
def projection_nia2020(P, pt_3d):
    pjt_2d = []
    for pt in pt_3d:
        pt = sum(pt, [])
        p = P @ pt
        p /= p[2]
        pjt_2d.append(p[:2].tolist())
    return pjt_2d

def projection_nia2022(P, pt_3d):
    pjt_2d = []
    for pt in pt_3d:
        pt = pt + [1.0]
        p = P @ pt
        p /= p[2]
        pjt_2d.append(p[:2].tolist())
    return pjt_2d

if __name__ == '__main__':
    # nia2020
    # cam_path = '/Users/lee/Downloads/Camera_json/15_F150C_4.json'
    # kpt2d_path = '/Users/lee/Downloads/010.사람인체자세3D/2.Validation/라벨링데이터_230714_add/2D_json/15_F150C/15_F150C_4_0.json'
    # kpt3d_path = '/Users/lee/Downloads/15_F150C/3D_15_F150C_0.json'

    cam_path = '/Users/lee/Downloads/Camera_json/60_M160B_1.json'
    kpt2d_path = '/Users/lee/Downloads/010.사람인체자세3D/2.Validation/라벨링데이터_230714_add/2D_json/60_M160B/60_M160B_1_60.json'
    kpt3d_path = '/Users/lee/Downloads/60_M160B/3D_60_M160B_60.json'
    run_projection_nia2020(cam_path, kpt2d_path, kpt3d_path)

    # nia2022
    # json_path = '/Users/lee/Downloads/4.Sample/2.라벨링데이터/상호작용데이터/여가 시간/게임 및 놀이/I_F001_T001-T01-00_T019-T02-00_B11883_L.json'
    # run_projection_nia2022(json_path)
