from __future__ import print_function
import numpy as np
import cv2
import time
# from utils.display_utils import *
from collections import OrderedDict
import json
import os
import pandas as pd
from glob import glob
from tqdm import tqdm
from collections import defaultdict
import csv


os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

np.set_printoptions(formatter={'float': '{: 0.3f}'.format})


def run_triangulation(gt_dataset, kpt_flag):
    """
    input:
            gt_dataset: contained ground truth 2d keypoints, camera intrinsic parameter, camera extrinsic parameter
    output:
            result: contained 3d keypoints from triangulation, 2d keypoints reprojected to 3 cameras
    """
    # get camera data
    # print(json.dumps(gt_dataset, ensure_ascii=False, indent=3))
    cam_L = gt_dataset['camera_L']
    cam_C = gt_dataset['camera_C']
    cam_R = gt_dataset['camera_R']

    '''
    # Take an inverse matrix for extinsic parameter
    L_extrinsic = np.linalg.inv(np.append(cam_L[2], np.array([[0, 0, 0, 1]]), axis=0))
    C_extrinsic = np.linalg.inv(np.append(cam_C[2], np.array([[0, 0, 0, 1]]), axis=0))
    R_extrinsic = np.linalg.inv(np.append(cam_R[2], np.array([[0, 0, 0, 1]]), axis=0))
    # make projection matrix to matrix multiplication intrinsic and extrinsic
    L_P = np.array(cam_L[1]) @ L_extrinsic[:3, :]
    C_P = np.array(cam_C[1]) @ C_extrinsic[:3, :]
    R_P = np.array(cam_R[1]) @ R_extrinsic[:3, :]
    '''

    L_P = np.array(cam_L[1]) @ np.array(cam_L[2])
    C_P = np.array(cam_C[1]) @ np.array(cam_C[2])
    R_P = np.array(cam_R[1]) @ np.array(cam_R[2])

    # start triangulation with two projection matrix and two 2d keypoints
    # kpt1 = cv_triangulation(L_P, C_P, np.array(cam_L[0]), np.array(cam_C[0]))
    # kpt2 = cv_triangulation(L_P, R_P, np.array(cam_L[0]), np.array(cam_R[0]))
    # kpt3 = cv_triangulation(C_P, R_P, np.array(cam_C[0]), np.array(cam_R[0]))

    rescaled_cam_L = [[float(element)/1920 for element in inner_list] for inner_list in cam_L[0]]
    rescaled_cam_C = [[float(element)/1920 for element in inner_list] for inner_list in cam_C[0]]
    rescaled_cam_R = [[float(element)/1920 for element in inner_list] for inner_list in cam_R[0]]
    kpt1 = cv_triangulation(L_P, C_P, np.array(rescaled_cam_L), np.array(rescaled_cam_C))
    kpt2 = cv_triangulation(L_P, R_P, np.array(rescaled_cam_L), np.array(rescaled_cam_R))
    kpt3 = cv_triangulation(C_P, R_P, np.array(rescaled_cam_C), np.array(rescaled_cam_R))

    # mean three 3d keypoints
    kpt_3d = np.mean([kpt1, kpt2, kpt3], axis=0)

    if kpt_flag:
        # get three 2d keypoints reprojected 3d keypoints
        joint2d = []
        joint2d.append(projection(L_P, kpt_3d))
        joint2d.append(projection(C_P, kpt_3d))
        joint2d.append(projection(R_P, kpt_3d))
        result = {'joint_2d': joint2d}
    else:
        result = {'joint_3d': kpt_3d[:, :3]}

    return result


# run opencv triangulation function to obtain 3D keypoints
def cv_triangulation(P1, P2, x1, x2):
    p1, p2 = x1.T, x2.T
    cvt = cv2.triangulatePoints(P1, P2, p1, p2)
    cvt /= cvt[3]
    return cvt.T


# project 3D keypoints to camera image plane
def projection(P, pt_3d):
    pjt_2d = []
    for pt in pt_3d:
        p = P @ pt
        p /= p[2]
        pjt_2d.append(p[:2].tolist())
    return pjt_2d


def dataloader():
    print("data loading...")
    df_3cam = pd.read_csv('ViTPose/ext_10per_3cam.csv')[["SA", "SAC"]]
    df_jpath = pd.read_csv('ViTPose/thd_under_sorted.csv')[["json_path", "SA", "SAC"]]

    print(df_3cam)
    print(df_jpath)

    df_merge = pd.merge(df_jpath, df_3cam["SAC"], on="SAC").sort_values(by='json_path', ignore_index=True)
    print(df_merge)

    info_dict = defaultdict(lambda:defaultdict(list))

    for idx, row in tqdm(df_merge.iterrows()):
        info_dict[row["SA"]][row["SAC"]].append(row["json_path"])

    result_list = []

    for sa, sac in tqdm(info_dict.items()):
        # print(sac.keys())
        # print(list(sac.keys())[0], len(sac[list(sac.keys())[0]]), list(sac.keys())[1], len(sac[list(sac.keys())[1]]), list(sac.keys())[2], len(sac[list(sac.keys())[2]]))

        min_sac = min(sac, key=lambda k: len(sac[k]) if isinstance(sac[k], list) else float('inf'))
        # print(sac[min_sac])
        for path in sac[min_sac]:
            try:
                temp_dict = {}
                temp = []

                path_rm_cam = path.rsplit('_', 2)[0] + "_" + path.rsplit('_', 2)[-1]
                path_rsplit = path_rm_cam.rsplit('/', 1)[0] + "/3D_" + path_rm_cam.rsplit('/', 1)[1]
                path_ori = path_rsplit.replace("2D_json_new", "3D_json")
                path_new = path_rsplit.replace("2D", "3D")
                temp_dict["path"] = [path_ori, path_new]

                with open(path.replace(min_sac, list(sac.keys())[0]), "r") as json_file:
                    json_data = json.load(json_file)
                    temp.append(json_data["annotations"]["new_2d_pos"])
                with open(f"/data/3d_pose_2023/Camera_json/{list(sac.keys())[0]}.json", "r") as json_file:
                    camera_data = json.load(json_file)
                    temp.append(camera_data['intrinsics'])
                    temp.append(camera_data['extrinsics'])
                temp_dict["camera_L"] = temp

                temp = []
                with open(path.replace(min_sac, list(sac.keys())[1]), "r") as json_file:
                    json_data = json.load(json_file)
                    temp.append(json_data["annotations"]["new_2d_pos"])
                with open(f"/data/3d_pose_2023/Camera_json/{list(sac.keys())[1]}.json", "r") as json_file:
                    camera_data = json.load(json_file)
                    temp.append(camera_data['intrinsics'])
                    temp.append(camera_data['extrinsics'])
                temp_dict["camera_C"] = temp

                temp = []
                with open(path.replace(min_sac, list(sac.keys())[2]), "r") as json_file:
                    json_data = json.load(json_file)
                    temp.append(json_data["annotations"]["new_2d_pos"])
                with open(f"/data/3d_pose_2023/Camera_json/{list(sac.keys())[2]}.json", "r") as json_file:
                    camera_data = json.load(json_file)
                    temp.append(camera_data['intrinsics'])
                    temp.append(camera_data['extrinsics'])
                temp_dict["camera_R"] = temp


                result_list.append(temp_dict)
            except Exception as e:
                # print(e)
                continue

    return result_list


def update_3d_keypoint(gt_dataset, kpt_flag):
    print("Updating 3D keypoints...")
    save_path = []
    for data in tqdm(gt_dataset):
        result = run_triangulation(data, kpt_flag)
        if not kpt_flag:
            try:
                with open(data['path'][0], "r") as json_file:
                    json_data = json.load(json_file)

                    # json_data['annotations']["new_3d_pos"] = result["joint_3d"].tolist()
                    new_3d_pos = []
                    for kpt in result["joint_3d"].tolist():
                        temp = []
                        for c in kpt:
                            temp.append([round(c, 4)])
                        temp.append([1.0])
                        new_3d_pos.append(temp)
                    json_data['annotations']["new_3d_pos"] = new_3d_pos

                    dir_path = data['path'][1].rsplit("/", 1)[0]
                    if not os.path.exists(dir_path):
                        os.makedirs(dir_path)
                    # JSON파일 업데이트
                    with open(data['path'][1], 'w', encoding='utf-8') as make_file:
                        json.dump(json_data, make_file, indent="\t")
                        save_path.append(data['path'][1])
            except Exception as e:
                print(e)
    with open('new_3d_pos_paths.csv', 'w', newline='') as f: csv.writer(f).writerow(save_path)


if __name__ == '__main__':
    gt_dataset = dataloader()
    with open('gt_dataset.csv', 'w', newline='') as f: csv.writer(f).writerow(gt_dataset)
    update_3d_keypoint(gt_dataset, False)
