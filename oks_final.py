import os
from argparse import ArgumentParser
import json
from json import load
from collections import defaultdict
import numpy as np
from glob import glob
import pandas as pd
from tqdm import tqdm
from numba import jit
import csv
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt

@jit(nopython=True, cache=True)
def jit_calculate_distance(pred_keypoint, gt_keypoint):
    distance = 0.0
    for i in range(len(pred_keypoint)):
        diff = pred_keypoint[i] - gt_keypoint[i]
        distance += diff * diff
    return np.sqrt(distance)


def calculate_distance(pred_keypoint, gt_keypoint):
    return np.linalg.norm(pred_keypoint - gt_keypoint)


@jit(nopython=True, cache=True)
def calculate_oks(gt_keypoints, pred_keypoints, len_kpt):  # , object_area):
    total_distance = 0
    # num_keypoints = len(pred_keypoints)
    num_keypoints = len_kpt

    for i in range(num_keypoints):
        pred_keypoint = pred_keypoints[i]
        gt_keypoint = gt_keypoints[i]
        distance = jit_calculate_distance(pred_keypoint, gt_keypoint)
        total_distance += distance

    average_distance = total_distance / num_keypoints
    # oks = average_distance / np.sqrt(object_area)
    oks = average_distance

    return oks

def main():
    csv_path = '/workspace/cliff_path_info.csv'
    paths = []
    with open(csv_path, 'r', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) > 0:
                paths = row

    oneper = int(len(paths)/100)
    count = 0
    updated_rows = []
    df_oks = pd.DataFrame(updated_rows, columns=['json_path', 'SA','oks_gt_vit', 'oks_vit_clf', 'oks_diff'])
    df_oks.to_csv('/workspace/result/result_oks_final.csv', index=False, mode='a')

    for path in tqdm(paths):
        count += 1
        try:
            with open(path, "r") as json_file:
                json_data = json.load(json_file)

                gt_keypoints = np.array(json_data['annotations']['2d_pos'])
                vit_keypoints = np.array(json_data['annotations']['new_2d_pos'])
                clf_keypoints = np.array(json_data['annotations']['new_2d_pos_cliff'])

                vit_oks = calculate_oks(gt_keypoints, vit_keypoints, len(gt_keypoints))
                clif_oks = calculate_oks(vit_keypoints, clf_keypoints, len(gt_keypoints))
                updated_rows.append([path, path.rsplit('/', 2)[1],vit_oks, clif_oks, abs(vit_oks - clif_oks)])

        except Exception as e:
            print("\n\n--------------Exception--------------")
            print(e.args)
            print(e)
            print(f"j_path: {path}\n\n")
            updated_rows.append([path, path.rsplit('/', 2)[1],-1.0, -1.0, -1.0])

    df_oks = pd.DataFrame(updated_rows, columns=['json_path', 'SA','oks_gt_vit', 'oks_vit_clf', 'oks_diff'])
    df_oks = df_oks.sort_values('oks_diff',ascending=False).reset_index(drop=True)

    df_oks.to_csv('/workspace/result/result_oks_final.csv', index=False, mode='a', header=False)


def check_result():
    csv_path = '/workspace/result/result_oks_final.csv'
    df = pd.read_csv(csv_path)

    print(len(df))
    print(df.head())
    print(df.tail())


def vis_keypoints(json_data, mode='g'):
    vis_img = load_img(json_data['annotations']['img_path'])
    kps = json_data['annotations']['2d_pos']
    alpha = 1
    kps_lines = ((0, 1), (0,2), (0,3), (1,4), (2,5), (3,6), (4,7), (5,8), (6,9),
            (7,10), (8,11), (9,13), (9,14), (9,12), (13,16), (14,17), (12,15),
            (16,18), (17,19), (18,20), (19,21), (20,22), (21,23))
    img = np.copy(vis_img)
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.

    R = (0,0,255)
    G = (0,255,0)
    B = (255,0,0)
    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    # Draw the keypoints.
    for l in range(len(kps_lines)):
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]
        # print(type(kps[0, i1]))

        p1 = kps[i1][0], kps[i1][1]
        p2 = kps[i2][0], kps[i2][1]
        # p1 = kps[0, i1].to(torch.int32), kps[1, i1].to(torch.int32)
        # p2 = kps[0, i2].to(torch.int32), kps[1, i2].to(torch.int32)
        # if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
        cv2.line(
            kp_mask, p1, p2,
            color=R, thickness=4, lineType=cv2.LINE_AA)
        # if kps[2, i1] > kp_thresh:
        cv2.circle(
            kp_mask, p1,
            radius=5, color=R, thickness=-1, lineType=cv2.LINE_AA)
        cv2.putText(kp_mask, str(i1), p1, cv2.FONT_HERSHEY_PLAIN, 1, R, 1)
        # if kps[2, i2] > kp_thresh:
        cv2.circle(
            kp_mask, p2,
            radius=5, color=R, thickness=-1, lineType=cv2.LINE_AA)
        cv2.putText(kp_mask, str(i2), p2, cv2.FONT_HERSHEY_PLAIN, 1, R, 1)
    cv2.putText(kp_mask, 'GT_KPT', (20,30), cv2.FONT_HERSHEY_PLAIN, 2, R, 2)


    kps = json_data['annotations']['new_2d_pos_cliff']

    # Draw the keypoints.
    for l in range(len(kps_lines)):
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]
        # print(type(kps[0, i1]))

        p1 = kps[i1][0], kps[i1][1]
        p2 = kps[i2][0], kps[i2][1]
        # p1 = kps[0, i1].to(torch.int32), kps[1, i1].to(torch.int32)
        # p2 = kps[0, i2].to(torch.int32), kps[1, i2].to(torch.int32)
        # if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
        cv2.line(
            kp_mask, p1, p2,
            color=G, thickness=4, lineType=cv2.LINE_AA)
        # if kps[2, i1] > kp_thresh:
        cv2.circle(
            kp_mask, p1,
            radius=5, color=G, thickness=-1, lineType=cv2.LINE_AA)
        cv2.putText(kp_mask, str(i1), p1, cv2.FONT_HERSHEY_PLAIN, 1, G, 1)
        # if kps[2, i2] > kp_thresh:
        cv2.circle(
            kp_mask, p2,
            radius=5, color=G, thickness=-1, lineType=cv2.LINE_AA)
        cv2.putText(kp_mask, str(i2), p2, cv2.FONT_HERSHEY_PLAIN, 1, G, 1)
    cv2.putText(kp_mask, 'CLIFF_KPT', (20, 60), cv2.FONT_HERSHEY_PLAIN, 2, G, 2)


    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)


def load_img(img_path):
    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        if isinstance(img_path, bytes):
            img_path = img_path.decode('utf-8')
        # with open(img_path.decode('utf-8'), 'rb') as f:
        with open(img_path, 'rb') as f:
            byte = bytearray(f.read())
            nparr = np.asarray(byte, dtype=np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    else:
        return img_bgr
    if img_bgr is None:
        raise FileNotFoundError(f"/workspace/result/{img_path} is not found.")
    else:
        return img_bgr


def create_kptimg():
    csv_path = '/workspace/result/result_oks_final.csv'
    df = pd.read_csv(csv_path)
    sa_dict = defaultdict(list)
    # df_concat = df[~(df.duplicated('SA', keep='first') & df.duplicated('SA', keep='last'))].sort_values(['SA','oks_diff']).reset_index(drop=True)
    # print(df_concat)
    for i in df.index:
        # print(sa_dict[df.iloc[i]['SA']])
        # print(sa_dict[df.iloc[i]['SA']][0])
        # print(sa_dict[df.iloc[i]['SA']][1])
        if len(sa_dict[df.iloc[i]['SA']]) < 3:
            sa_dict[df.iloc[i]['SA']].append(i)

    idx = []
    for k,v in sa_dict.items():
        idx+=v

    for i in idx:
        with open(df.iloc[i]['json_path'], "r") as json_file:
            json_data = json.load(json_file)

        vis_img = vis_keypoints(json_data)
        cv2.imwrite(f"/workspace/result/{json_data['info']['action_category_id']}_{json_data['info']['actor_id']}_{i}.jpg", vis_img)

    # df_head = df.head(10)
    # df_mid = df.iloc[int(len(df)/2)-5:int(len(df)/2)+5]
    # df_tail = df.tail(10)
    #
    # df_concat = pd.concat([df_head,df_mid]).reset_index(drop=True)
    # df_concat = pd.concat([df_concat,df_tail]).reset_index(drop=True)

    # df_concat = pd.concat([df.iloc[0:2], df.iloc[5:7], df.iloc[20:22], df.iloc[45:47], df.iloc[131:133], df.iloc[377:379], df.iloc[874:876], df.iloc[1169:1171], df.iloc[1980:1982], df.iloc[4977:4979], df.iloc[34444:34446]]).reset_index(drop=True)
    # print(df_concat)
    # for i in df_concat.index:
    #     with open(df_concat.iloc[i]['json_path'], "r") as json_file:
    #         json_data = json.load(json_file)
    #
    # vis_img = vis_keypoints(json_data)
    # cv2.imwrite(f"/workspace/result/test_{i}.jpg", vis_img)

if __name__ == '__main__':
    # main()
    # check_result()
    create_kptimg()



