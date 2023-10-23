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
import os.path
from os import path

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
def calculate_oks(pred_keypoints, gt_keypoints, len_kpt):  # , object_area):
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


def save_info():
    print('start saving')
    data_paths1 = np.array([x for x in glob("/data/3d_pose_2023/train/2D_json/" + f'/**/*.json', recursive=True)])
    df_train = pd.DataFrame(data_paths1)
    df_train.columns = ['json_path']
    print(df_train)
    data_paths2 = np.array([x for x in glob("/data/3d_pose_2023/validation/2D_json/" + f'/**/*.json', recursive=True)])
    df_val = pd.DataFrame(data_paths2)
    df_val.columns = ['json_path']
    print(df_val)
    df_concat = pd.concat([df_train, df_val]).reset_index(drop=True)
    print(df_concat)
    df_train.to_csv('json_train.csv', index=False)
    df_val.to_csv('json_val.csv', index=False)
    df_concat.to_csv('json_concat.csv', index=False)

def main():
    # save_info()

    df = pd.read_csv('json_info.csv', usecols=['new_json_path'])
    df.columns = ['json_path']
    df_concat = pd.read_csv('json_concat.csv', usecols=['json_path'])
    df_concat['json_path'] = df_concat['json_path'].apply(lambda x: "/data/3d_pose_2023/2D_json_new/" + "/".join(x.split("/")[5:]))

    print(f"새로 추출한 json: {df.shape[0]}")
    print(f"train+val json: {df_concat.shape[0]}")
    df_merge = pd.merge(df, df_concat, how='outer', indicator=True)

    df_left_only = df_merge.query("_merge=='left_only'")
    print(f"새로 추출한 json에만 있는 값: {df_left_only.shape[0]}")

    df_right_only = df_merge.query("_merge=='right_only'")
    print(f"train+val json에만 있는 값: {df_right_only.shape[0]}")

    df_both = df_merge.query("_merge=='both'")
    print(f"둘다 있는 값: {df_both.shape[0]}")



    df_add_ext = pd.DataFrame(df_right_only["json_path"],columns=['json_path']).reset_index(drop=True)
    df_add_ext['json_path'] = df_add_ext['json_path'].str.replace('/2D_json_new/', '/2D_json/')

    df_add_ext.to_csv('missing_kpt_json.csv', index=False)
    df_add_ext = pd.read_csv('missing_kpt_json.csv')["json_path"]
    # df_add_ext.columns = ["path"]
    print(df_add_ext)
    img_list = []
    for data in df_add_ext:
        # print(data.split("/")[-1].replace(".json",".jpg"))

        img_list.append("/data/3d_pose_2023/Image/"+"_".join(data.split("/")[-1].split("_")[0:3]) + "/"+ data.split("/")[-1].replace(".json",".jpg"))
    print(img_list[0])
    df_img = pd.DataFrame(img_list,columns=["img_path"])
    print(df_img)
    df_img.to_csv('missing_kpt_img.csv', index=False)

if __name__ == '__main__':
    main()
