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


def save_info(dir_path):
    print('start saving')
    data_paths = np.array([x for x in glob(dir_path + f'/**/*.json', recursive=True)])
    df = pd.DataFrame(data_paths)
    print(df)
    df.columns = ['new_json_path']
    df['oks'] = round(0.0, 10)
    df.to_csv('json_info2.csv', index=False)
    print(df)


def main():
    parser = ArgumentParser()
    parser.add_argument('--dir-path', type=str, default='', help='Json Directory Path')

    args = parser.parse_args()

    kpt_idx_list = [1, 2, 4, 5, 7, 8, 16, 17, 18, 19, 20, 21]
    # object_area = 1920  # Area of the object in pixels

    # save_info(args.dir_path)

    df = pd.read_csv('json_info.csv')
    # df = pd.read_csv('json_info2.csv')
    # print(df)

    count = 0
    updated_rows = []

    update_df = pd.DataFrame(updated_rows, columns=['new_json_path', 'oks'])
    update_df.to_csv('json_info_updated.csv', index=False, mode='a')

    for data in tqdm(df['new_json_path']):
        count += 1
        try:
            with open(data, "r") as json_file:
                json_data = json.load(json_file)

                pred_keypoints = np.array(json_data['annotations']["new_2d_pos"])
                gt_keypoints = np.array(json_data['annotations']["2d_pos"])

                oks_score = calculate_oks(pred_keypoints, gt_keypoints, len(kpt_idx_list))

                updated_rows.append([data, oks_score])

        except Exception as e:
            print("\n\n--------------Exception--------------")
            print(e.args)
            print(e)
            print(f"j_path: {data}\n\n")
            updated_rows.append([data, -1.0])

        if count % 1000 == 0:
            update_df = pd.DataFrame(updated_rows, columns=['new_json_path', 'oks'])
            update_df.to_csv('json_info_updated.csv', index=False, mode='a', header=False)
            updated_rows = []

    if len(updated_rows) > 0:
        update_df = pd.DataFrame(updated_rows, columns=['new_json_path', 'oks'])
        update_df.to_csv('json_info_updated.csv', index=False, mode='a', header=False)


if __name__ == '__main__':
    main()
