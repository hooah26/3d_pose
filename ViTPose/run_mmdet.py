# Copyright (c) OpenMMLab. All rights reserved.
import os
import warnings
from argparse import ArgumentParser

from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         process_mmdet_results, vis_pose_result)
from mmpose.datasets import DatasetInfo
import pandas as pd
import json
import numpy as np
import copy
from tqdm import tqdm
import csv

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False


def main():
    det_model = init_detector(
        'demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py',
        'ckpt/det_checkpoint.pth',
        device='cuda:0'.lower())
    paths = []
    with open('/workspace/bbox_path.csv', 'r', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            paths = row

    bbox_thr = 0.3
    cat_id = 1

    for json_name in tqdm(paths):
        json_data = {}
        json_data_bkup = {}
        image_name=""
        try:
            with open(json_name, "r") as json_file:
                json_data = json.load(json_file)
                json_data_bkup = copy.deepcopy(json_data)
                image_name = "/data/3d_pose_2023/Image" + str(json_data["annotations"]["img_path"])
        except Exception as e:
            print(json_name)
            break
        try:
            mmdet_results = inference_detector(det_model, image_name)
            person_results = process_mmdet_results(mmdet_results, cat_id)

            # Change for-loop preprocess each bbox to preprocess all bboxes at once.
            bboxes = np.array([box['bbox'] for box in person_results])

            # Select bboxes by score threshold
            if bbox_thr is not None:
                assert bboxes.shape[1] == 5
                valid_idx = np.where(bboxes[:, 4] > bbox_thr)[0]
                person_results = [person_results[i] for i in valid_idx]

            # '''
            # {
            #     "image_id": 1877420, =>"img_no"
            #     "category_id": 1, =>cat_id
            #     "bbox": [       =>xywh
            #         309.1705017089844,
            #         252.84469604492188,
            #         326.1686096191406,
            #         368.1951599121094
            #     ],
            #     "score": 0.9997870326042175
            # }
            # '''

            # json_data["annotations"]["category_id"] = cat_id
            # json_data["annotations"]["bbox"] = person_results[0]["bbox"][:-1].tolist()
            # json_data["annotations"]["score"] = float(person_results[0]["bbox"][-1])


        #     # JSON파일 업데이트
        #     with open(json_name, 'w', encoding='utf-8') as make_file:
        #         json.dump(json_data, make_file, indent="\t")
        except Exception as e:
            print(json_name)
            # with open(json_name, 'w', encoding='utf-8') as make_file:
            #     json.dump(json_data_bkup, make_file, indent="\t")




if __name__ == '__main__':
    main()

    # import numpy as np
    # import cv2
    # import pandas as pd
    # import json
    #
    # img = cv2.imread("/data/3d_pose_2023/Image/02_F170B_1/02_F170B_1_0.jpg")
    #
    # # [8.43569214e+02, 2.31143600e+02, 1.00981635e+03, 8.30502258e+02,
    # #  9.99590099e-01]
    # x_min = int(8.43569214e+02)
    # y_min = int(2.31143600e+02)
    # x_max = int(1.00981635e+03)
    # y_max = int(8.30502258e+02)
    # #
    # # # 바운딩 박스 그리기
    # cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    #
    # # cv2.imshow('image', img)
    # # cv2.imshow('image2', img2)
    # # cv2.waitKey()
    # # cv2.destroyAllWindows()
    # cv2.imwrite('test1234.jpg', img)
    # # cv2.imwrite('img2.png',img2)
