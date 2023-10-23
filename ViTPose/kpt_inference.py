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
from glob import glob
import datetime

from mmdet.apis import inference_detector, init_detector
# try:
#     from mmdet.apis import inference_detector, init_detector
#     has_mmdet = True
# except (ImportError, ModuleNotFoundError):
#     has_mmdet = False


def main():
    """Visualize the demo images.

    Using mmdet to detect the human.
    """
    parser = ArgumentParser()
    # parser.add_argument('det_config', help='Config file for detection')
    # parser.add_argument('det_checkpoint', help='Checkpoint file for detection')
    # parser.add_argument('pose_config', help='Config file for pose')
    # parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')
    parser.add_argument('--img-root', type=str, default='', help='Image root')
    # parser.add_argument('--img', type=str, default='', help='Image file')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show img')
    # parser.add_argument(
    #     '--out-img-root',
    #     type=str,
    #     default='',
    #     help='root of the output img file. '
    #     'Default not saving the visualization images.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=1,
        help='Category id for bounding box detection model')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.3,
        help='Bounding box score threshold')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
    parser.add_argument(
        '--radius',
        type=int,
        default=4,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')

    parser.add_argument('--loop', type=str, default='0/1', help='Image root')

    # assert has_mmdet, 'Please install mmdet to run the demo.'

    args = parser.parse_args()
    args.det_config = 'demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py'
    # faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
    args.det_checkpoint = 'ckpt/det_checkpoint.pth'

    # args.pose_config = 'configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/hrnet_w48_coco_wholebody_384x288_dark_plus.py'
    # # hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth
    # args.pose_checkpoint = 'ckpt/wholebody_pose_checkpoint.pth'


    args.pose_config = 'configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py'
    # hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth
    args.pose_checkpoint = 'ckpt/pose_checkpoint.pth'


    # assert args.show or (args.out_img_root != '')
    # assert args.img != ''
    assert args.det_config is not None
    assert args.det_checkpoint is not None
    det_model = init_detector(
        'demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py',
        'ckpt/det_checkpoint.pth',
        device='cuda:0'.lower())
    # det_model = init_detector(
    #     args.det_config, args.det_checkpoint, device=args.device.lower())
    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.device.lower())

    dataset = pose_model.cfg.data['test']['type']
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)

    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)
    image_name = '/workspace/ViTPose/MicrosoftTeams-image.png'
    # test a single image, the resulting box is (x1, y1, x2, y2)
    mmdet_results = inference_detector(det_model, image_name)

    # keep the person class bounding boxes.
    person_results = process_mmdet_results(mmdet_results, args.det_cat_id)

    pose_results, returned_outputs = inference_top_down_pose_model(
        pose_model,
        image_name,
        person_results,
        bbox_thr=args.bbox_thr,
        format='xyxy',
        dataset=dataset,
        dataset_info=dataset_info,
        return_heatmap=return_heatmap,
        outputs=output_layer_names)

    out_file = '/workspace/ViTPose/MicrosoftTeams-image_result.png'

    # show the results
    vis_pose_result(
        pose_model,
        image_name,
        pose_results,
        dataset=dataset,
        dataset_info=dataset_info,
        kpt_score_thr=args.kpt_thr,
        radius=args.radius,
        thickness=args.thickness,
        show=args.show,
        out_file=out_file)

    # dir_path = args.img_root
    # # 최종 검토
    # org_kpt_idx_list = [1, 2, 4, 5, 7, 8, 16, 17, 18, 19, 20, 21]
    # new_kpt_idx_list = [11, 12, 13, 14, 15, 16, 5, 6, 7, 8, 9, 10]
    # # data_paths = np.array([x for x in glob(dir_path + f'/**/*.jpg', recursive=False)])[int(args.roof.split('/')[0]):int(args.roof.split('/')[1])]
    # data_paths = pd.read_csv('image_paths.csv')['img_path'][int(args.roof.split('/')[0]):int(args.roof.split('/')[1])]
    # total_count=len(data_paths)
    # print(total_count)
    #
    # count = 0
    #
    # # for (root, directories, files) in os.walk(dir_path):
    # #     for file in files:
    # for data in data_paths:
    #     count += 1
    #     if count % 100 == 1:
    #         print(f"{count}/{total_count} - {round((count/total_count)*100,3)}%")
    #
    #     root = dir_path + data.split('.')[0].split('/')[4] #/data/3d_pose_2023/Image/01_F160B_2
    #     file = data.split('/')[-1] # 01_F160B_2_9.jpg
    #
    #     image_name = os.path.join(root, file)
    #
    #     j_path = f'/data/3d_pose_2023/2D_json/{root.split("/")[-1].split("_")[0]}_{root.split("/")[-1].split("_")[1]}/{file.split(".")[0]}.json'
    #     new_j_dir = f'/data/3d_pose_2023/2D_json_new/{root.split("/")[-1].split("_")[0]}_{root.split("/")[-1].split("_")[1]}/'
    #     new_j_path = f'/data/3d_pose_2023/2D_json_new/{root.split("/")[-1].split("_")[0]}_{root.split("/")[-1].split("_")[1]}/{file.split(".")[0]}.json'
    #
    #     # print(new_j_path, os.path.isfile(new_j_path))
    #     if not os.path.isfile(new_j_path):
    #         # test a single image, the resulting box is (x1, y1, x2, y2)
    #         mmdet_results = inference_detector(det_model, image_name)
    #
    #         # keep the person class bounding boxes.
    #         person_results = process_mmdet_results(mmdet_results, args.det_cat_id)
    #
    #         pose_results, returned_outputs = inference_top_down_pose_model(
    #             pose_model,
    #             image_name,
    #             person_results,
    #             bbox_thr=args.bbox_thr,
    #             format='xyxy',
    #             dataset=dataset,
    #             dataset_info=dataset_info,
    #             return_heatmap=False,
    #             outputs=None)
    #
    #         json_data = None
    #         try:
    #             with open(j_path, "r") as json_file:
    #                 json_data = json.load(json_file)
    #
    #                 json_data['annotations']["new_2d_pos"] = copy.deepcopy(json_data['annotations']["2d_pos"])
    #                 # org(json) -> new(pose_results)
    #                 for i in range(len(org_kpt_idx_list)):
    #                     json_data['annotations']["new_2d_pos"][org_kpt_idx_list[i]] = list(map(int, np.round(pose_results[0]['keypoints'][new_kpt_idx_list[i]][0:2], 0)))
    #
    #                 if not os.path.exists(new_j_dir):
    #                     os.makedirs(new_j_dir)
    #                 # JSON파일 업데이트
    #                 with open(new_j_path, 'w', encoding='utf-8') as make_file:
    #                     json.dump(json_data, make_file, indent="\t")
    #
    #         except FileNotFoundError as e:
    #             print("\n\n--------------FileNotFoundError--------------")
    #             print("json_data", json_data)
    #             print(f"file: {file}, j_path: {j_path}\n\n")
    #         except Exception as e:
    #             print("\n\n--------------Exception--------------")
    #             print(e.args)
    #             print(e)
    #             print("pose_results",pose_results)
    #             print("json_data", json_data)
    #             print(f"file: {file}, j_path: {j_path}\n\n")
    #         finally:
    #             json_data = None
    #
    # print(f"{count}/{total_count} - {round((count / total_count) * 100, 3)}%")

if __name__ == '__main__':
    main()
