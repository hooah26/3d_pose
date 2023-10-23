# Copyright (c) OpenMMLab. All rights reserved.
import os
import numpy as np
import warnings
from argparse import ArgumentParser
import json
import cv2
from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         process_mmdet_results, vis_pose_result)
from mmpose.datasets import DatasetInfo

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

def joints_dict():
    joints = {
        "coco": {
            "keypoints": {
                0: "nose",
                1: "left_eye",
                2: "right_eye",
                3: "left_ear",
                4: "right_ear",
                5: "left_shoulder",
                6: "right_shoulder",
                7: "left_elbow",
                8: "right_elbow",
                9: "left_wrist",
                10: "right_wrist",
                11: "left_hip",
                12: "right_hip",
                13: "left_knee",
                14: "right_knee",
                15: "left_ankle",
                16: "right_ankle"
            },
            "skeleton": [
                # # [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8],
                # # [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]
                # [15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7],
                # [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]
                [15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7],
                [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4],  # [3, 5], [4, 6]
                [0, 5], [0, 6]
            ]
        },
        "mpii": {
            "keypoints": {
                0: "right_ankle",
                1: "right_knee",
                2: "right_hip",
                3: "left_hip",
                4: "left_knee",
                5: "left_ankle",
                6: "pelvis",
                7: "thorax",
                8: "upper_neck",
                9: "head top",
                10: "right_wrist",
                11: "right_elbow",
                12: "right_shoulder",
                13: "left_shoulder",
                14: "left_elbow",
                15: "left_wrist"
            },
            "skeleton": [
                # [5, 4], [4, 3], [0, 1], [1, 2], [3, 2], [13, 3], [12, 2], [13, 12], [13, 14],
                # [12, 11], [14, 15], [11, 10], # [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]
                [5, 4], [4, 3], [0, 1], [1, 2], [3, 2], [3, 6], [2, 6], [6, 7], [7, 8], [8, 9],
                [13, 7], [12, 7], [13, 14], [12, 11], [14, 15], [11, 10],
            ]
        },
    }
    return joints


def main():
    """Visualize the demo images.

    Using mmdet to detect the human.
    """
    parser = ArgumentParser()
    parser.add_argument('det_config', help='Config file for detection')
    parser.add_argument('det_checkpoint', help='Checkpoint file for detection')
    parser.add_argument('pose_config', help='Config file for pose')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')
    parser.add_argument('--video-path', type=str, help='Video path')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show visualizations.')
    parser.add_argument(
        '--out-video-root',
        default='',
        help='Root of the output video file. '
        'Default not saving the visualization video.')
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

    assert has_mmdet, 'Please install mmdet to run the demo.'

    args = parser.parse_args()

    assert args.show or (args.out_video_root != '')
    assert args.det_config is not None
    assert args.det_checkpoint is not None

    det_model = init_detector(
        args.det_config, args.det_checkpoint, device=args.device.lower())
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

    folder_path = './demo/Sample_test_video'
    file_list = os.listdir(folder_path)
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = []
    images = []
    labels = []
    count = 0



    for file_name in file_list:
        file_extension = os.path.splitext(file_name)[-1].lower()
        if file_extension in video_extensions:
            video_path = os.path.join(folder_path, file_name)
            video_files.append(video_path)

        for video_file in video_files:
            cap = cv2.VideoCapture(video_file)
            assert cap.isOpened(), f'Faild to load video file {video_file}'

            if args.out_video_root == '':
                save_out_video = False
            else:
                os.makedirs(args.out_video_root, exist_ok=True)
                save_out_video = True

            if save_out_video:
                fps = cap.get(cv2.CAP_PROP_FPS)
                size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                videoWriter = cv2.VideoWriter(
                    os.path.join(args.out_video_root,
                                f'vis_{os.path.basename(video_file)}'), fourcc,
                    fps, size)

            # optional
            return_heatmap = False

            # e.g. use ('backbone', ) to return backbone feature
            output_layer_names = None
            cnt = 0
            angle_buffer = []
            frame_dict = {"data": []}
            video_file_name = "./demo/2d_json/" + str(os.path.splitext(os.path.basename(video_file))[0])
            while (cap.isOpened()):
                flag, img = cap.read()
                if not flag:
                    break
                # test a single image, the resulting box is (x1, y1, x2, y2)
                mmdet_results = inference_detector(det_model, img)
                person_results = process_mmdet_results(mmdet_results, args.det_cat_id)

                # test a single image, with a list of bboxes.
                pose_results, returned_outputs = inference_top_down_pose_model(
                    pose_model,
                    img,
                    person_results,
                    bbox_thr=args.bbox_thr,
                    format='xyxy',
                    dataset=dataset,
                    dataset_info=dataset_info,
                    return_heatmap=return_heatmap,
                    outputs=output_layer_names)

                if pose_results == None:
                    print("-------------------------None------------------------")
                    continue
                # show the results
                vis_img = vis_pose_result(
                    pose_model,
                    img,
                    pose_results,
                    dataset=dataset,
                    dataset_info=dataset_info,
                    kpt_score_thr=args.kpt_thr,
                    radius=args.radius,
                    thickness=args.thickness,
                    show=False)

                if vis_img is None:
                    print("-------------------------None------------------------")
                    continue

                keypoints_list = [pose_result['keypoints'] for pose_result in pose_results]
                #print("key!!!", keypoints_list)
                for keypoints in keypoints_list:
                    key_point = keypoints
                coord_x1, coord_y1 = key_point[5][0], key_point[5][1]
                coord_x2, coord_y2 = key_point[7][0], key_point[7][1]
                offset_x, offset_y = coord_x2-coord_x1 , coord_y2-coord_y1
                a = np.rad2deg(np.arccos(np.dot([offset_x, offset_y] / np.linalg.norm([offset_x, offset_y]),[0, 1])))
                angle_buffer.append(a)
                angle_buffer = angle_buffer[-20:]

                if round(np.mean(angle_buffer),2) > 60:
                    print("--------------------shoot--------------------")
                    keypoints = [{"x": float(key_point[i][0]), "y": float(key_point[i][1]), "name": joints_dict()['coco']['keypoints'][i]}
                                for i in range(len(key_point))]
                    frame_dict["data"].append({"label": "shoot", "keypoints": keypoints})
                    #cv2.imwrite(f'./demo/shoot_img/{os.path.splitext(os.path.basename(video_file))[0]}_{cnt}.jpg',vis_img)
                    #cnt+=1



                    #cv2.putText(vis_img, "SHOOT", (50, 100), cv2.FONT_HERSHEY_PLAIN, 3, (150, 222, 209), 3)
                    #cv2.putText(vis_img, "FPS:{}".format(fps), (50, 150), cv2.FONT_HERSHEY_PLAIN, 3, (150, 222, 209), 3)

                else:
                    print("--------------------Rest--------------------")
                    keypoints = [{"x": float(key_point[i][0]), "y": float(key_point[i][1]), "name": joints_dict()['coco']['keypoints'][i]}
                                for i in range(len(key_point))]
                    frame_dict["data"].append({"label": "rest", "keypoints": keypoints})
                    #cv2.imwrite(f'./demo/rest_img/{os.path.splitext(os.path.basename(video_file))[0]}_{cnt}.jpg',vis_img)
                    #cnt+=1


                    #cv2.putText(vis_img, "REST", (50, 100), cv2.FONT_HERSHEY_PLAIN, 3, (150, 222, 209), 3)
                    #cv2.putText(vis_img, "FPS:{}".format(fps), (50, 150), cv2.FONT_HERSHEY_PLAIN, 3, (150, 222, 209), 3)
                #out.write(image)

                if args.show:
                    cv2.imshow('Image', vis_img)

                #cv2.imwrite(f'./test_{cnt}.jpg',vis_img)
                #cnt+=1
                if save_out_video:
                    videoWriter.write(vis_img)

                if args.show and cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            if save_out_video:
                videoWriter.release()
            if args.show:
                cv2.destroyAllWindows()

            with open(video_file_name + ".json","w") as json_file:
                json.dump(frame_dict, json_file)


if __name__ == '__main__':
    main()
