from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import shutil

from PIL import Image
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision
import cv2
import numpy as np
import sys

import _init_paths
import models
from config import cfg
from config import update_config
from core.function import get_final_preds
from utils.transforms import get_affine_transform
from fvcore.common.file_io import PathManager

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  #过滤截断图像

sys.path.append('/home/chenfei/Datadisk/home/chenfei/github/TDSignCornerNet_large/demo/detectron2/demo/')   ############# demo.py
import demo

def get_pose_estimation_prediction(pose_model, image, center, scale):
    rotation = 0

    # pose estimation transformation
    trans = get_affine_transform(center, scale, rotation, cfg.MODEL.IMAGE_SIZE)
    model_input = cv2.warpAffine(    # origin image --->object box(192*256)
        image,
        trans,
        (int(cfg.MODEL.IMAGE_SIZE[0]), int(cfg.MODEL.IMAGE_SIZE[1])),
        flags=cv2.INTER_LINEAR)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    # pose estimation inference
    model_input = transform(model_input).unsqueeze(0)
    # switch to evaluate mode
    pose_model.eval()
    with torch.no_grad():
        # compute output heatmap
        output = pose_model(model_input)  # torch.Size([1, 4, 64, 48])

        preds, _ = get_final_preds(
            cfg,
            output.clone().cpu().numpy(),
            np .asarray([center]),
            np.asarray([scale]))

        return preds


def box_to_center_scale(box, model_image_width, model_image_height):
    center = np.zeros((2), dtype=np.float32)

    bottom_left_corner = box[0]  ##box = [(pred_box[0], pred_box[1]), (pred_box[2], pred_box[3])]
    top_right_corner = box[1]
    box_width = top_right_corner[0]-bottom_left_corner[0]
    box_height = top_right_corner[1]-bottom_left_corner[1]
    bottom_left_x = bottom_left_corner[0]
    bottom_left_y = bottom_left_corner[1]
    center[0] = bottom_left_x + box_width * 0.5
    center[1] = bottom_left_y + box_height * 0.5

    aspect_ratio = model_image_width * 1.0 / model_image_height   # 192/256
    pixel_std = 200

    if box_width > aspect_ratio * box_height:  # box   w/h>192/256(宽型)，宽不变，高要按比例拉大，矮胖变高
        box_height = box_width * 1.0 / aspect_ratio
    elif box_width < aspect_ratio * box_height:       # box   w/h<192/256(窄型），高不变，宽按比例拉大，瘦高变粗
        box_width = box_height * aspect_ratio
    scale = np.array(
        [box_width * 1.0 / pixel_std, box_height * 1.0 / pixel_std],
        dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25

    return center, scale


def prepare_output_dirs(prefix='output/'):
    pose_dir = prefix+'poses/'
    box_dir = prefix+'boxes/'
    if os.path.exists(pose_dir) and os.path.isdir(pose_dir):
        shutil.rmtree(pose_dir)
    if os.path.exists(box_dir) and os.path.isdir(box_dir):
        shutil.rmtree(box_dir)
    os.makedirs(pose_dir, exist_ok=True)
    os.makedirs(box_dir, exist_ok=True)
    return pose_dir, box_dir


def read_image(file_name, format=None):
    with PathManager.open(file_name, "rb") as f:
        image = Image.open(f)

        if format is not None:
            # PIL only supports RGB, so convert to RGB and flip channels over below
            conversion_format = format
            if format == "BGR":
                conversion_format = "RGB"
            image = image.convert(conversion_format)
        image = np.asarray(image)
        if format == "BGR":
            # flip channels if needed
            image = image[:, :, ::-1]
        # PIL squeezes out the channel dimension for "L", so make it HWC
        if format == "L":
            image = np.expand_dims(image, -1)
        return image


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    #parser.add_argument('--cfg', type=str, required=True)
    #parser.add_argument('--videoFile', type=str, required=True)
    parser.add_argument('--cfg', type=str, default='demo/inference-config.yaml')
    parser.add_argument('--videoFile', type=str, default='demo/hrnet-demo.gif')
    parser.add_argument('--outputDir', type=str, default='output/')
    parser.add_argument('--inferenceFps', type=int, default=10)
    parser.add_argument('--writeBoxFrames', action='store_true')
    parser.add_argument('opts',
                        help='Modify config options using the command-line',
                        default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument(
        "--imagepath",
        default="/home/chenfei/Datadisk/home/chenfei/github/signCorner/evaluation/newTrainVal/val2017/",
        help="image to be detected path",
    )
    parser.add_argument(
        "--imagesave",
        default="/home/chenfei/Datadisk/home/chenfei/github/TDSignCornerNet_large/result/imagesave/",
        help="image to be saved path",
    )

    args = parser.parse_args()

    # args expected by supporting codebase
    args.modelDir = ''
    args.logDir = ''
    args.dataDir = ''
    args.prevModelDir = ''
    return args


def main():

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    args = parse_args()
    update_config(cfg, args)

    pose_dir, box_dir = prepare_output_dirs(args.outputDir)

    # model and weights
    pose_model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=False
    )

    if cfg.TEST.MODEL_FILE:
        print('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        pose_model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
    else:
        print('expected model defined in config at TEST.MODEL_FILE')

    pose_model = torch.nn.DataParallel(pose_model, device_ids=cfg.GPUS).cuda()

    image_paths = args.imagepath
    imagelist = os.listdir(image_paths)
    image_save = args.imagesave
    imagelist.sort()
    print(len(imagelist))

    for image in imagelist:
        image_path = os.path.join(image_paths, image)

        print(image_path)
        if os.path.getsize(image_path) == 0:
            print('image is 0byte')
            continue

        pred_boxes = demo.boxinfer(image_path, box_dir)   #  list.append([int(x1), int(y1), int(x2), int(y2), str(label)])

        # Loading image
        image_name = image_path.split('/')[-1]
        image_bgr = read_image(image_path, format="BGR")
        image = image_bgr[:, :, [2, 1, 0]]
        image = image.copy()  #######

        # pose estimation
        imagecorner = []
        for i in range(0, len(pred_boxes)):
            pred_box = pred_boxes[i]  # box model result
            #print(pred_box)
            box = [(pred_box[0], pred_box[1]), (pred_box[2], pred_box[3])]
            boxclass = str(pred_box[4])
            #print(type(boxclass))

            corner_pred = [[0 for _ in range(2)]  for _ in range(4)]
            if boxclass == '5' or boxclass == '6':  # 圆牌的角点直接赋值矩形角点不进点模型
                corner_pred[0][0] = pred_box[0]
                corner_pred[0][1] = pred_box[1]
                corner_pred[1][0] = pred_box[2]
                corner_pred[1][1] = pred_box[1]
                corner_pred[2][0] = pred_box[2]
                corner_pred[2][1] = pred_box[3]
                corner_pred[3][0] = pred_box[0]
                corner_pred[3][1] = pred_box[3]
            else: # 其他类别进点模型
                center, scale = box_to_center_scale(box, cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1])
                image_pose = image.copy() if cfg.DATASET.COLOR_RGB else image_bgr.copy()
                # print(center, scale)

                pose_preds = get_pose_estimation_prediction(pose_model, image_pose, center, scale)
                corner_pred = pose_preds[0]  ##

            cv2.rectangle(image, (pred_box[0], pred_box[1]), (pred_box[2], pred_box[3]), (255, 255, 0), 1)  # plot box
            image = cv2.putText(image, boxclass, (pred_box[0], pred_box[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

            signinstance = []
            for _, mat in enumerate(corner_pred):  # plot corner
                x_coord, y_coord = int(mat[0]), int(mat[1])
                cv2.circle(image, (x_coord, y_coord), int(image.shape[0]/300), (255, 0, 0), -1)  ###
                signinstance.append([mat[0]-0.01, mat[1]-0.01])

            Image.fromarray(image).save(os.path.join(image_save, image_name))


            #删除三角形多余点
            from scipy.spatial.distance import pdist
            if boxclass == '1' or boxclass == '3':
                assert len(signinstance) ==4
                if len(signinstance) == 4 :
                    deletFlag = 0
                    for j in range(0, 2):
                        for i in range(j+1, len(signinstance) - 1):
                            XY = np.vstack([signinstance[j], signinstance[i]])  # [[x1,x2],[y1,y2]]
                            if pdist(XY) < 2:
                                deletFlag = 1
                                deletesigncorner = np.delete(signinstance, j, axis=0) # 距离小于2的删除重复的一个就行
                                signinstance = deletesigncorner.tolist()
                                continue
                        if deletFlag == 1 :
                            continue
                    if deletFlag == 0 :
                        deletesigncorner = np.delete(signinstance, 1, axis=0) # 没有距离小于２的删除第二个，还存在问题
                        signinstance = deletesigncorner.tolist()
            signinstance = {str(boxclass): signinstance}
            imagecorner.append(signinstance)
        print('cornerResult--------------',imagecorner)

        # to NTD txt format
        list = []  # ############save txt
        listdata = ''
        if len(imagecorner) > 0:
            for idx in range(0, len(imagecorner)):
                # print(newboxes[idx], newlabels[idx])
                # box = imagecorner[idx].values()
                # label = imagecorner[idx].keys()
                for label, box in imagecorner[idx].items():
                    if label == '1':
                        label = 'traffic3'
                    if label == '2':
                        label = 'traffic4'
                    if label == '3':
                        label = 'traffic3-back'
                    if label == '4':
                        label = 'traffic4-back'
                    if label == '5':
                        label = 'circle'
                    if label == '6':
                        label = 'circle-back'

                    corners = []
                    cornersdata = ''
                    for i in range(0, len(box)):
                        corners.append(int(box[i][0]))
                        corners.append(int(box[i][1]))
                        cornersdata = cornersdata + str(int(box[i][0])) + ',' + str(int(box[i][1])) + ','
                    corners.append(str(label))
                    cornersdata = cornersdata + str(label) + ' '
                    # print(corners)
                list.append(corners)
                listdata = listdata + cornersdata
            if listdata is not None:
                listdata = listdata[:-1]

            imagetxtname = (image_path.split('/')[-1])[:-4] + '.txt'
            labelout = '/home/chenfei/Datadisk/home/chenfei/github/TDSignCornerNet_large/result/outtxt/' + imagetxtname
            with open(labelout, 'w') as f:
                # f.write(imagename)
                f.write(image_path.split('/')[-1])
                f.write(' ')
                if listdata is not None:
                    f.write(listdata)


if __name__ == '__main__':
    main()
