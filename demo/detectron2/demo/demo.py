# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo

# constants
WINDOW_NAME = "COCO detections"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin models")
    parser.add_argument(
        "--config-file",
        default="/home/chenfei/Datadisk/home/chenfei/github/signCorner/demo/detectron2/configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml",   ####1
        #default="/home/chenfei/Datadisk/home/chenfei/github/signCorner/demo/detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",

        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument("--input", nargs="+", help="A list of space separated input images")
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        #default=[],
        default=['MODEL.WEIGHTS','/home/chenfei/Datadisk/home/chenfei/github/signCorner/demo/detectron2/output/newtrainval_sign_model_res101.pth'],  #sign

        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--imagepath",
        default="/home/chenfei/Datadisk/home/chenfei/github/signCorner/demo/detectron2/resultout/imagein/",
        help="image to be detected path",
    )
    parser.add_argument(
        "--imagesave",
        default="/home/chenfei/Datadisk/home/chenfei/github/signCorner/demo/detectron2/resultout/imageout/",
        help="image to be saved path",
    )
    parser.add_argument(
        "--labelsave",
        default="/home/chenfei/Datadisk/home/chenfei/github/signCorner/demo/detectron2/resultout/",
        help="label to be saved path",
    )
    return parser


def boxinfer(imagepath,boxdir):

    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)


    image_save = boxdir
    label_save = boxdir


    if imagepath:
        image = imagepath.split('/')[-1]
        path = imagepath
        img = read_image(imagepath, format="BGR")
        start_time = time.time()
        predictions, visualized_output, newboxes, newlabels = demo.run_on_image(img)  #################
        # predictions, visualized_output = demo.run_on_image(img)
        logger.info(
            "{}: detected {} instances in {:.2f}s".format(
                path, len(predictions["instances"]), time.time() - start_time
            )
        )

        if image_save:  # ####### save image
            if os.path.isdir(image_save):
                assert os.path.isdir(image_save), image_save
                out_filename = os.path.join(image_save, os.path.basename(path))
            visualized_output.save(out_filename)
        list = []  # ############save txt
        for idx in range(0, len(newlabels)):
            # print(newboxes[idx], newlabels[idx])
            box = newboxes[idx]
            label = newlabels[idx].split(' ')[0]
            x1 = (box[0])
            y1 = (box[1])
            x2 = (box[2])
            y2 = (box[3])

            #sign
            if label == 'traffic3':
                label = 1
            if label == 'traffic4':
                label = 2
            if label == 'traffic3-back':
                label = 3
            if label == 'traffic4-back':
                label = 4
            if label == 'circle':
                label = 5
            if label == 'circle-back':
                label = 6


            #lane
            #if label == 'lanecorner':
            #    label = 1

            list.append([int(x1), int(y1), int(x2), int(y2), str(label)])
        #print(list)
        # exit()

        '''
        labelout = label_save + 'result.txt'
        #print(labelout)
        with open(labelout, 'a') as f:
            # f.write(imagename)
            f.write(image)
            f.write(' ')
            if len(list) > 0:
                output = " ".join(
                    [','.join([str(x1), str(y1), str(x2), str(y2), str(label)]) for x1, y1, x2, y2, label in list])
                #print(output)
                f.write(output)
            f.write("\n")
        '''

        return list

    elif args.webcam:
        assert args.input is None, "Cannot have both --input and --webcam!"
        cam = cv2.VideoCapture(0)
        for vis in tqdm.tqdm(demo.run_on_video(cam)):
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, vis)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        cv2.destroyAllWindows()
    elif args.video_input:
        video = cv2.VideoCapture(args.video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(args.video_input)

        if args.output:
            if os.path.isdir(args.output):
                output_fname = os.path.join(args.output, basename)
                output_fname = os.path.splitext(output_fname)[0] + ".mkv"
            else:
                output_fname = args.output
            assert not os.path.isfile(output_fname), output_fname
            output_file = cv2.VideoWriter(
                filename=output_fname,
                # some installation of opencv may not support x264 (due to its license),
                # you can try other format (e.g. MPEG)
                fourcc=cv2.VideoWriter_fourcc(*"x264"),
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )
        assert os.path.isfile(args.video_input)
        for vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
            if args.output:
                output_file.write(vis_frame)
            else:
                cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                cv2.imshow(basename, vis_frame)
                if cv2.waitKey(1) == 27:
                    break  # esc to quit
        video.release()
        if args.output:
            output_file.release()
        else:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    boxinfer(args,cfg)

