'''
IMPORT LIBRARIES
'''
from argparse import ArgumentParser
import os

import torch, torchvision
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances

import matplotlib.pyplot as plt
import numpy as np
import cv2

'''
TRAINING FUNCTION
'''
def train(output, itr, use_cpu=False):
    register_coco_instances("train", {}, args.annotation, args.directory)
    metadata = MetadataCatalog.get("train")
    dataset_dicts = DatasetCatalog.get("train")
    
    # SET CONFIG
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("train",)
    cfg.DATASETS.TEST = ()
    cfg.TEST.EVAL_PERIOD = 100
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 6
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.MAX_ITER = itr
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 19
    cfg.OUTPUT_DIR = output
    if use_cpu:
        cfg.MODEL.DEVICE = 'cpu'

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-a", "--annotation", metavar="path", type=str, help="path to coco annotation json", required=True)
    parser.add_argument("-d", "--directory", metavar="path", type=str, help="directory to images", required=True)
    parser.add_argument("-o", "--output", default='models/train', metavar="path", type=str, help="path to save the model", required=False)
    parser.add_argument("-i", "--itr", default=40000, metavar='num', help="numer of iterations to train", required=False)
    parser.add_argument("-g", "--gpu", default=0, metavar='num', help="gpu id", required=False)

    args = parser.parse_args()
    print(' '.join(f'{k}={v}' for k, v in vars(args).items()))

    if str(args.gpu) == '-1':
        DEVICE = 'cpu'
        use_cpu = True
        print('using cpu...\nthis may be slow, gpu is recommended')
    else:
        use_cpu = False
        os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
        print('using cuda device', args.gpu, '...')
    
    train(args.output, int(args.itr), use_cpu)