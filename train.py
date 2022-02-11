from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-a", "--annotation", metavar="path", type=str, help="path to coco annotation json", required=True)
parser.add_argument("-d", "--directory", metavar="path", type=str, help="directory to images", required=True)
parser.add_argument("-o", "--output", metavar="path", type=str, help="path to save the model", required=True)
parser.add_argument("-i", "--itr", default=6000, metavar='num', help="numer of iterations to train", required=True)
parser.add_argument("-g", "--gpu", default=0, metavar='num', help="gpu id", required=False)

args = parser.parse_args()
print(args.annotation)
print(' '.join(f'{k}={v}' for k, v in vars(args).items()))

'''
IMPORT LIBRARIES
'''
import os
os.environ["CUDA_VISIBLE_DEVICES"]='1'

import torch, torchvision
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2.engine import DefaultPredictor
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
def train():
    # register_coco_instances("train", {}, args.a, args.d)
    # metadata = MetadataCatalog.get("train")
    # dataset_dicts = DatasetCatalog.get("train")
    return


if __name__ == '__main__':
    print(args.a, args.b)
    train()