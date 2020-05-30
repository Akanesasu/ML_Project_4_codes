import datetime
import logging
import time
from collections import OrderedDict
from contextlib import contextmanager
import torch
import numpy as np
# Used in my_inference_on_dataset
from detectron2.utils.comm import get_world_size, is_main_process
from detectron2.utils.logger import log_every_n_seconds
from detectron2.evaluation import DatasetEvaluators, inference_context, inference_on_dataset
# Used in main
from detectron2.data.datasets import register_coco_instances
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.evaluation import COCOEvaluator
from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultTrainer
from detectron2.structures import Boxes, Instances
import os
# Used in crop_and_pred and recombine
import torchvision.utils as vutils
import torchvision.transforms.functional as vF
import matplotlib
# matplotlib.use('AGG')
import matplotlib.pyplot as plt

DATASET_DIR = '../../proj_dataset/fracture/'
OUTPUT_DIR = 'output_3/'
os.makedirs(OUTPUT_DIR, exist_ok=True)


if __name__ == '__main__':
    register_coco_instances("train", {}, DATASET_DIR + "/annotations/anno_train.json", DATASET_DIR + "/train/")
    register_coco_instances("val", {}, DATASET_DIR + "/annotations/anno_val.json", DATASET_DIR + "/val/")
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"))
    cfg.DATASETS.TRAIN = ("train", )
    cfg.DATASETS.TEST = ("val", )
    cfg.OUTPUT_DIR = OUTPUT_DIR
    cfg.MODEL.PIXEL_MEAN = [84.6518, 84.6518, 84.6518]
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    # anchor size according to bbox size when short side = 2400 
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[16], [32], [64], [128], [256]]
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.SOLVER.BASE_LR = 0.0001
    cfg.SOLVER.MAX_ITER = 30000	
    cfg.SOLVER.STEPS = (10000, 20000)
    # INPUT settings are very important !!
    cfg.INPUT.CROP.ENABLED = True
    cfg.INPUT.CROP.TYPE = "relative"
    cfg.INPUT.CROP.SIZE = [0.4, 0.4]
    cfg.INPUT.MIN_SIZE_TRAIN = (1080, ) # 2700 * 0.4
    cfg.INPUT.MAX_SIZE_TRAIN = 1620
    cfg.INPUT.MIN_SIZE_TEST = 3000
    cfg.INPUT.MAX_SIZE_TEST = 4500
   
    cfg.MODEL.WEIGHTS = os.path.join(OUTPUT_DIR, "model_0024999.pth")
    
    print(cfg)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    
    evaluator = COCOEvaluator("val", cfg, False, output_dir=os.path.join(cfg.OUTPUT_DIR, "inferences"))
    val_loader = build_detection_test_loader(cfg, "val")
    #my_inference_on_dataset(trainer.model, val_loader, evaluator)
    inference_on_dataset(trainer.model, val_loader, evaluator)