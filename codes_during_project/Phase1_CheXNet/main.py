"""
The main model implementation for fraction detection.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

import detectron2
from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec, build_model
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader, DatasetCatalog, MetadataCatalog
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.checkpoint import DetectionCheckpointer
import detectron2.model_zoo as model_zoo

DATASET_DIR = '../../proj_dataset/fracture/'
OUTPUT_DIR = 'output_Naive/'

def setup(args):
	"""
	Create configs and perform basic setups.
	"""
	cfg = get_cfg()
	# cfg.merge_from_file(args.config_file)
	cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"))
	cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
	# prepare dataset
	register_coco_instances("train", {},
							DATASET_DIR + 'annotations/anno_train.json',
							DATASET_DIR + 'train/', )
	register_coco_instances("valid", {},
							DATASET_DIR + 'annotations/anno_val.json',
							DATASET_DIR + 'val/', )
	cfg.DATASETS.TRAIN = ("train",)
	cfg.DATASETS.TEST = ("valid",)
	
	# Image Format Mode : L (8-bit pixels, black and white)
	cfg.INPUT.FORMAT = "RGB"
	# mean and standard deviation computed on fracture dataset
	cfg.MODEL.PIXEL_MEAN = [84.6518, 84.6518, 84.6518]
	# cfg.MODEL.PIXEL_STD = [54.2385, 54.2385, 54.2385]
	
	# Dataloader stuff
	cfg.DATALOADER.NUM_WORKERS = 2
	# Anchor sizes
	cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[16], [32], [48], [64], [90]]
	# ROI HEADS options
	cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 # fracture or not
	# Solver
	cfg.SOLVER.MAX_ITER = 30000
	cfg.SOLVER.BASE_LR = 0.00025
	# The iteration number to decrease learning rate by GAMMA.
	cfg.SOLVER.STEPS = (10000, 20000)
	cfg.SOLVER.CHECKPOINT_PERIOD = 5000
	# Number of images per batch across all machines.
	# we have only 2 GPU, so it will see 2 images per batch.
	cfg.SOLVER.IMS_PER_BATCH = 4
	# OUTPUT
	cfg.OUTPUT_DIR = OUTPUT_DIR
	cfg.SEED = 9998 # for reproducing
	cfg.merge_from_list(args.opts)
	cfg.freeze()
	default_setup(cfg, args)
	return cfg

class Trainer(DefaultTrainer):

	@classmethod
	def build_evaluator(cls, cfg, dataset_name, output_folder=None):
		if output_folder is None:
			output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
		return COCOEvaluator(dataset_name, cfg, True, output_folder)


def main(args):
	cfg = setup(args)
	
	if args.eval_only:
		model = Trainer.build_model(cfg)
		DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
			cfg.MODEL.WEIGHTS, resume=args.resume
		)
		res = Trainer.test(cfg, model)
		if cfg.TEST.AUG.ENABLED:
			res.update(Trainer.test_with_TTA(cfg, model))
		if comm.is_main_process():
			verify_results(cfg, res)
		return res

	"""
	If you'd like to do anything fancier than the standard training logic,
	consider writing your own training loop or subclassing the trainer.
	"""
	trainer = Trainer(cfg)
	trainer.resume_or_load(resume=True)
	return trainer.train()


if __name__ == '__main__':
	args = default_argument_parser().parse_args()
	print("Command Line Args:", args)
	launch(
		main,
		args.num_gpus,
		num_machines=args.num_machines,
		machine_rank=args.machine_rank,
		dist_url=args.dist_url,
		args=(args,),
	)
	
	

		
