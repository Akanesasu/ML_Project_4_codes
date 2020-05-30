"""
The main model implementation for fraction detection.
"""

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
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.checkpoint import DetectionCheckpointer
import detectron2.model_zoo as model_zoo
from detectron2.utils.visualizer import Visualizer
import os
import cv2
import random

PRE_TRAINED_CKPT_PATH = '../CheXNet/model.pth.tar'
DATASET_DIR = '../../proj_dataset/fracture/'
OUTPUT_DIR = 'output_ohhh/'

@BACKBONE_REGISTRY.register()
class CheXNet(Backbone):
	"""
	Use CheXNet for backbone (abstracting 2D features)
	Code is acquired and changed from
	https://github.com/arnoweng/CheXNet/blob/master/model.py
	"""
	def __init__(self, cfg=None, input_shape=None):
		super(CheXNet, self).__init__()
		densenet121 = torchvision.models.densenet121(pretrained=False).cuda()
		# load checkpoint (pre-trained model on chest X-ray images)
		# we only need the abstracted features
		self.features = densenet121.features
		if os.path.isfile(PRE_TRAINED_CKPT_PATH):
			print("=> loading pre-trained model weights")
			checkpoint = torch.load(PRE_TRAINED_CKPT_PATH)
			self.features.load_state_dict(checkpoint['state_dict'],
										  strict=False)
			print("=> loaded pre-trained model weights")
		else:
			raise FileNotFoundError\
				("=> no pre-trained model weights found")
		# freeze the pre-trained weights
		# TODO: we can choose not to freeze the params,
		#  but this will make the training process very slow,
		#  and I dont know whether it's beneficial
		for params in self.features.parameters():
			params.requires_grad = False
		"""
		for params in self.features.conv0.parameters():
			params.requires_grad = False
		for params in self.features.norm0.parameters():
			params.requires_grad = False
		for params in self.features.relu0.parameters():
			params.requires_grad = False
		for params in self.features.pool0.parameters():
			params.requires_grad = False
		for params in self.features.denseblock1.parameters():
			params.requires_grad = False
		for params in self.features.transition1.parameters():
			params.requires_grad = False
		for params in self.features.denseblock2.parameters():
			params.requires_grad = False
		"""
	
	def output_shape(self):
		"""
		Returns:
			dict[str->ShapeSpec]
		"""
		return {"2D-Feature": ShapeSpec(channels=1024, stride=32)}
	
	def forward(self, image):
		# the input is a grey-scale image, we convert it to RGB
		# image = image.repeat(3, 1, 1)
		return {"2D-Feature": self.features(image)}


if __name__ == '__main__':
	
	cfg = get_cfg()
	cfg.merge_from_file('../detectron2-ResNeSt/configs/COCO-Detection/'
						'faster_cascade_rcnn_R_50_FPN_syncbn_range-scale_1x.yaml')
	
	cfg.MODEL.DEVICE = "cuda" # Train on GPU
	
	cfg.MODEL.BACKBONE.NAME = 'CheXNet' # Change to CheXNet
	# I freeze it manually in __init__
	cfg.MODEL.BACKBONE.FREEZE_AT = 2
	
	# prepare dataset
	register_coco_instances("train", {},
							DATASET_DIR + 'annotations/anno_train.json',
							DATASET_DIR + 'train/', )
	register_coco_instances("valid", {},
							DATASET_DIR + 'annotations/anno_val.json',
							DATASET_DIR + 'val/', )
	cfg.DATASETS.TRAIN = ("train",)
	cfg.DATASETS.TEST = ("valid",)
	
	# the dataloader will scale the image to (1200 x ?) (assume H < W) (1200 < ? < 2000)
	#cfg.INPUT.MIN_SIZE_TRAIN = (800, )
	#cfg.INPUT.MAX_SIZE_TRAIN = 1333
	#cfg.INPUT.MIN_SIZE_TEST = 800
	#cfg.INPUT.MAX_SIZE_TEST = 1333
	#cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = "choice"
	
	# Image Size after Crop: H * [0.8 ~ 1] x W * [0.8 ~ 1]
	#cfg.INPUT.CROP.ENABLED = True
	#cfg.INPUT.CROP.TYPE = "relative_range"
	#cfg.INPUT.CROP.SIZE = [0.9, 0.9]
	
	# Image Format Mode : L (8-bit pixels, black and white)
	cfg.INPUT.FORMAT = "RGB"
	# mean and standard deviation computed on fracture dataset
	# cfg.MODEL.PIXEL_MEAN = [84.6518, 84.6518, 84.6518]
	# cfg.MODEL.PIXEL_STD = [54.2385, 54.2385, 54.2385]
	
	# Dataloader stuff
	cfg.DATALOADER.NUM_WORKERS = 2
	# I think there is no image with empty annotation, but I still filter them
	cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True
	
	# Anchor generator options
	# Anchor sizes (i.e. sqrt of area) in absolute pixels w.r.t. the network input.
	cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8, 16, 32, 64, 128]]
	# Anchor aspect ratios. For each area given in `SIZES`, anchors with different aspect
	# ratios are generated by an anchor generator.
	cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]
	# Anchor angles.
	cfg.MODEL.ANCHOR_GENERATOR.ANGLES = [[-90, 0, 90]]
	
	# RPN options
	cfg.MODEL.RPN.IN_FEATURES = ["2D-Feature"]
	
	# ROI HEADS options
	cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 # fracture or not
	cfg.MODEL.ROI_HEADS.IN_FEATURES = ["2D-Feature"]
	# RoI minibatch size *per image* (number of regions of interest [ROIs])
	# Total number of RoIs per training minibatch =
	#   ROI_HEADS.BATCH_SIZE_PER_IMAGE * SOLVER.IMS_PER_BATCH
	# E.g., a common configuration is: 512 * 16 = 8192
	cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
	# Target fraction of RoI minibatch that is labeled foreground (i.e. class > 0)
	cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.25
	
	# Solver
	cfg.SOLVER.MAX_ITER = 3000
	cfg.SOLVER.BASE_LR = 0.0001
	# The iteration number to decrease learning rate by GAMMA.
	cfg.SOLVER.STEPS = (1000, 2000)
	cfg.SOLVER.CHECKPOINT_PERIOD = 3000
	# Number of images per batch across all machines.
	# we have only 1 GPU, so it will see 2 images per batch.
	cfg.SOLVER.IMS_PER_BATCH = 2
	
	# OUTPUT
	cfg.OUTPUT_DIR = OUTPUT_DIR
	cfg.SEED = 998 # for reproducing
	
	cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
	print(cfg)
	
	metadata = MetadataCatalog.get("train")
	dataset_dicts = DatasetCatalog.get("train")
	
	trainer = DefaultTrainer(cfg)
	trainer.resume_or_load(resume=True)

	evaluator = COCOEvaluator("train", cfg, False, output_dir=os.path.join(cfg.OUTPUT_DIR, "inference"))
	val_loader = build_detection_test_loader(cfg, "train")
	inference_on_dataset(trainer.model, val_loader, evaluator)

	predictor = DefaultPredictor(cfg)
	for d in random.sample(dataset_dicts, 3):
		img = cv2.imread(d["file_name"])
		outputs = predictor(img)
		print(d["file_name"])
		visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1)
		visualizer = visualizer.draw_instance_predictions(outputs["instances"].to("cpu"))
		visualizer = visualizer.get_image()[:, :, ::-1]
		visualizer = Visualizer(visualizer[:, :, ::-1], metadata=metadata, scale=0.3)
		visualizer = visualizer.draw_dataset_dict(d)
		cv2.imshow('img', visualizer.get_image()[:, :, ::-1])
		cv2.waitKey(0)
	

		