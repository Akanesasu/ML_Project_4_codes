from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
import detectron2.model_zoo as model_zoo
import os

DATASET_DIR = '../../proj_dataset/fracture_gaussian0.002_10/'
OUTPUT_DIR = 'output_800_30000_gaussian0.002_10/'




if __name__ == "__main__":
	register_coco_instances("train", {}, DATASET_DIR + "/annotations/anno_train.json", DATASET_DIR + "/train/")
	register_coco_instances("val", {}, DATASET_DIR + "/annotations/anno_val.json", DATASET_DIR + "/val/")
	cfg = get_cfg()
	cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"))
	cfg.DATASETS.TRAIN = ("train",)
	cfg.DATASETS.TEST = ("val",)
	#cfg.INPUT.FORMAT = "RGB"
	cfg.MODEL.PIXEL_MEAN = [84.6518, 84.6518, 84.6518]
	#cfg.MODEL.PIXEL_STD = [54.2385, 54.2385, 54.2385]
	cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8], [16], [32], [64], [128]]
	cfg.DATALOADER.NUM_WORKERS = 2
	cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
	cfg.SOLVER.IMS_PER_BATCH = 2
	cfg.SOLVER.BASE_LR = 0.0001  # pick a good LR
	# 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
	cfg.SOLVER.MAX_ITER = 30000
	cfg.SOLVER.STEPS = (15000, )
	cfg.SOLVER.CHECKPOINT_PERIOD = 5000
	cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
	cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon)
	cfg.OUTPUT_DIR = OUTPUT_DIR
	print(cfg)
	os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
	trainer = DefaultTrainer(cfg)
	trainer.resume_or_load(resume=True)
	trainer.train()