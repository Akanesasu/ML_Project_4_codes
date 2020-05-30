from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader, DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
import detectron2.model_zoo as model_zoo
import os
import cv2
import random
import argparse

DATASET_DIR = '../../proj_dataset/fracture/'
OUTPUT_DIR = './output_800_10000_0.4_960_16_1e-3_4k/'

def main(args):
    register_coco_instances("test", {}, args.anno_path, args.data_dir)
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"))
    cfg.DATASETS.TEST = ("test",)
    cfg.MODEL.WEIGHTS = args.model_path
    cfg.MODEL.PIXEL_MEAN = [84.6518, 84.6518, 84.6518]
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[16], [32], [64], [128], [256]]
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.MAX_ITER = 10000
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon)
    cfg.OUTPUT_DIR = OUTPUT_DIR
    
    # INPUT settings are very important !!
    cfg.INPUT.CROP.ENABLED = True
    cfg.INPUT.CROP.TYPE = "relative"
    cfg.INPUT.CROP.SIZE = [0.4, 0.4]
    cfg.INPUT.MIN_SIZE_TRAIN = (960,)  # 2400 * 0.4
    cfg.INPUT.MAX_SIZE_TRAIN = 1440
    cfg.INPUT.MIN_SIZE_TEST = 2400
    cfg.INPUT.MAX_SIZE_TEST = 3600
    
    # print(cfg)
    
    predictor = DefaultPredictor(cfg)
    
    evaluator = COCOEvaluator("test", cfg, False, output_dir=args.output_path)
    val_loader = build_detection_test_loader(cfg, "test")
    inference_on_dataset(predictor.model, val_loader, evaluator)
    
    """
	predictor = DefaultPredictor(cfg)
	for d in random.sample(dataset_dicts, 5):
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
	"""
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Rib Fracture Detection Testing Program')
    parser.add_argument('--data_dir',
                        help='path to the x-ray images of the testing dataset',
                        type=str,
                        default=None)
    parser.add_argument('--anno_path',
                        help='path to the annotations corresponding to the images (optional) ',
                        type=str,
                        default=None)
    parser.add_argument('--output_path',
                        help='specify a directory to output the predictions of our model there',
                        type=str,
                        default=None)
    parser.add_argument('--model_path',
                        help='path to our trained model',
                        type=str,
                        default=None)
    args = parser.parse_args()
    main(args)
    