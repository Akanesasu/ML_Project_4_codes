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

DATASET_DIR = '../../proj_dataset/fracture/'
OUTPUT_DIR = 'output_1600_10000/'

if __name__ == "__main__":
    register_coco_instances("train", {}, DATASET_DIR + "/annotations/anno_train.json", DATASET_DIR + "/train/")
    register_coco_instances("val", {}, DATASET_DIR + "/annotations/anno_val.json", DATASET_DIR + "/val/")
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"))
    cfg.DATASETS.TRAIN = ("train",)
    cfg.DATASETS.TEST = ("val",)
    cfg.INPUT.MIN_SIZE_TEST = 2400
    cfg.INPUT.MAX_SIZE_TEST = 3600
    cfg.INPUT.MIN_SIZE_TRAIN = (960, )
    cfg.INPUT.MAX_SIZE_TRAIN = 1440
    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = "choice"
    #cfg.INPUT.FORMAT = "RGB"
    cfg.MODEL.PIXEL_MEAN = [84.6518, 84.6518, 84.6518]
    #cfg.MODEL.PIXEL_STD = [54.2385, 54.2385, 54.2385]
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8], [16], [32], [64], [128]]
    cfg.DATALOADER.NUM_WORKERS = 2
    # Let training initialize from model zoo
    #cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.0001  # pick a good LR
    # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
    cfg.SOLVER.MAX_ITER = 10000    
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon)
    cfg.OUTPUT_DIR = OUTPUT_DIR
    
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    #cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set the testing threshold for this model
    print(cfg)

    metadata = MetadataCatalog.get("val")
    dataset_dicts = DatasetCatalog.get("val")
    
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)

    evaluator = COCOEvaluator("val", cfg, False, output_dir=os.path.join(cfg.OUTPUT_DIR, "inference"))
    val_loader = build_detection_test_loader(cfg, "val")
    inference_on_dataset(trainer.model, val_loader, evaluator)
    """
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
    """
    