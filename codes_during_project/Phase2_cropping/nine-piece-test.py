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
OUTPUT_DIR = 'output_1/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

RELATIVE_CROP_H = 0.4
RELATIVE_CROP_W = 0.4
OUTSET_Hs = [0, 0.3, 0.6]
OUTSET_Ws = [0, 0.3, 0.6]

def crop_and_pred(input, model):
    """
    Crop the input image into 9 pieces.
    Args:
        input: (dict)
            format is
            {
            'file_name': string,
            'height': int, 'width': int,
            'image': tensor of size (C x H x W)
             }
        model: (nn.Module)
            the model picked from Trainer
    Returns:
        list[dict]: list of results, each corresponding to 1 piece.
    """
    image = input["image"]
    [_, H, W] = image.shape
    image_pil = vF.to_pil_image(image)
    cropped_inputs = []
    cropped_images = []
    height = int(H * RELATIVE_CROP_H)
    width = int(W * RELATIVE_CROP_W)
    for outset_h in OUTSET_Hs:
        for outset_w in OUTSET_Ws:
            top = int(H * outset_h)
            left = int(W * outset_w)
            cropped_image_pil = vF.crop(image_pil, top, left, height, width)
            #cropped_images.append(vF.to_tensor(cropped_image_pil))
            cropped_image = vF.to_tensor(cropped_image_pil).mul(255) # rescale to [0, 255]
            # print(cropped_image.size())
            cropped_inputs.append({"image":cropped_image,
                                   "height":int(input["height"] * RELATIVE_CROP_H),
                                   "width":int(input["width"] * RELATIVE_CROP_W)})
    # use model to gain predictions (the output)
    # check the results
    #plt.imshow(
    #    np.transpose(vutils.make_grid(cropped_images, 3), (1, 2, 0)))
    #plt.show()
    # cropped_inputs = [{"image":cropped_image} for cropped_image in cropped_images]
    # cropped_inputs = cropped_inputs[:1]
    outputs = model(cropped_inputs)
    return outputs

def recombine(crop_outputs, input):
    height = input["height"]
    width = input["width"]
    id = 0
    num_instances = 0
    boxes = torch.zeros(0, 4, dtype=torch.float32)
    scores = torch.zeros(0, dtype=torch.float32)
    classes = torch.zeros(0, dtype=torch.int64)
    for outset_h in OUTSET_Hs:
        for outset_w in OUTSET_Ws:
            top = int(height * outset_h)
            left = int(width * outset_w)
            crop_instances = crop_outputs[id]["instances"].to(device=torch.device("cpu"))
            id += 1
            
            num_instances += len(crop_instances)
            # add offset
            crop_boxes = crop_instances.pred_boxes.tensor + torch.tensor([left, top, left, top])
            crop_scores = crop_instances.scores
            crop_classes = crop_instances.pred_classes
            
            boxes = torch.cat([boxes, crop_boxes], dim=0)
            scores = torch.cat([scores, crop_scores], dim=0)
            classes = torch.cat([classes, crop_classes], dim=0)

    k = min(50, num_instances)
    #k = num_instances
    [_, idxes] = torch.topk(scores, k, largest=True, sorted=False)

    boxes = Boxes(boxes[idxes])
    scores = scores[idxes]
    classes = classes[idxes]
    #print(k, len(idxes))
    
    output = {"instances": Instances(image_size=(height,width),
                                     pred_boxes=boxes,
                                     scores=scores,
                                     pred_classes=classes)}
    
    return output
    



def my_inference_on_dataset(model, data_loader, evaluator):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.forward` accurately.
    The model will be used in eval mode.
    
    Extra (by Fan Fei):
    	Before putting the image into the model, we crop it into 9 pieces with overlapping.
    	Then we put it into the model, and recombine the boxes as the output.

    Args:
        model (nn.Module): a module which accepts an object from
            `data_loader` and returns some outputs. It will be temporarily set to `eval` mode.

            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator (DatasetEvaluator): the evaluator to run. Use `None` if you only want
            to benchmark, but don't want to do any evaluation.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} images".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    if evaluator is None:
        # create a no-op evaluator
        evaluator = DatasetEvaluators([])
    evaluator.reset()

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_compute_time = 0
    with inference_context(model), torch.no_grad():
        for idx, inputs in enumerate(data_loader):
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0

            start_compute_time = time.perf_counter()
            #wrap the input and output
            assert len(inputs) == 1
            input = inputs[0]
            output = recombine(crop_and_pred(input, model), input)
            outputs = [output]
            # outputs = model(inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            evaluator.process(inputs, outputs)

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Inference done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                )

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )

    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results

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
    cfg.INPUT.MIN_SIZE_TRAIN = (960, ) # 2400 * 0.4
    cfg.INPUT.MAX_SIZE_TRAIN = 1440
    cfg.INPUT.MIN_SIZE_TEST = 2400
    cfg.INPUT.MAX_SIZE_TEST = 4500
   
    cfg.MODEL.WEIGHTS = os.path.join(OUTPUT_DIR, "model_0004999.pth")
    
    print(cfg)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    
    evaluator = COCOEvaluator("val", cfg, False, output_dir=os.path.join(cfg.OUTPUT_DIR, "inferences"))
    val_loader = build_detection_test_loader(cfg, "val")
    #my_inference_on_dataset(trainer.model, val_loader, evaluator)
    inference_on_dataset(trainer.model, val_loader, evaluator)