import os
import argparse
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.evaluation import COCOEvaluator
from detectron2.engine import DefaultTrainer
from detectron2.engine.hooks import BestCheckpointer
from detectron2.config import LazyConfig
from detectron2.config.instantiate import instantiate

os.environ["CUDA_VISIBLE_DEVICES"] = '3'

def parse_args():
    parser = argparse.ArgumentParser(description="Train Detectron2 with best AP checkpointing")
    parser.add_argument("--dataset_name", default="my_dataset_train", type=str)
    parser.add_argument("--json_path", default="data/all.json", type=str)
    parser.add_argument("--img_dir", default="data/train", type=str)
    
    parser.add_argument("--max_iter", default= 40000, type=int)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--num_classes", default=4, type=int)
    parser.add_argument("--eval_period", default=1000, type=int, help="Evaluation period in iterations")

    parser.add_argument("--model", default="mask_rcnn_X_101_32x8d_FPN_3x.yaml", type=str)
    parser.add_argument("--output_dir", default="checkpoints/models/X_101_dice_90", type=str)
    return parser.parse_args()


class TrainerWithBestAP(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, False, output_folder)

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.append(
            BestCheckpointer(self.cfg.TEST.EVAL_PERIOD, self.checkpointer, "segm/AP")
        )
        return hooks


def main():
    args = parse_args()

    # Register dataset
    register_coco_instances(args.dataset_name, {}, args.json_path, args.img_dir)
    register_coco_instances('my_dataset_val', {}, 'data/val2.json', args.img_dir)
    MetadataCatalog.get(args.dataset_name)
    DatasetCatalog.get(args.dataset_name)

    # Build config
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(f"COCO-InstanceSegmentation/{args.model}"))
    # cfg = LazyConfig.load("/mnt/HDD1/tuong/lam_selected/HW3/detectron2/configs/new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py")
    

    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(f"COCO-InstanceSegmentation/{args.model}")

    cfg.DATASETS.TRAIN = (args.dataset_name,)
    cfg.DATASETS.TEST = (args.dataset_name,)
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.SOLVER.MAX_ITER = args.max_iter
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = args.num_classes    
    cfg.OUTPUT_DIR = args.output_dir
    cfg.TEST.EVAL_PERIOD = args.eval_period
    cfg.DATALOADER.SAMPLER_TRAIN = "RepeatFactorTrainingSampler"
    cfg.DATALOADER.REPEAT_THRESHOLD = 0.05
    cfg.SOLVER.CHECKPOINT_PERIOD = 50000
    cfg.SOLVER.WARMUP_ITERS = 1000
    cfg.TEST.DETECTIONS_PER_IMAGE = 1000 
    # Higher resolution features for more precise masks
    cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 7
    cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION = 14
    # Retain tiny proposals
    # cfg.MODEL.RPN.MIN_SIZE = 0

    # Adjust thresholds to increase recall
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
    
    # ##### Custom settings #####
    cfg.MODEL.FPN.IN_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.RPN.IN_FEATURES = ["p2", "p3", "p4", "p5", "p6"]
    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 3000
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST  = 1500
    
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[4], [8], [16], [32], [64]]
    # ##### Custom settings #####


    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Save the configuration to a file.
    config_file_path = os.path.join(cfg.OUTPUT_DIR, "config.yaml")
    with open(config_file_path, "w", encoding="utf-8") as file_obj:
        file_obj.write(cfg.dump())
    
    trainer = TrainerWithBestAP(cfg)
    trainer.resume_or_load(resume=False)

    # Print total and trainable parameters
    model = trainer.model
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel parameter summary:")
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}\n")

    trainer.train()

if __name__ == "__main__":
    main()
