from experiments.segmentation import run_semantic_segmentation_pipeline
from experiments.evaluate import eval_segmentation
from common.utils import get_config

cfg = get_config()

run_semantic_segmentation_pipeline()

res_path = f'{cfg["output_dir"]}/Test-SemSeg'
gt_path = f'{cfg["data_dir"]}/Test/GT_Object'

print('\nEvaluage predicted segmentation masks\n')
eval_segmentation(res_path, gt_path)