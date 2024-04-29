import cv2
import os
import tqdm

class SemSegMetrics():
    def __init__(self):
        self.metrics = {
                "miou": 0,
                "pixel_acc": 0,
                "dice": 0,
        }
        self.count = 0

    def update(self, pred, gt):
        self.count += 1
        self.metrics["pixel_acc"] += (pred == gt).sum()/(gt.shape[0] * gt.shape[1])
        self.metrics["dice"] += 2 * (pred * gt).sum()/(pred.sum() + gt.sum())
        self.metrics["miou"] += (pred * gt).sum()/((pred + gt)>0).sum()

    def __str__(self):
        met = {}
        for k in self.metrics:
            met[k] = self.metrics[k]/ self.count

        s = " | ".join(["{}: {:.2f}".format(k, 100*met[k]) for k in met])
        return s



def eval_segmentation(result_folder, gt_folder):
    metrics = SemSegMetrics()
    for fname in tqdm.tqdm(os.listdir(result_folder)):
        pred_path = os.path.join(result_folder, fname)
        gt_path = os.path.join(gt_folder, fname)
        if ".jpg" in gt_path:
            gt_path = gt_path.replace(".jpg", ".png")

        pred = cv2.imread(pred_path, 0)
        gt   = cv2.imread(gt_path, 0)
        metrics.update(pred>0, gt>0)

    print(metrics)
    return metrics
