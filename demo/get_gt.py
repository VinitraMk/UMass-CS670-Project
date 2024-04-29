import os 
import shutil

gt_dir = "/Users/mustafa/Downloads/TestDataset/COD10K/GT"
for fname in os.listdir("./data/images"):
    if "DS" in fname: continue
    gt_path = os.path.join(gt_dir, fname.split(".")[0] + ".png")
    new_gt_path = os.path.join("./data/gt/", fname.split(".")[0] + ".png")
    shutil.copyfile(gt_path, new_gt_path)