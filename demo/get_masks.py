import os 
import time

root_folders = {
    "gt": "/scratch/mchasmai/work/cs670/TestDataset/COD10K/GT",
    "sam": "/scratch/mchasmai/work/cs670/sam_predictions/COD10K",
    "sinet_base": "/scratch/mchasmai/work/cs670/sinet_r18_predictions/COD10K",
    "sinet_style": "/scratch/mchasmai/work/cs670/sinet_style_predictions/style_and_original/COD10K",
    "sinet_synth": "/scratch/mchasmai/work/cs670/sinet_r18_predictions_synthetic/synthetic_and_cod_balance/COD10K"
}

image_dir = "./data/images"

for dir_name in root_folders:
    os.makedirs(os.path.join("./data", dir_name), exist_ok=True)
    for fname in os.listdir(image_dir):
        seg_name = fname
        if "sam" not in dir_name:
            seg_name = seg_name.replace(".jpg", ".png")
        ssh_path = os.path.join(root_folders[dir_name], seg_name)
        os.system("scp euclid:{} ./data/{}/{}".format(ssh_path, dir_name, seg_name))
        time.sleep(10)