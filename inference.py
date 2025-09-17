import os
import sys
import shutil
import time
import glob
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.common_function import parse_option
from engines.common import Inference

def run_inference_for_modality(modality, nii_files, config_path, config):
    tmp_dir = "/workspace/tmp_input"
    os.makedirs(tmp_dir, exist_ok=True)

    if modality == "mr":
        selected_files = [f for f in nii_files if os.path.basename(f).lower().startswith(("mr", "amos"))]
    elif modality == "pet":
        selected_files = [f for f in nii_files if not os.path.basename(f).lower().startswith(("mr", "amos"))]

    if not selected_files:
        print(f"No {modality.upper()} images found, skipping...")
        return

    for f in selected_files:
        shutil.copy(f, tmp_dir)

    config.defrost()
    config.DATASET.VAL_IMAGE_PATH = tmp_dir

    if modality == "mr":
        print("Running inference for MR...")
        config.TRAINING_TYPE = "coarse-fine"
        config.FINE_MODEL_PATH = "/workspace/MR_checkpoints"
        config.CHECK_POINT_NAME = "fine_checkpoint.pth"
        config.COARSE_MODEL_PATH = "/workspace/MR_checkpoints"
        config.COARSE_CHECK_POINT_NAME = "coarse_checkpoint.pth"
        config.MODEL.FINE.BASE_NUM_FEATURES = 24
        config.MODEL.FINE.NUM_HEADS = [3, 6, 12, 24]
        config.DATASET.FINE.SIZE = [96, 192, 192]
        config.DATASET.FINE.PREPROCESS_SIZE = [192, 192, 96]
    elif modality == "pet":
        print("Running inference for PET...")
        config.TRAINING_TYPE = "fine"
        config.FINE_MODEL_PATH = "/workspace/PET_checkpoints"
        config.CHECK_POINT_NAME = "500_checkpoint.pth"
        config.DATASET.FINE.SIZE = [96, 256, 256]
        config.DATASET.FINE.PREPROCESS_SIZE = [96, 256, 256]

    predict = Inference(config)
    predict.run()

    shutil.rmtree(tmp_dir)
    print(f"Finished inference for {modality.upper()} and deleted temporary files.")

if __name__ == "__main__":
    torch.cuda.synchronize()
    t_start = time.time()

    _, config = parse_option("other", "/workspace/configs/inference/PET_big_inference_2.yaml")

    val_image_path = config.DATASET.VAL_IMAGE_PATH
    nii_files = sorted(glob.glob(os.path.join(val_image_path, "*.nii.gz")))
    if not nii_files:
        raise RuntimeError(f"No .nii.gz files found in {val_image_path}")

    # 分别处理 MR 和 PET
    run_inference_for_modality("mr", nii_files, "/workspace/configs/inference/PET_big_inference_2.yaml", config.clone())
    run_inference_for_modality("pet", nii_files, "/workspace/configs/inference/PET_big_inference_2.yaml", config.clone())

    torch.cuda.synchronize()
    t_end = time.time()
    print("Total_time: {:.2f} s".format(t_end - t_start))