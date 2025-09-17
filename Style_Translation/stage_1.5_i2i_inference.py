import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"
import torch
from sklearn.cluster import KMeans
import numpy as np
from i2i_solver import i2iSolver
import random
import argparse
import nibabel as nib
import SimpleITK as sitk
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    "--ckpt_path",
    type=str,
    default="/databak/results/lichangrui/Previous_Models/DAR-UNet/datasets/CT2PET_Large/i2i_checkpoints/enc_0200.pt",
)
parser.add_argument(
    "--source_img_dirpath",
    type=str,
    default="/databak/results/lichangrui/datasets/large_Model/Training/CT_preprocess/Img-Img",
)
parser.add_argument(
    "--target_img_dirpath",
    type=str,
    default="/databak/results/lichangrui/datasets/large_Model/Training/PET_preprocess/Img",
)
parser.add_argument(
    "--save_nii_dirpath",
    type=str,
    default="/databak/results/lichangrui/datasets/large_Model/Stage1_Inference/CT2PET",
)


opts = parser.parse_args()
if not os.path.exists(opts.save_nii_dirpath):
    os.makedirs(opts.save_nii_dirpath)
trainer = i2iSolver(None)
state_dict = torch.load(opts.ckpt_path)
trainer.enc_c.load_state_dict(state_dict["enc_c"])
trainer.enc_s_a.load_state_dict(state_dict["enc_s_a"])
trainer.enc_s_b.load_state_dict(state_dict["enc_s_b"])
trainer.dec.load_state_dict(state_dict["dec"])
trainer.cuda()

target_images = os.listdir(opts.target_img_dirpath)
for f in os.listdir(opts.source_img_dirpath):
    imgs = nib.load(os.path.join(opts.source_img_dirpath, f)).get_fdata()

    idx = random.randint(0, len(target_images) - 1)
    target_img = nib.load(os.path.join(opts.target_img_dirpath, target_images[idx])).get_fdata()
    target_img = target_img[:, :, int(target_img.shape[-1] / 2)]
    with torch.no_grad():
        single_img = (
            torch.from_numpy((target_img * 2 - 1))
            .unsqueeze(0)
            .unsqueeze(0)
            .cuda()
            .float()
        )
        s = trainer.enc_s_b(single_img)[0].unsqueeze(0)
    nimgs = np.zeros_like(imgs, dtype=np.float32)
    for i in range(imgs.shape[-1]):
        img = imgs[:, :, i]
        single_img = (
            torch.from_numpy((img * 2 - 1)).unsqueeze(0).unsqueeze(0).cuda().float()
        )
        transfered_img = trainer.inference(single_img, s)
        transfered_img = (((transfered_img + 1) / 2).cpu().numpy()).astype(np.float32)[
            0, 0
        ]
        nimgs[:, :, i] = transfered_img
    nimgs = nimgs.transpose(2, 0, 1)
    img = sitk.GetImageFromArray(nimgs)
    sitk.WriteImage(
        img, os.path.join(opts.save_nii_dirpath, f.replace(".npy", ".nii.gz"))
    )
    print("End processing %s." % f)
