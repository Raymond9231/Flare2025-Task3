import os
import random
import shutil

from pathlib import Path

# 设置路径
folder_A = Path('/home/raymond9231/flare/Random1000MidOut')  # 标签图像，例如 segmentation
folder_B = Path('/home/raymond9231/flare/SmallPETSeg')  # 随机选出的标签图像输出
folder_C = Path('/home/raymond9231/flare/FLARE-Task3-DomainAdaption/train_PET_unlabeled')  # 原始图像
folder_D = Path('/home/raymond9231/flare/SmallPETImg')  # 对应图像输出

# 创建输出文件夹
folder_B.mkdir(parents=True, exist_ok=True)
folder_D.mkdir(parents=True, exist_ok=True)

# 获取A中所有图像文件
all_files = [f for f in folder_A.glob('*.nii.gz')]
assert len(all_files) >= 450, f"A 中文件不足 450 张，只有 {len(all_files)} 张"

# 随机选择450张
selected_files = random.sample(all_files, 450)

for label_path in selected_files:
    filename = label_path.name  # 比如 abc.nii.gz
    stem = filename.replace('.nii.gz', '')  # abc

    # 复制标签图像到B
    shutil.copy(label_path, folder_B / filename)

    # 找对应图像名
    image_name = f"{stem}_0000.nii.gz"
    image_path = folder_C / image_name

    if not image_path.exists():
        print(f"⚠️ 找不到图像: {image_path}")
        continue

    # 复制对应图像到D
    shutil.copy(image_path, folder_D / image_name)

print("✅ 完成：标签图像已复制到 B，对应原图像已复制到 D。")