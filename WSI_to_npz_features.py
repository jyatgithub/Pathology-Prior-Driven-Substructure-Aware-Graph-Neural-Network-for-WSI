# Copyright (c) 2021 Wenqi Lu, Fayyaz Minhas, University of Warwick
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import numpy as np
import openslide
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects, binary_dilation, disk
from tqdm import tqdm

# ============================================================
# 配置参数
# ============================================================
WSI_DIR = 'path/to/your/WSI/directory'           # SVS文件目录
OUTPUT_DIR = 'path/to/your/output/directory'   # NPZ输出目录
PATCH_SIZE = 256               # patch大小（像素，20x分辨率下）
PATCH_LEVEL = 0                # 读取倍率层级（0=最高分辨率）
STRIDE = 256                   # 步长（等于patch_size即不重叠）
TISSUE_THRESHOLD = 0.5         # patch中组织占比阈值
BATCH_SIZE = 64                # 特征提取batch size
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Step 1：加载特征提取模型
# ============================================================
def load_feature_extractor():
    model = models.resnet50(pretrained=True)
    model.fc = torch.nn.Identity()  # 去掉分类头，输出2048维
    model = model.to(DEVICE)
    model.eval()
    return model

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# Step 2：组织分割，返回mask
# ============================================================
def get_tissue_mask(slide):
    # 在低倍率下做组织分割，速度快
    # 取level 2或最低倍率缩略图
    level = slide.level_count - 1
    dims = slide.level_dimensions[level]
    thumbnail = slide.read_region((0, 0), level, dims).convert('RGB')
    thumb_np = np.array(thumbnail)

    # 转灰度，Otsu阈值
    gray = np.mean(thumb_np, axis=2)
    thresh = threshold_otsu(gray)
    binary = gray < thresh  # 组织比背景暗

    # 形态学处理去噪
    binary = remove_small_objects(binary, min_size=64)
    binary = binary_dilation(binary, disk(3))

    # 计算缩放比例（低倍率mask -> 高倍率坐标）
    scale_x = slide.dimensions[0] / dims[0]
    scale_y = slide.dimensions[1] / dims[1]

    return binary, scale_x, scale_y


# ============================================================
# Step 3：提取patch坐标（只保留有组织的patch）
# ============================================================
def get_patch_coords(slide, tissue_mask, scale_x, scale_y):
    w, h = slide.dimensions
    coords = []

    for y in range(0, h - PATCH_SIZE, STRIDE):
        for x in range(0, w - PATCH_SIZE, STRIDE):
            # 对应到低倍率mask的位置
            mask_x = int(x / scale_x)
            mask_y = int(y / scale_y)
            mask_x = min(mask_x, tissue_mask.shape[1] - 1)
            mask_y = min(mask_y, tissue_mask.shape[0] - 1)

            # 检查该区域在mask中的组织占比
            mask_patch_x_end = min(
                int((x + PATCH_SIZE) / scale_x), tissue_mask.shape[1]
            )
            mask_patch_y_end = min(
                int((y + PATCH_SIZE) / scale_y), tissue_mask.shape[0]
            )
            mask_region = tissue_mask[
                mask_y:mask_patch_y_end,
                mask_x:mask_patch_x_end
            ]

            if mask_region.size == 0:
                continue
            tissue_ratio = mask_region.sum() / mask_region.size
            if tissue_ratio >= TISSUE_THRESHOLD:
                coords.append((x, y))

    return coords


# ============================================================
# Step 4：批量提取特征
# ============================================================
def extract_features(slide, coords, model):
    features = []
    batch_imgs = []
    batch_coords = []

    for i, (x, y) in enumerate(tqdm(coords, desc='Extracting features')):
        patch = slide.read_region(
            (x, y), PATCH_LEVEL, (PATCH_SIZE, PATCH_SIZE)
        ).convert('RGB')
        batch_imgs.append(transform(patch))
        batch_coords.append((x, y))

        # 攒够一个batch再推理
        if len(batch_imgs) == BATCH_SIZE or i == len(coords) - 1:
            batch_tensor = torch.stack(batch_imgs).to(DEVICE)
            with torch.no_grad():
                feats = model(batch_tensor)
            features.append(feats.cpu().numpy())
            batch_imgs = []

    features = np.concatenate(features, axis=0)  # (N, 2048)
    return features


# ============================================================
# Step 5：保存为NPZ（SlideGraph+要求的格式）
# ============================================================
def save_npz(coords, features, output_path):
    x_coords = np.array([c[0] for c in coords])
    y_coords = np.array([c[1] for c in coords])
    np.savez(
        output_path,
        x_coordinate=x_coords,
        y_coordinate=y_coords,
        feature=features
    )
    print(f'Saved: {output_path}, patches: {len(coords)}, '
          f'feature dim: {features.shape[1]}')


# ============================================================
# 主函数：处理所有WSI
# ============================================================
def process_all_wsi():
    model = load_feature_extractor()
    wsi_files = [f for f in os.listdir(WSI_DIR) if f.endswith('.svs')]
    print(f'Found {len(wsi_files)} WSI files')

    for wsi_file in wsi_files:
        wsi_path = os.path.join(WSI_DIR, wsi_file)
        wsi_name = os.path.splitext(wsi_file)[0]
        output_path = os.path.join(OUTPUT_DIR, wsi_name + '.npz')

        # 跳过已处理的
        if os.path.exists(output_path):
            print(f'Skipping {wsi_name} (already processed)')
            continue

        print(f'\nProcessing: {wsi_name}')
        try:
            slide = openslide.OpenSlide(wsi_path)
            print(f'  Dimensions: {slide.dimensions}')
            print(f'  Levels: {slide.level_count}')

            # 组织分割
            tissue_mask, scale_x, scale_y = get_tissue_mask(slide)
            print(f'  Tissue ratio: '
                  f'{tissue_mask.sum()/tissue_mask.size:.2%}')

            # 提取有效patch坐标
            coords = get_patch_coords(
                slide, tissue_mask, scale_x, scale_y
            )
            print(f'  Valid patches: {len(coords)}')

            if len(coords) == 0:
                print(f'  WARNING: No valid patches found, skipping')
                continue

            # 特征提取
            features = extract_features(slide, coords, model)

            # 保存NPZ
            save_npz(coords, features, output_path)
            slide.close()

        except Exception as e:
            print(f'  ERROR processing {wsi_name}: {e}')
            continue

    print('\nAll done!')


if __name__ == '__main__':
    process_all_wsi()