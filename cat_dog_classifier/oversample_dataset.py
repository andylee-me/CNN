import os
import shutil
import random
from glob import glob
from tqdm import tqdm

# 原始訓練資料夾
src_dir = "file/kaggle_cats_vs_dogs_f/train"

# 輸出 Oversampling 資料夾
dst_dir = "file/kaggle_cats_vs_dogs_f/train_oversampled"
os.makedirs(dst_dir, exist_ok=True)

# 目標類別
classes = ["cat", "dog"]

# 計算每類數量
img_paths = {cls: sorted(glob(os.path.join(src_dir, cls, "*.jpg"))) for cls in classes}
img_counts = {cls: len(paths) for cls, paths in img_paths.items()}
print("📊 原始數量:", img_counts)

# 找出最大類別數量
max_count = max(img_counts.values())

# 針對每一類 oversample
for cls in classes:
    dst_cls_dir = os.path.join(dst_dir, cls)
    os.makedirs(dst_cls_dir, exist_ok=True)

    # 複製原始圖片
    for path in img_paths[cls]:
        shutil.copy(path, os.path.join(dst_cls_dir, os.path.basename(path)))

    # 開始 oversampling
    shortfall = max_count - img_counts[cls]
    if shortfall > 0:
        print(f"🔁 Oversampling {cls}：補 {shortfall} 張")
        for i in tqdm(range(shortfall)):
            src = random.choice(img_paths[cls])
            new_name = f"aug_{i}_{os.path.basename(src)}"
            dst = os.path.join(dst_cls_dir, new_name)
            shutil.copy(src, dst)
    else:
        print(f"✅ {cls} 類已達最大數量，無需補資料")

# 驗證
final_counts = {cls: len(glob(os.path.join(dst_dir, cls, "*.jpg"))) for cls in classes}
print("✅ 完成後的數量:", final_counts)
