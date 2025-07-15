import os
import shutil
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1️⃣ 模型路徑
model_path = "model/catdog_model.h5"
assert os.path.exists(model_path), f"❌ 找不到模型 {model_path}"
model = load_model(model_path)

# 2️⃣ 資料夾
train_dir = "file/kaggle_cats_vs_dogs_f/train"
val_dir = "file/kaggle_cats_vs_dogs_f/val"
img_size = (128, 128)

# 3️⃣ 預處理器
datagen = ImageDataGenerator(rescale=1./255)

datasets = {"train": train_dir, "val": val_dir}

# 4️⃣ 每次先刪掉舊的 misclassified
misclassified_dir = "misclassified"
if os.path.exists(misclassified_dir):
    shutil.rmtree(misclassified_dir)
os.makedirs(misclassified_dir, exist_ok=True)

def ensure_subdirs(base_path):
    for cls in ["cat", "dog"]:
        os.makedirs(os.path.join(base_path, cls), exist_ok=True)

def evaluate_and_save(dataset_name, dataset_path):
    print(f"\n🚀 開始檢查 {dataset_name} 資料集...")
    gen = datagen.flow_from_directory(
        dataset_path,
        target_size=img_size,
        batch_size=1,
        class_mode='binary',
        shuffle=False
    )

    print(f"➡ 共找到 {len(gen.filepaths)} 張圖片")
    pred_probs = model.predict(gen, verbose=1)
    pred_labels = (pred_probs > 0.5).astype(int).flatten()
    true_labels = gen.classes
    file_paths = gen.filepaths

    # 建立 dataset 對應資料夾
    dataset_mis_dir = os.path.join(misclassified_dir, dataset_name)
    os.makedirs(dataset_mis_dir, exist_ok=True)
    ensure_subdirs(dataset_mis_dir)

    misclassified_count = 0
    debug_samples = 0

    for idx, (true_label, pred_label) in enumerate(zip(true_labels, pred_labels)):
        if true_label != pred_label:
            src_path = file_paths[idx]
            filename = os.path.basename(src_path)
            
            # 看看 src_path 到底是不是存在的
            if not os.path.exists(src_path):
                print(f"❌ 找不到原始檔案: {src_path}")
                continue
            
            dst_cls = "cat" if true_label == 0 else "dog"
            dst_path = os.path.join(dataset_mis_dir, dst_cls, f"wrong_pred_{filename}")

            shutil.copy(src_path, dst_path)
            misclassified_count += 1

            if debug_samples < 5:
                print(f"❌ Misclassified -> {src_path} → {dst_path}")
                debug_samples += 1

    total_images = len(true_labels)
    acc = (1 - misclassified_count / total_images) * 100
    print(f"✅ {dataset_name} 集: 共 {total_images} 張, 錯誤 {misclassified_count}, 準確率 {acc:.2f}%")

for name, path in datasets.items():
    evaluate_and_save(name, path)

print("\n📂 檢查 misclassified/ 資料夾結構：")
print(os.listdir(misclassified_dir))
