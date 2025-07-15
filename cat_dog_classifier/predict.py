import os
import shutil
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1️⃣ 模型路徑
model_path = "model/catdog_model.h5"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ 找不到模型檔案 {model_path}，請先訓練模型！")
model = load_model(model_path)

# 2️⃣ 定義資料夾
train_dir = "file/kaggle_cats_vs_dogs_f/train"
val_dir = "file/kaggle_cats_vs_dogs_f/val"
img_size = (128, 128)

# 3️⃣ 預處理器
datagen = ImageDataGenerator(rescale=1./255)

# 4️⃣ 要處理的資料集
datasets = {
    "train": train_dir,
    "val": val_dir
}

# 5️⃣ 建立 misclassified 資料夾（先刪除舊的，確保乾淨）
misclassified_dir = "misclassified"
if os.path.exists(misclassified_dir):
    print("🧹 清理舊的 misclassified 資料夾...")
    shutil.rmtree(misclassified_dir)
os.makedirs(misclassified_dir, exist_ok=True)

def ensure_subdirs(base_path):
    """確保 base_path 下有 cat/ dog/ 兩個子資料夾"""
    for cls in ["cat", "dog"]:
        os.makedirs(os.path.join(base_path, cls), exist_ok=True)

def evaluate_and_save(dataset_name, dataset_path):
    """預測資料集，並將錯誤圖片複製到 misclassified/{dataset_name}/"""
    print(f"\n🔍 開始檢查 {dataset_name} 資料集...")

    gen = datagen.flow_from_directory(
        dataset_path,
        target_size=img_size,
        batch_size=1,   # 一次一張，方便對應
        class_mode='binary',
        shuffle=False
    )

    # 預測
    pred_probs = model.predict(gen, verbose=1)
    pred_labels = (pred_probs > 0.5).astype(int).flatten()
    true_labels = gen.classes
    file_paths = gen.filepaths

    # 建立 dataset 對應的 misclassified 子資料夾
    dataset_mis_dir = os.path.join(misclassified_dir, dataset_name)
    os.makedirs(dataset_mis_dir, exist_ok=True)
    ensure_subdirs(dataset_mis_dir)

    # 統計錯誤
    misclassified_count = 0
    for idx, (true_label, pred_label) in enumerate(zip(true_labels, pred_labels)):
        if true_label != pred_label:
            src_path = file_paths[idx]
            filename = os.path.basename(src_path)

            # 根據真實標籤分類到 cat/dog
            if true_label == 0:  # 真實是 cat
                dst_path = os.path.join(dataset_mis_dir, "cat", f"wrong_pred_{filename}")
            else:  # 真實是 dog
                dst_path = os.path.join(dataset_mis_dir, "dog", f"wrong_pred_{filename}")

            # 複製檔案
            shutil.copy(src_path, dst_path)
            misclassified_count += 1

    # 計算準確率
    total_images = len(true_labels)
    accuracy = (1 - misclassified_count / total_images) * 100
    print(f"✅ {dataset_name} 集總共 {total_images} 張")
    print(f"❌ 錯誤 {misclassified_count} 張，正確率 {accuracy:.2f}%")

    return total_images, misclassified_count, accuracy

# 6️⃣ 執行 train 與 val 的錯誤檢查
results = {}
for name, path in datasets.items():
    total, wrong, acc = evaluate_and_save(name, path)
    results[name] = {"total": total, "wrong": wrong, "accuracy": acc}

# 7️⃣ 最後輸出總結
print("\n📊 最終結果總結：")
for ds, info in results.items():
    print(f"➡ {ds}: {info['accuracy']:.2f}% (錯誤 {info['wrong']}/{info['total']})")

print("\n✅ 所有錯誤圖片已複製到 misclassified/ 資料夾內")
