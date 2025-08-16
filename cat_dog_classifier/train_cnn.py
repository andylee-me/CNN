import os
import json
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.imagenet_utils import preprocess_input

# =========================
# 🔧 參數設定
# =========================
# 是否使用預訓練權重（建議 True）
USE_PRETRAINED = False             # True: ImageNet 預訓練；False: 從頭訓練
UNFREEZE_LAST_BLOCK = True        # True: 微調 VGG16 的最後一個 conv_block（更高準確率）
img_size = (224, 224)             # VGG 的標準輸入尺寸
batch_size = 32
train_dir = "file/kaggle_cats_vs_dogs_f/train"
val_dir   = "file/kaggle_cats_vs_dogs_f/val"
model_dir = "model"
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "vgg16_catdog.h5")
class_indices_path = os.path.join(model_dir, "class_indices.json")

# =========================
# 🔁 資料前處理（VGG = 'caffe' 模式）
# - 不要再 rescale=1./255
# - 使用 preprocess_input(..., mode='caffe')
# =========================
train_datagen = ImageDataGenerator(
    preprocessing_function=lambda x: preprocess_input(x, mode="caffe"),
    rotation_range=8,
    width_shift_range=0.03,
    height_shift_range=0.03,
    zoom_range=0.08,
    horizontal_flip=True,
    fill_mode='reflect'
)

val_datagen = ImageDataGenerator(
    preprocessing_function=lambda x: preprocess_input(x, mode="caffe")
)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True
)

val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

# 儲存 class_indices 供預測使用
with open(class_indices_path, "w") as f:
    json.dump(train_gen.class_indices, f, indent=2, ensure_ascii=False)
print("🗂️ Saved class_indices:", train_gen.class_indices)

# =========================
# 🧱 模型：VGG16 backbone + 自訂分類頭
# =========================
weights = "imagenet" if USE_PRETRAINED else None
base = VGG16(include_top=False, weights=weights, input_shape=(img_size[0], img_size[1], 3))

# 預設凍結全部
for layer in base.layers:
    layer.trainable = False

# 可選：解凍最後一個 block（block5_*）
if USE_PRETRAINED and UNFREEZE_LAST_BLOCK:
    for layer in base.layers:
        if layer.name.startswith("block5"):
            layer.trainable = True

x = base.output
# VGG16 原生是 Flatten + FC，這裡用 GAP + 小頭，訓練更穩定
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.3)(x)
out = Dense(1, activation="sigmoid")(x)  # 二分類

model = Model(inputs=base.input, outputs=out)

# =========================
# ⚙ 編譯
# - 如果從頭訓練，建議把 lr 調低一點並延長 epoch
# =========================
init_lr = 1e-4 if USE_PRETRAINED else 5e-4
model.compile(optimizer=Adam(learning_rate=init_lr),
              loss="binary_crossentropy",
              metrics=["accuracy"])

model.summary()

# =========================
# ⏱️ Callbacks
# =========================
callbacks = [
    ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=5, verbose=1, min_lr=1e-6),
    EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True, verbose=1),
    ModelCheckpoint(model_path, monitor="val_accuracy", save_best_only=True, verbose=1)
]

# =========================
# 🚀 訓練
# =========================
epochs = 60 if USE_PRETRAINED else 100
history = model.fit(
    train_gen,
    epochs=epochs,
    validation_data=val_gen,
    callbacks=callbacks,
    workers=4,
    use_multiprocessing=True
)

# 另存一份最終
model.save(os.path.join(model_dir, "vgg16_catdog_final.h5"))
print("✅ Saved:", os.path.join(model_dir, "vgg16_catdog_final.h5"))

# =========================
# 🔎 評估＆輸出 misclassified 影像
# =========================
print("\n================  評估與錯誤樣本匯出  ================")

# 重新載入最佳模型
assert os.path.exists(model_path), f"❌ 找不到模型 {model_path}"
print(f"✅ 載入最佳模型: {model_path}")
best_model = load_model(model_path)

# 與訓練一致的前處理
eval_datagen = ImageDataGenerator(
    preprocessing_function=lambda x: preprocess_input(x, mode="caffe")
)

datasets = {"train": train_dir, "val": val_dir}
misclassified_root = "misclassified_vgg16"

# 先清空舊的
if os.path.exists(misclassified_root):
    shutil.rmtree(misclassified_root)
os.makedirs(misclassified_root, exist_ok=True)

# index->name 對應（用剛才存的 class_indices）
with open(class_indices_path, "r") as f:
    class_indices = json.load(f)
idx_to_name = {v: k for k, v in class_indices.items()}

def ensure_subdirs(base_path, class_names=("cat","dog")):
    for cls in class_names:
        os.makedirs(os.path.join(base_path, cls), exist_ok=True)

def evaluate_and_save(dataset_name, dataset_path):
    print(f"\n🚀 開始檢查 {dataset_name} 資料集...")

    gen = eval_datagen.flow_from_directory(
        dataset_path,
        target_size=img_size,
        batch_size=1,
        class_mode='binary',
        shuffle=False
    )

    total_images = len(gen.filepaths)
    print(f"➡ 共找到 {total_images} 張圖片")

    pred_probs = best_model.predict(gen, verbose=1)
    pred_labels = (pred_probs > 0.5).astype(int).flatten()
    true_labels = gen.classes
    file_paths  = gen.filepaths

    dataset_mis_dir = os.path.join(misclassified_root, dataset_name)
    os.makedirs(dataset_mis_dir, exist_ok=True)
    ensure_subdirs(dataset_mis_dir, class_names=list(idx_to_name.values()))

    misclassified_count = 0
    debug_samples = 0

    for idx, (t, p) in enumerate(zip(true_labels, pred_labels)):
        if t != p:
            src_path = file_paths[idx]
            filename = os.path.basename(src_path)

            if not os.path.exists(src_path):
                print(f"⚠️ 找不到原始檔案: {src_path}")
                continue

            true_name = idx_to_name.get(int(t), str(t))
            dst_path = os.path.join(dataset_mis_dir, true_name, f"wrong_pred_{filename}")
            shutil.copy(src_path, dst_path)
            misclassified_count += 1

            if debug_samples < 5:
                print(f"❌ Misclassified -> {src_path} → {dst_path}")
                debug_samples += 1

    acc = (1 - misclassified_count / total_images) * 100 if total_images > 0 else 0.0
    print(f"✅ {dataset_name} 集: 共 {total_images} 張, 錯誤 {misclassified_count}, 準確率 {acc:.2f}%")

for name, path in datasets.items():
    evaluate_and_save(name, path)

print("\n📂 最終 misclassified_vgg16 資料夾結構：")
for root, dirs, files in os.walk(misclassified_root):
    level = root.replace(misclassified_root, "").count(os.sep)
    indent = " " * 2 * level
    print(f"{indent}📁 {os.path.basename(root)}/")
    for f in files:
        print(f"{indent}  ├── {f}")
