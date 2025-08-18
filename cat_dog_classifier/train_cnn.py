import os, json, shutil, tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import AdamW

# ============== 基本參數（無預訓練） ==============
IMG_SIZE = (192, 192)            # 先用 192x192 加速；穩定再改回 224
BATCH_SIZE = 64                  # 盡量加大，能提速也穩梯度
EPOCHS = 80
TRAIN_DIR = "file/kaggle_cats_vs_dogs_f/train"
VAL_DIR   = "file/kaggle_cats_vs_dogs_f/val"
MODEL_DIR = "model"; os.makedirs(MODEL_DIR, exist_ok=True)
BEST_PATH = os.path.join(MODEL_DIR, "catdog_model.h5")
IDX_PATH  = os.path.join(MODEL_DIR, "class_indices.json")
MIS_DIR   = "misclassified_vgg16_scratch"

# ============== 加速開關 ==============
# 混合精度（Apple/M-series or 支援半精度的 GPU 上很有感）
mixed_precision.set_global_policy('mixed_float16')
AUTOTUNE = tf.data.AUTOTUNE

# ============== tf.data 載入 ==============
def build_ds(root, training):
    ds = tf.keras.utils.image_dataset_from_directory(
        root, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
        label_mode='binary', shuffle=training
    )
    # VGG 的 'caffe' 前處理（與 rescale=1/255 不同！）
    def _pp(x, y):
        x = tf.cast(x, tf.float32)           # 到 float32 再做 caffe preprocess
        x = preprocess_input(x, mode='caffe')
        return x, y
    ds = ds.map(_pp, num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.cache().shuffle(2048)
    else:
        ds = ds.cache()
    ds = ds.prefetch(AUTOTUNE)
    return ds

train_ds = build_ds(TRAIN_DIR, training=True)
val_ds   = build_ds(VAL_DIR,   training=False)

# 儲存 class_indices（由 dataset 讀不到，手動取目錄順序）
class_names = sorted(next(os.walk(TRAIN_DIR))[1])
class_indices = {name: i for i, name in enumerate(class_names)}
with open(IDX_PATH, "w") as f: json.dump(class_indices, f, indent=2, ensure_ascii=False)
print("🗂️ class_indices:", class_indices)

# ============== 模型（VGG16 從零開始） ==============
base = VGG16(include_top=False, weights=None, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
for l in base.layers:
    l.trainable = True  # 從零訓練，全部可訓

x = base.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu", dtype="float32")(x)  # head 用 float32 更穩
x = Dropout(0.35)(x)
out = Dense(1, activation="sigmoid", dtype="float32")(x)

model = Model(inputs=base.input, outputs=out)

# AdamW + weight decay，學習率適中；配合 ReduceLROnPlateau
opt = AdamW(learning_rate=1e-3, weight_decay=1e-4)
model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])

model.summary()

# ============== Callbacks ==============
cbs = [
    ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5, min_lr=1e-6, verbose=1),
    EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
    ModelCheckpoint(BEST_PATH, monitor="val_accuracy", save_best_only=True, verbose=1)
]

# ============== 訓練 ==============
history = model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds, callbacks=cbs)

# 另存最終
model.save(os.path.join(MODEL_DIR, "catdog_model.h5"))

# ============== 評估 & 匯出 misclassified ==============
print("\n===== 評估與錯誤樣本匯出 =====")
best = load_model(BEST_PATH)

# 方便用 dataset 的檔路徑：再建一份  batch=1 的驗證/訓練 generator（只做推論）
from tensorflow.keras.preprocessing.image import ImageDataGenerator
eval_datagen = ImageDataGenerator(preprocessing_function=lambda x: preprocess_input(x, mode="caffe"))

def dump_mis(dataset_name, root_dir):
    gen = eval_datagen.flow_from_directory(
        root_dir, target_size=IMG_SIZE, batch_size=1,
        class_mode='binary', shuffle=False
    )
    total = len(gen.filepaths)
    probs = best.predict(gen, verbose=1)
    preds = (probs > 0.5).astype(int).flatten()
    trues = gen.classes
    paths = gen.filepaths

    # 建立輸出資料夾
    dest = os.path.join(MIS_DIR, dataset_name)
    if os.path.exists(dest): shutil.rmtree(dest)
    os.makedirs(dest, exist_ok=True)
    for name in class_names: os.makedirs(os.path.join(dest, name), exist_ok=True)

    wrong = 0
    for i, (t, p) in enumerate(zip(trues, preds)):
        if int(t) != int(p):
            src = paths[i]; fname = os.path.basename(src)
            true_name = class_names[int(t)]
            shutil.copy(src, os.path.join(dest, true_name, f"wrong_pred_{fname}"))
            wrong += 1

    acc = (1 - wrong / total) * 100 if total else 0.0
    print(f"✅ {dataset_name}: {total} 張，錯 {wrong}，準確率 {acc:.2f}%")

for name, path in {"train": TRAIN_DIR, "val": VAL_DIR}.items():
    dump_mis(name, path)

print("\n📂 misclassified_vgg16_scratch 目錄：")
for root, dirs, files in os.walk(MIS_DIR):
    level = root.replace(MIS_DIR, "").count(os.sep)
    indent = "  " * level
    print(f"{indent}📁 {os.path.basename(root)}/")
    for f in files:
        print(f"{indent}  ├─ {f}")
