import os
import json
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import regularizers
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow.keras.applications.imagenet_utils import preprocess_input

# -----------------------
# 🔧 基本參數
# -----------------------
img_size = (224, 224)
batch_size = 32
train_dir = "file/kaggle_cats_vs_dogs_f/train"
val_dir = "file/kaggle_cats_vs_dogs_f/val"
model_dir = "model"
os.makedirs(model_dir, exist_ok=True)

# 可選：開啟混合精度（若你的 GPU 支援，能加速且省顯存）
try:
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_policy(policy)
    print("⚡ Using mixed precision (mixed_float16).")
except Exception as e:
    print("ℹ️ Mixed precision not enabled:", e)

# -----------------------
# 🔁 前處理與資料增強
#   - 與 PyTorch 版本對齊：ImageNet mean/std（mode='torch'）
#   - 不再使用 rescale=1./255
# -----------------------
train_datagen = ImageDataGenerator(
    preprocessing_function=lambda x: preprocess_input(x, mode="torch"),
    rotation_range=8,
    width_shift_range=0.03,
    height_shift_range=0.03,
    zoom_range=0.08,
    horizontal_flip=True,
    fill_mode='reflect'
)

val_datagen = ImageDataGenerator(
    preprocessing_function=lambda x: preprocess_input(x, mode="torch")
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

# 儲存 class_indices 供 predict.py 使用
with open(os.path.join(model_dir, "class_indices.json"), "w") as f:
    json.dump(train_gen.class_indices, f, indent=2, ensure_ascii=False)
print("🗂️ Saved class_indices.json:", train_gen.class_indices)

# -----------------------
# 🧱 模型（加深 + GAP + 輕度正則）
# -----------------------
l2 = regularizers.l2(1e-4)

model = Sequential([
    # Block 1
    Conv2D(32, (3,3), padding='same', activation='relu', kernel_regularizer=l2, input_shape=(img_size[0], img_size[1], 3)),
    BatchNormalization(),
    Conv2D(32, (3,3), padding='same', activation='relu', kernel_regularizer=l2),
    BatchNormalization(),
    MaxPooling2D(2,2),

    # Block 2
    Conv2D(64, (3,3), padding='same', activation='relu', kernel_regularizer=l2),
    BatchNormalization(),
    Conv2D(64, (3,3), padding='same', activation='relu', kernel_regularizer=l2),
    BatchNormalization(),
    MaxPooling2D(2,2),

    # Block 3
    Conv2D(128, (3,3), padding='same', activation='relu', kernel_regularizer=l2),
    BatchNormalization(),
    Conv2D(128, (3,3), padding='same', activation='relu', kernel_regularizer=l2),
    BatchNormalization(),
    MaxPooling2D(2,2),

    # Block 4（稍微再深一層）
    Conv2D(256, (3,3), padding='same', activation='relu', kernel_regularizer=l2),
    BatchNormalization(),
    MaxPooling2D(2,2),

    # Head
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # 二分類
])

# 若啟用 mixed precision，最後一層以 float32 計算（Keras 會自動處理大部分情況）
# 但有時需要指定 loss_scale 或確保輸出 dtype，這裡讓 Keras 自動處理

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -----------------------
# ⏱️ Callbacks：學習率排程 + 早停 + 最佳權重保存
# -----------------------
callbacks = [
    ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=5, verbose=1, min_lr=1e-6
    ),
    EarlyStopping(
        monitor='val_loss', patience=15, restore_best_weights=True, verbose=1
    ),
    ModelCheckpoint(
        filepath=os.path.join(model_dir, "catdog_model.h5"),
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True
    )
]

# -----------------------
# 🚀 訓練
# 建議 100–150 epochs；有 LR 排程 + 早停，實際不會跑滿
# -----------------------
history = model.fit(
    train_gen,
    epochs=150,
    validation_data=val_gen,
    callbacks=callbacks,
    workers=4,
    use_multiprocessing=True
)

# 另存一份最終權重（不一定是最佳）
model.save(os.path.join(model_dir, "catdog_model_final.h5"))
print("✅ Saved:", os.path.join(model_dir, "catdog_model_final.h5"))
