# train_cnn.py
import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, BatchNormalization, ReLU, MaxPooling2D, GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# === train_cnn.py ===
import os
import json
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# 資料夾路徑
train_dir = 'file/kaggle_cats_vs_dogs_f/train'
val_dir = 'file/kaggle_cats_vs_dogs_f/val'

# 圖像參數
img_size = (128, 128)
batch_size = 32

# 圖像增強與資料生成器
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir, target_size=img_size, batch_size=batch_size,
    class_mode='binary', shuffle=True
)
val_gen = val_datagen.flow_from_directory(
    val_dir, target_size=img_size, batch_size=batch_size,
    class_mode='binary', shuffle=False
)

# 儲存類別對應
with open('file/cat_dog_classifier/class_indices.json', 'w') as f:
    json.dump(train_gen.class_indices, f)

# 模仿 MobileNetV2 風格的模型
def build_mobilenet_like(input_shape):
    inputs = tf.keras.Input(shape=input_shape)

    x = layers.Rescaling(1./255)(inputs)

    for filters in [32, 64, 128]:
        x = layers.DepthwiseConv2D(kernel_size=3, padding='same')(x)
        x = layers.Conv2D(filters, kernel_size=1)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    return models.Model(inputs, outputs)

model = build_mobilenet_like((128, 128, 3))

model.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    metrics=['accuracy']
)

# 回呼函數
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ReduceLROnPlateau(patience=3, factor=0.5, verbose=1),
    ModelCheckpoint('model/catdog_model.h5', save_best_only=True)
]

# 建立資料夾
os.makedirs('model', exist_ok=True)

# 訓練模型
model.fit(
    train_gen,
    epochs=20,
    validation_data=val_gen,
    callbacks=callbacks
)

import json

# 儲存 class_indices
with open("file/cat_dog_classifier/class_indices.json", "w") as f:
    json.dump(train_gen.class_indices, f)

