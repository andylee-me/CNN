import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, DepthwiseConv2D, BatchNormalization,
                                     ReLU, Add, GlobalAveragePooling2D, Dense)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# 資料夾路徑
train_dir = 'file/kaggle_cats_vs_dogs_f/train'
val_dir = 'file/kaggle_cats_vs_dogs_f/val'
img_size = (128, 128)
batch_size = 32

# 資料增強
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir, target_size=img_size, batch_size=batch_size, class_mode='binary'
)
val_gen = val_datagen.flow_from_directory(
    val_dir, target_size=img_size, batch_size=batch_size, class_mode='binary'
)

# 模仿 MobileNetV2 的 block
def inverted_residual_block(x, expansion, out_channels, strides):
    in_channels = x.shape[-1]

    expanded = Conv2D(in_channels * expansion, (1, 1), padding='same', use_bias=False)(x)
    expanded = BatchNormalization()(expanded)
    expanded = ReLU(6.)(expanded)

    dw = DepthwiseConv2D(kernel_size=(3, 3), strides=strides, padding='same', use_bias=False)(expanded)
    dw = BatchNormalization()(dw)
    dw = ReLU(6.)(dw)

    projected = Conv2D(out_channels, (1, 1), padding='same', use_bias=False)(dw)
    projected = BatchNormalization()(projected)

    if strides == 1 and in_channels == out_channels:
        return Add()([x, projected])
    else:
        return projected

# 建立模型
def build_mobilenet_cnn(input_shape=(128, 128, 3), num_classes=1):
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), strides=(2, 2), padding='same', use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = ReLU(6.)(x)

    x = inverted_residual_block(x, expansion=1, out_channels=16, strides=1)
    x = inverted_residual_block(x, expansion=6, out_channels=24, strides=2)
    x = inverted_residual_block(x, expansion=6, out_channels=24, strides=1)
    x = inverted_residual_block(x, expansion=6, out_channels=32, strides=2)
    x = inverted_residual_block(x, expansion=6, out_channels=32, strides=1)

    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(num_classes, activation='sigmoid')(x)

    return Model(inputs, outputs)

# 編譯模型
model = build_mobilenet_cnn()
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 回調函數
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint('model/best_model.keras', save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
]

# 訓練
model.fit(
    train_gen,
    epochs=20,
    validation_data=val_gen,
    callbacks=callbacks
)

# 保存模型
os.makedirs('model', exist_ok=True)
model.save('model/catdog_model.h5')

# 驗證模型是否儲存成功
if os.path.exists('model/catdog_model.h5'):
    print("✅ Model saved successfully!")
else:
    print("❌ Model save failed.")
