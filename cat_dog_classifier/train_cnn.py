# train_cnn.py
import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, BatchNormalization, ReLU, MaxPooling2D, GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# 資料夾路徑
train_dir = 'file/kaggle_cats_vs_dogs_f/train'
val_dir = 'file/kaggle_cats_vs_dogs_f/val'

# 圖像參數
img_size = (128, 128)
batch_size = 32

# 資料增強
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    brightness_range=[0.7, 1.3],
    horizontal_flip=True,
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'
)
val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'
)

# 模仿 MobileNetV2 結構：使用 Depthwise Separable Convolution
model = Sequential([
    Conv2D(32, (3, 3), padding='same', input_shape=(128, 128, 3)),
    BatchNormalization(),
    ReLU(),

    DepthwiseConv2D(kernel_size=(3,3), padding='same'),
    BatchNormalization(),
    ReLU(),
    Conv2D(64, (1,1), padding='same'),
    BatchNormalization(),
    ReLU(),
    MaxPooling2D(2,2),

    DepthwiseConv2D(kernel_size=(3,3), padding='same'),
    BatchNormalization(),
    ReLU(),
    Conv2D(128, (1,1), padding='same'),
    BatchNormalization(),
    ReLU(),
    MaxPooling2D(2,2),

    DepthwiseConv2D(kernel_size=(3,3), padding='same'),
    BatchNormalization(),
    ReLU(),
    Conv2D(256, (1,1), padding='same'),
    BatchNormalization(),
    ReLU(),
    GlobalAveragePooling2D(),

    Dropout(0.4),
    Dense(1, activation='sigmoid')
])

model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(learning_rate=0.0001),
    metrics=['accuracy']
)

# Callback
early_stop = EarlyStopping(patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint("model/best_model.keras", save_best_only=True)
reduce_lr = ReduceLROnPlateau(patience=3, factor=0.5, verbose=1)

# 訓練
model.fit(
    train_gen,
    epochs=20,
    validation_data=val_gen,
    callbacks=[early_stop, checkpoint, reduce_lr]
)

# 儲存模型
os.makedirs('model', exist_ok=True)
model.save('model/catdog_model.h5')

# 儲存紀錄
df = pd.DataFrame([
    {"檔案": "train_cnn.py", "路徑": "file/cat_dog_classifier/train_cnn.py"},
    {"檔案": "predict.py", "路徑": "file/cat_dog_classifier/predict.py"},
])
print("Model training and saving complete.")
