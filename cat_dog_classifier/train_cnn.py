import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

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
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
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

# 建立 CNN 模型
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),  # 防止過擬合
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# ✅ 加入 EarlyStopping + ModelCheckpoint
os.makedirs("model", exist_ok=True)
callbacks_list = [
    callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    callbacks.ModelCheckpoint("model/catdog_model.h5", save_best_only=True)
]

# 模型訓練
model.fit(
    train_gen,
    epochs=30,
    validation_data=val_gen,
    callbacks=callbacks_list
)


# 在您的訓練腳本中，確保模型保存路徑正確
import os

# 創建目錄（如果不存在）
os.makedirs('model', exist_ok=True)

# 保存模型
model.save('model/catdog_model.h5')

# 驗證文件是否存在
if os.path.exists('model/catdog_model.h5'):
    print("Model saved successfully!")
else:
    print("Model save failed!")



# 儲存到檔案系統
os.makedirs("file/cat_dog_classifier", exist_ok=True)

# 建立檔案列表 DataFrame
df = pd.DataFrame([
    {"檔案": "train_cnn.py", "路徑": "file/cat_dog_classifier/train_cnn.py"},
    {"檔案": "environment.yml", "路徑": "file/environment.yml"},
])

