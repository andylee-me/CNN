import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator


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
    shear_range=0.2,
    zoom_range=0.2,
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

# 建立 CNN 模型
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# 建立 CNN 模型
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(img_size, img_size, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])


# 編譯模型
model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(learning_rate=0.0001),
    metrics=['accuracy']
)

# 定義 EarlyStopping 回呼函數
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
ModelCheckpoint('best_model.keras', save_best_only=True),
ReduceLROnPlateau(patience=2)

# 開始訓練
model.fit(
    train_gen,
    epochs=20,
    validation_data=val_gen,
    callbacks=[early_stop]  # ← 注意：這裡才使用 early_stop，並且要在上面先定義好
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

