import os

# 建立基本專案結構與必要檔案內容
project_structure = {
    "cat_dog_classifier/": {
        "train_cnn.py": '''import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# 資料夾路徑
train_dir = 'kaggle_cats_vs_dogs_f/train'
val_dir = 'kaggle_cats_vs_dogs_f/val'

# 圖像參數
img_size = (128, 128)
batch_size = 32

# 圖像增強與資料生成器
train_datagen = ImageDataGenerator(rescale=1./255)
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
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# 模型訓練
model.fit(
    train_gen,
    epochs=10,
    validation_data=val_gen
)
''',
        "environment.yml": '''name: catdog-env
channels:
  - defaults
dependencies:
  - python=3.10
  - pip
  - pip:
    - tensorflow
    - matplotlib
    - numpy
'''
    }
}

base_path = "file"  # 改成相對目錄
for folder, files in project_structure.items():
    folder_path = os.path.join(base_path, folder)
    os.makedirs(folder_path, exist_ok=True)
    for filename, content in files.items():
        file_path = os.path.join(folder_path, filename)
        with open(file_path, "w") as f:
            f.write(content)


# 顯示結果目錄
import pandas as pd

file_list = []
for folder, files in project_structure.items():
    for filename in files:
        file_list.append({"檔案": filename, "路徑": os.path.join(folder, filename)})

df = pd.DataFrame(file_list)
# 儲存模型
model.save("file/model/catdog_model.h5")
