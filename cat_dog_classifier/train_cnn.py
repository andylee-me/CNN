import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, BatchNormalization, ReLU, Add, GlobalAveragePooling2D, Dense



def inverted_residual_block(x, expansion, out_channels, strides):
    in_channels = x.shape[-1]

    # 1x1 Expand
    expanded = Conv2D(in_channels * expansion, (1, 1), padding='same', use_bias=False)(x)
    expanded = BatchNormalization()(expanded)
    expanded = ReLU(6.)(expanded)

    # 3x3 Depthwise
    dw = DepthwiseConv2D(kernel_size=(3, 3), strides=strides, padding='same', use_bias=False)(expanded)
    dw = BatchNormalization()(dw)
    dw = ReLU(6.)(dw)

    # 1x1 Project
    projected = Conv2D(out_channels, (1, 1), padding='same', use_bias=False)(dw)
    projected = BatchNormalization()(projected)

    # Shortcut Connection (if input and output have same shape)
    if strides == 1 and in_channels == out_channels:
        return Add()([x, projected])
    else:
        return projected

def build_mobilenet_cnn(input_shape=(128, 128, 3), num_classes=1):
    inputs = Input(shape=input_shape)

    # Initial layer
    x = Conv2D(32, (3, 3), strides=(2, 2), padding='same', use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = ReLU(6.)(x)

    # 模仿 MobileNetV2 結構（部分簡化）
    x = inverted_residual_block(x, expansion=1, out_channels=16, strides=1)
    x = inverted_residual_block(x, expansion=6, out_channels=24, strides=2)
    x = inverted_residual_block(x, expansion=6, out_channels=24, strides=1)
    x = inverted_residual_block(x, expansion=6, out_channels=32, strides=2)
    x = inverted_residual_block(x, expansion=6, out_channels=32, strides=1)

    # Global Pooling + Dense
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(num_classes, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    return model


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
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model

base_model = MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # 冻結預訓練權重

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(1, activation='sigmoid')(x)

from tensorflow.keras.optimizers import Adam
model = build_mobilenet_cnn(input_shape=(128, 128, 3), num_classes=1)
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

