import os
import json

# 強化版 CNN 模型內容（模仿 MobileNet 結構）
import os
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# 資料夾
train_dir = 'file/kaggle_cats_vs_dogs_f/train'
val_dir = 'file/kaggle_cats_vs_dogs_f/val'

# 參數
img_size = 128
batch_size = 32
epochs = 30

# 資料增強
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
    train_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='binary'
)
val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='binary'
)

# 儲存 class_indices
os.makedirs('file/cat_dog_classifier', exist_ok=True)
with open('file/cat_dog_classifier/class_indices.json', 'w') as f:
    json.dump(train_gen.class_indices, f)

# 建立 MobileNet-like Block
def depthwise_separable_conv(x, pointwise_filters, strides=1):
    x = DepthwiseConv2D(kernel_size=3, strides=strides, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(pointwise_filters, kernel_size=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x

# 模型架構
input_tensor = Input(shape=(img_size, img_size, 3))
x = Conv2D(32, 3, padding='same', activation='relu')(input_tensor)
x = depthwise_separable_conv(x, 64)
x = depthwise_separable_conv(x, 128, strides=2)
x = Dropout(0.25)(x)
x = depthwise_separable_conv(x, 128)
x = depthwise_separable_conv(x, 256, strides=2)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=input_tensor, outputs=output)

# 編譯
model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# Callback
callbacks = [
    EarlyStopping(patience=5, monitor='val_loss', restore_best_weights=True),
    ModelCheckpoint('model/catdog_model.h5', save_best_only=True),
    ReduceLROnPlateau(patience=2)
]

# 建立儲存資料夾
os.makedirs('model', exist_ok=True)

# 訓練模型
model.fit(
    train_gen,
    epochs=epochs,
    validation_data=val_gen,
    callbacks=callbacks
)



