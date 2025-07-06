import os
import json
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import DepthwiseConv2D, Conv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Dropout, Dense, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# 資料夾路徑
train_dir = 'file/kaggle_cats_vs_dogs_f/train'
val_dir = 'file/kaggle_cats_vs_dogs_f/val'

# 參數
img_size = (128, 128)
batch_size = 32
epochs = 20

# 圖像增強
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
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

# 建立 MobileNetV2-like 模型
def conv_block(x, filters):
    x = DepthwiseConv2D(kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters, kernel_size=1)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x

inputs = Input(shape=(128, 128, 3))
x = Conv2D(32, 3, padding='same')(inputs)
x = BatchNormalization()(x)
x = ReLU()(x)

x = conv_block(x, 64)
x = conv_block(x, 128)
x = conv_block(x, 256)

x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
outputs = Dense(1, activation='sigmoid')(x)

model = Model(inputs, outputs)
model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks
os.makedirs('model', exist_ok=True)
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint('model/catdog_model.h5', save_best_only=True)
lr_scheduler = ReduceLROnPlateau(patience=2)

# 訓練
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=epochs,
    callbacks=[early_stop, checkpoint, lr_scheduler]
)

# 儲存 class_indices
with open('cat_dog_classifier/class_indices.json', 'w') as f:
    json.dump(train_gen.class_indices, f)
