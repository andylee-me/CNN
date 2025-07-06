import os
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# 資料路徑
train_dir = 'file/kaggle_cats_vs_dogs_f/train'
val_dir = 'file/kaggle_cats_vs_dogs_f/val'
img_size = (128, 128)
batch_size = 32

# 資料增強
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
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

# 儲存 class_indices
os.makedirs('file/cat_dog_classifier', exist_ok=True)
with open('file/cat_dog_classifier/class_indices.json', 'w') as f:
    json.dump(train_gen.class_indices, f)

# 模仿 MobileNet 的強化結構
def depthwise_block(x, filters, strides):
    x = DepthwiseConv2D(kernel_size=3, strides=strides, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters, kernel_size=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x

inputs = Input(shape=(128, 128, 3))
x = Conv2D(32, 3, strides=2, padding='same')(inputs)
x = BatchNormalization()(x)
x = ReLU()(x)

x = depthwise_block(x, 64, 1)
x = depthwise_block(x, 128, 2)
x = depthwise_block(x, 128, 1)
x = depthwise_block(x, 256, 2)
x = depthwise_block(x, 256, 1)
x = depthwise_block(x, 512, 2)

x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
outputs = Dense(1, activation='sigmoid')(x)

model = Model(inputs, outputs)

model.compile(
    optimizer=Adam(1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

early_stop = EarlyStopping(patience=5, monitor='val_loss', restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(patience=2)
checkpoint = ModelCheckpoint('model/catdog_model.h5', save_best_only=True)

os.makedirs('model', exist_ok=True)

history = model.fit(
    train_gen,
    epochs=20,
    validation_data=val_gen,
    callbacks=[early_stop, reduce_lr, checkpoint]
)
'''

os.makedirs("file/cat_dog_classifier", exist_ok=True)
with open("file/cat_dog_classifier/train_cnn.py", "w", encoding="utf-8") as f:
    f.write(train_cnn_strong)
