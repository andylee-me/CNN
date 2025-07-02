import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 圖片路徑與參數
val_dir = 'file/kaggle_cats_vs_dogs_f/val'
train_dir = 'file/kaggle_cats_vs_dogs_f/train'
img_size = (128, 128)
batch_size = 32

# 載入模型
model = tf.keras.models.load_model('model/catdog_model.h5')

# 建立驗證資料產生器
val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'
)
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'
)

# 評估模型
loss, acc = model.evaluate(val_generator)
print(f'驗證結果：val_loss = {loss:.4f}, val_accuracy = {acc:.4f}')


loss, acc = model.evaluate(train_generator)
print(f'驗證結果：train_loss = {loss:.4f}, train_accuracy = {acc:.4f}')
