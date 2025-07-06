import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# 圖片路徑與參數
val_dir = 'file/kaggle_cats_vs_dogs_f/val'
train_dir = 'file/kaggle_cats_vs_dogs_f/train'
img_size = (128, 128)
batch_size = 32

# 載入模型
model = tf.keras.models.load_model('model/catdog_model.h5')

# 建立資料產生器
val_datagen = ImageDataGenerator(rescale=1./255)
train_datagen = ImageDataGenerator(rescale=1./255)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False  # 混淆矩陣需要保持順序
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

# 進行預測
pred_probs = model.predict(val_generator)
pred_labels = (pred_probs > 0.5).astype(int).flatten()
true_labels = val_generator.classes

# 混淆矩陣
cm = confusion_matrix(true_labels, pred_labels)
labels = list(val_generator.class_indices.keys())

# 畫圖
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix on Validation Set")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
os.makedirs('file/model', exist_ok=True)
plt.savefig('file/model/confusion_matrix.png')

# 評估模型
val_loss, val_acc = model.evaluate(val_generator)
print(f'驗證結果：val_loss = {val_loss:.4f}, val_accuracy = {val_acc:.4f}')

train_loss, train_acc = model.evaluate(train_generator)
print(f'訓練結果：train_loss = {train_loss:.4f}, train_accuracy = {train_acc:.4f}')

# 類別與樣本資訊
print("類別對應字典：", train_generator.class_indices)
print("前10個訓練標籤：", train_generator.classes[:10])
print("訓練樣本數：", train_generator.samples, "驗證樣本數：", val_generator.samples)

# 顯示模型摘要
model.summary()
