import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

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
    shuffle=False  # 混淆矩陣需要順序對應
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

# 模型預測
pred_probs = model.predict(val_generator)
pred_labels = (pred_probs > 0.5).astype(int).flatten()
true_labels = val_generator.classes

# 混淆矩陣
cm = confusion_matrix(true_labels, pred_labels)
labels = list(val_generator.class_indices.keys())

# 繪製混淆矩陣
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix on Validation Set")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()

# 儲存圖檔
os.makedirs('file/model', exist_ok=True)
plt.savefig('file/model/confusion_matrix.png')

# 評估模型
val_loss, val_acc = model.evaluate(val_generator, verbose=0)
train_loss, train_acc = model.evaluate(train_generator, verbose=0)

(val_loss, val_acc, train_loss, train_acc, labels[:2], true_labels[:10], train_generator.samples, val_generator.samples)




import numpy as np

# 印出類別索引與類別分布
print("class indices:", val_generator.class_indices)
print("val labels distribution:", np.bincount(val_generator.classes))

# 預測與混淆矩陣
pred_probs = model.predict(val_generator)
pred_labels = (pred_probs > 0.5).astype(int).flatten()
true_labels = val_generator.classes

from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(true_labels, pred_labels)
print("Confusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(true_labels, pred_labels, target_names=list(val_generator.class_indices.keys())))
