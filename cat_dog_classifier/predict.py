import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 載入模型與類別對應表
model = load_model('model/catdog_model.h5')
with open('model/class_indices.json') as f:
    class_indices = json.load(f)
class_labels = {v: k for k, v in class_indices.items()}

# 圖像參數
img_size = (128, 128)
batch_size = 32

# 準備資料
val_dir = 'file/kaggle_cats_vs_dogs_f/val'
datagen = ImageDataGenerator(rescale=1./255)
val_gen = datagen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

# 預測
pred_probs = model.predict(val_gen)
pred_classes = (pred_probs > 0.5).astype('int32').flatten()
true_classes = val_gen.classes

# 混淆矩陣
cm = confusion_matrix(true_classes, pred_classes)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_labels.values(), yticklabels=class_labels.values(), cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('confusion_matrix.png')
plt.close()

# 顯示分類報告
report = classification_report(true_classes, pred_classes, target_names=class_labels.values())
print(report)
