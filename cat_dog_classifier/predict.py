import json
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 資料路徑
val_dir = 'file/kaggle_cats_vs_dogs_f/val'
model_path = 'model/catdog_model.h5'
class_indices_path = 'file/cat_dog_classifier/class_indices.json'

# 載入模型與 class label 對應
model = tf.keras.models.load_model(model_path)
with open(class_indices_path, 'r') as f:
    class_indices = json.load(f)

# 建立反向對應
idx_to_class = {v: k for k, v in class_indices.items()}

# 預處理
datagen = ImageDataGenerator(rescale=1./255)
val_gen = datagen.flow_from_directory(
    val_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# 預測
y_true = val_gen.classes
y_pred = model.predict(val_gen)
y_pred_label = (y_pred > 0.5).astype(int).flatten()

# 混淆矩陣
cm = confusion_matrix(y_true, y_pred_label)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=idx_to_class.values(), yticklabels=idx_to_class.values(), cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('file/cat_dog_classifier/confusion_matrix.png')
plt.close()

# 分類報告
report = classification_report(y_true, y_pred_label, target_names=idx_to_class.values())
print(report)

with open('file/cat_dog_classifier/classification_report.txt', 'w') as f:
    f.write(report)
