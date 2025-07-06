import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# 參數與路徑
val_dir = 'file/kaggle_cats_vs_dogs_f/val'
train_dir = 'file/kaggle_cats_vs_dogs_f/train'
img_size = (128, 128)
batch_size = 32

# 載入模型
model = tf.keras.models.load_model('model/catdog_model.h5')

# 重新編譯（必要）
model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(learning_rate=0.001),
    metrics=['accuracy']
)

# 建立資料產生器
val_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)
train_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

# 印出資料分佈
print("Class indices:", train_gen.class_indices)
print("Train class distribution:", np.bincount(train_gen.classes))
print("Val labels distribution:", np.bincount(val_gen.classes))

# 預測
pred_probs = model.predict(val_gen)
pred_labels = (pred_probs > 0.5).astype(int).flatten()
true_labels = val_gen.classes

# 顯示預測機率分布
print("Prediction probabilities (first 10):", pred_probs[:10].flatten())
print("Predicted labels distribution:", np.bincount(pred_labels))

# 混淆矩陣與報表
cm = confusion_matrix(true_labels, pred_labels)
labels = list(val_gen.class_indices.keys())

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix on Validation Set")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
os.makedirs('file/model', exist_ok=True)
plt.savefig('file/model/confusion_matrix.png')

print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(true_labels, pred_labels, target_names=labels))

# 評估準確率
val_loss, val_acc = model.evaluate(val_gen)
print(f'驗證集：val_loss = {val_loss:.4f}, val_accuracy = {val_acc:.4f}')

train_loss, train_acc = model.evaluate(train_gen)
print(f'訓練集：train_loss = {train_loss:.4f}, train_accuracy = {train_acc:.4f}')

# 顯示模型摘要
model.summary()
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# 參數與路徑
val_dir = 'file/kaggle_cats_vs_dogs_f/val'
train_dir = 'file/kaggle_cats_vs_dogs_f/train'
img_size = (128, 128)
batch_size = 32

# 載入模型
model = tf.keras.models.load_model('model/catdog_model.h5')

# 重新編譯（必要）
model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(learning_rate=0.001),
    metrics=['accuracy']
)

# 建立資料產生器
val_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)
train_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

# 印出資料分佈
print("Class indices:", train_gen.class_indices)
print("Train class distribution:", np.bincount(train_gen.classes))
print("Val labels distribution:", np.bincount(val_gen.classes))

# 預測
pred_probs = model.predict(val_gen)
pred_labels = (pred_probs > 0.5).astype(int).flatten()
true_labels = val_gen.classes

# 顯示預測機率分布
print("Prediction probabilities (first 10):", pred_probs[:10].flatten())
print("Predicted labels distribution:", np.bincount(pred_labels))

# 混淆矩陣與報表
cm = confusion_matrix(true_labels, pred_labels)
labels = list(val_gen.class_indices.keys())

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix on Validation Set")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
os.makedirs('file/model', exist_ok=True)
plt.savefig('file/model/confusion_matrix.png')

print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(true_labels, pred_labels, target_names=labels))

# 評估準確率
val_loss, val_acc = model.evaluate(val_gen)
print(f'驗證集：val_loss = {val_loss:.4f}, val_accuracy = {val_acc:.4f}')

train_loss, train_acc = model.evaluate(train_gen)
print(f'訓練集：train_loss = {train_loss:.4f}, train_accuracy = {train_acc:.4f}')

# 顯示模型摘要
model.summary()
