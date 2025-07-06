import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# 載入訓練好的模型
model = tf.keras.models.load_model('model/catdog_mobilenet_like.h5')

# 圖片參數
img_size = (128, 128)

# 資料夾路徑
val_dir = 'file/kaggle_cats_vs_dogs_f/val'

# 收集所有圖片路徑
image_paths = []
labels = []

for label in ['cat', 'dog']:
    folder = os.path.join(val_dir, label)
    for fname in os.listdir(folder):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_paths.append(os.path.join(folder, fname))
            labels.append(0 if label == 'cat' else 1)

# 預測
correct = 0
for path, true_label in zip(image_paths, labels):
    img = image.load_img(path, target_size=img_size)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)[0][0]
    pred_label = 1 if pred > 0.5 else 0

    if pred_label == true_label:
        correct += 1

accuracy = correct / len(labels)
print(f"✅ 總共 {len(labels)} 張圖片，準確率為：{accuracy:.4f}")
