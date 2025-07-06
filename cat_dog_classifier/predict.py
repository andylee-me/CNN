# predict.py
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# 載入模型
model = load_model('model/catdog_model.h5')

# 預測單張圖片
img_path = 'file/sample_image.jpg'  # 請替換成實際圖片
img = image.load_img(img_path, target_size=(128, 128))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x /= 255.

pred = model.predict(x)[0][0]
label = 'dog' if pred > 0.5 else 'cat'
print(f"預測為：{label} (機率: {pred:.2f})")
