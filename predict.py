import tensorflow as tf
from tensorflow import keras
import numpy as np

img_path = "test_images/image.jpg"  # put a test image here

model = keras.models.load_model("tennis_cnn.h5")

img_size = (128, 128)

img = keras.utils.load_img(img_path, target_size=img_size)
x = keras.utils.img_to_array(img)
x = x / 255.0
x = np.expand_dims(x, axis=0)  # shape (1, 128, 128, 3)

pred = model.predict(x)[0][0]

if pred >= 0.5:
    print(f"Prediction: TENNIS BALL (score={pred:.3f})")
else:
    print(f"Prediction: NOT TENNIS BALL (score={pred:.3f})")
