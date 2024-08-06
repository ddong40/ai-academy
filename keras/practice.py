import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt

path = 'C:/Users/ddong40/ai_2/_data/image/me/me.jpg'

img = load_img(path, target_size=(100,100))
print(img)

# print(type(img))
# plt.imshow(img)
# plt.show()

arr = img_to_array(img)
print(arr)
print(arr.shape) #(100, 100, 3)
print(type(arr))