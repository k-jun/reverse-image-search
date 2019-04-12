import os
import numpy as np

from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, VGG16
from keras.models import Model

from annoy import AnnoyIndex

ANNOY_MODEL_PATH = "./models/ramen.ann"
ANNOY_DIMENTION = 4096
SEARCH_IMAGE_PATH = "ramen_images/ramen1001.jpg"

# VGG19から中間層を抽出
base_model = VGG16(weights="imagenet")
model = Model(inputs=base_model.input, outputs=base_model.get_layer("fc2").output)


# Annoyのモデルを構築
loaded_model = AnnoyIndex(ANNOY_DIMENTION)
loaded_model.load(ANNOY_MODEL_PATH)

# 検索対象の画像をロードして、ベクトルに変換
img_path = SEARCH_IMAGE_PATH
img = image.load_img(img_path, target_size=(224, 224))

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

fc2_features = model.predict(x)

# Annoyで検索
items = loaded_model.get_nns_by_vector(fc2_features[0], 3, search_k=-1, include_distances=False)
print(items)

