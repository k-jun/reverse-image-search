import numpy as np

from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, VGG16
from keras.models import Model

# from google.colab import drive
# drive.mount('./gdrive')

# !pip install annoy
from annoy import AnnoyIndex

IMAGE_BASE_PATH  = "./ramen_images/"
ANNOY_MODEL_PATH = "./models/ramen.ann"
ANNOY_DIMENTION  = 4096

# VGG19から中間層を抽出
base_model = VGG16(weights="imagenet")
model = Model(inputs=base_model.input, outputs=base_model.get_layer("fc2").output)


# Annoyのモデルを構築
annoy_model = AnnoyIndex(ANNOY_DIMENTION)

# 画像をベクトルに変換してAnnoyに登録
for i in range (1, 1001):
    img_path = IMAGE_BASE_PATH + "ramen" +str(i)+ ".jpg"
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    fc2_features = model.predict(x)

    annoy_model.add_item(i, fc2_features[0])
    print(img_path, "Done!")

annoy_model.build(1000)
annoy_model.save(ANNOY_MODEL_PATH)
