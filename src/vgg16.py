import numpy as np
from PIL import Image
from keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions

model = VGG16(weights='imagenet')


def vgg16_embedding(image_input):

    input_image = Image.open(image_input)
    image_resize = input_image.resize((224, 224))
    image_array = image.img_to_array(image_resize)
    image_array = np.expand_dims(image_array, axis=0)
    image_preprocessed = preprocess_input(image_array)
    embedding = model.predict(image_preprocessed)

    return embedding


def vgg16_model_score(image_one, image_two):

    image_one_embedding = vgg16_embedding(image_one)
    image_two_embedding = vgg16_embedding(image_two)
    cosine_score = cosine_similarity(image_one_embedding, image_two_embedding)

    return cosine_score
