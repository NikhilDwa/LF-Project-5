import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity


def process_image(image_input):

    input_image = Image.open(image_input)
    image_resize = input_image.resize((224, 224))
    image_array = image.img_to_array(image_resize)
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


def pca_process_image(image_one):

    pca_input_image = Image.open(image_one)
    pca_image_resize = pca_input_image.resize((224, 224))
    pca_image_array = image.img_to_array(pca_image_resize)
    pca_image_array = pca_image_array[:, :, 0]

    return pca_image_array


def cosine_similarity_calculation(embedding_one, embedding_two):

    cosine_score = cosine_similarity(embedding_one, embedding_two)
    cosine_score = cosine_score[0][0]

    return cosine_score
