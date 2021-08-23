from keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

from .main import process_image, cosine_similarity_calculation

model = VGG16(weights='imagenet')


def vgg16_embedding(image_input):
    """
        Takes image array and compute embedding using VGG16 model.

        Returns
        ------------------------
        [array]
            [returns an image embedding.]
    """

    image_array = process_image(image_input)
    image_preprocessed = preprocess_input(image_array)
    embedding = model.predict(image_preprocessed)

    return embedding


def vgg16_model_score(image_one, image_two):
    """
        Takes two images, preprocess the image and find similarity score.

        Returns
        ------------------------
        [float]
            [returns similarity score between the image.]
    """
    image_one_embedding = vgg16_embedding(image_one)
    image_two_embedding = vgg16_embedding(image_two)
    cosine_score = cosine_similarity_calculation(image_one_embedding, image_two_embedding)

    return cosine_score
