from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

from .main import process_image, cosine_similarity_calculation

model = ResNet50(weights='imagenet')


def resnet_embedding(image_input):

    image_array = process_image(image_input)
    image_preprocessed = preprocess_input(image_array)
    embedding = model.predict(image_preprocessed)

    return embedding


def resnet_model_score(image_one, image_two):

    image_one_embedding = resnet_embedding(image_one)
    image_two_embedding = resnet_embedding(image_two)
    cosine_score = cosine_similarity_calculation(image_one_embedding, image_two_embedding)

    return cosine_score
