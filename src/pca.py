import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA

from .main import pca_process_image


def pca_features(image_array):

    pca_100 = PCA(n_components=100)
    pca_reduced = pca_100.fit_transform(image_array)
    pca_recovered = pca_100.inverse_transform(pca_reduced)

    return pca_recovered


def pca_model_score(image_one, image_two):

    pca_image_one_array = pca_process_image(image_one)
    pca_image_two_array = pca_process_image(image_two)

    image_one_pca_100 = pca_features(pca_image_one_array)
    image_two_pca_100 = pca_features(pca_image_two_array)

    cosine_loss = tf.keras.losses.CosineSimilarity(axis=1)
    similarity_score = cosine_loss(image_one_pca_100, image_two_pca_100).numpy()

    return similarity_score
