import time

from src import pca_model_score
from src import vgg16_model_score
from src import resnet_model_score

# Provide images path
image_1 = "./data/cat.jpg"
image_2 = "./data/dog.jpg"

# Calculate similarity score using VGG16 model and checking time of computations
vgg_start = time.perf_counter()
vgg16_similarity_score = vgg16_model_score(image_1, image_2)
vgg_end = time.perf_counter() - vgg_start

# Calculate similarity score using ResNet model and checking time of computations
resnet_start = time.perf_counter()
resnet_similarity_score = resnet_model_score(image_1, image_2)
resnet_end = time.perf_counter() - resnet_start

# Calculate similarity score doing PCA on image and checking time of computations
pca_start = time.perf_counter()
pca_similarity_score = pca_model_score(image_1, image_2)
pca_end = time.perf_counter() - pca_start

# Output all the similarity scores
print("The vgg16 similarity score between the two input images is: ", vgg16_similarity_score)
print("The resnet similarity score between the two input images is: ", resnet_similarity_score)
print("Using PCA, the similarity score between the two input images is: ", pca_similarity_score)

# Output all model computation time
print("\n Time taken to compute VGG16 is :", vgg_end)
print("Time taken to compute Resnet is :", resnet_end)
print("Time taken to compute PCA is :", pca_end)
