from src import resnet_model_score
from src import vgg16_model_score

image_1 = "./data/dog.jpg"
image_2 = "./data/cat.jpg"

vgg16_similarity_score = vgg16_model_score(image_1, image_2)
resnet_similarity_score = resnet_model_score(image_1, image_2)

print("The vgg16 similarity score between the two input images is: ", vgg16_similarity_score)
print("The resnet similarity score between the two input images is: ", resnet_similarity_score)

