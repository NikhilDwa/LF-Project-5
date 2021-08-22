from src.resnet50 import *

image_1 = "./data/dog.jpg"
image_2 = "./data/cat.jpg"

similarity_score = resnet_model_score(image_1, image_2)

print("The similarity score between the two input images is: ", similarity_score)
