import torch
import torchvision
from PIL import Image
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity

model = torchvision.models.resnet50(pretrained=True)


def resnet_embedding(image_input):

    input_image = Image.open(image_input)
    preprocessing = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocessing(input_image)
    input_batch = input_tensor.unsqueeze(0)
    with torch.no_grad():
        output = model(input_batch)
    value = torch.nn.functional.softmax(output[0], dim=0)

    return value


def resnet_model_score(image_one, image_two):

    embedding_one = resnet_embedding(image_one).reshape(1, -1)
    embedding_two = resnet_embedding(image_two).reshape(1, -1)
    cosine_scores = cosine_similarity(embedding_one, embedding_two)[0][0]
    # print(cosine_scores[0][0] * 100)
    return cosine_scores
