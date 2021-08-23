# Image Similarity

Image Similarity is a program to check the similarity score between two different using different types of models. 

## How to use?

Run the app.py file and it will run the program.

```bash
python run app.py
```

## Usage

We provided two different image of cat and dog. The similarity scores were calculated on VGG16, ResNet50 and PCA model.

### Analyzing the result
```bash
The vgg16 similarity score between the two input images is:  0.00045263517
The resnet similarity score between the two input images is:  8.47548e-05
Using PCA, the similarity score between the two input images is:  -0.9369974
``` 

Also, the computation time of all the model were stamped. And we find out the following result.

```bash 
Time taken to compute VGG16 is : 1.6684953
Time taken to compute Resnet is : 1.175218899999999
Time taken to compute PCA is : 0.30439409999999967
```
Looking at the time computation, although PCA has reduce the features but 99% of image was recovered, the time computation is faster doing PCA than any other model.  

## Call for Contributions
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## Authors
[Nikhil] & [Roman]
