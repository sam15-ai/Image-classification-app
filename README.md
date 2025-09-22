# Image-classification-app

🖼️Image Classification using CIFAR-10

This is a Streamlit web app that uses a ResNet-18 model trained on CIFAR-10 to classify uploaded images into one of the 10 CIFAR-10 classes.


->Demo

Link


->Features:

Upload an image (.jpg, .png, .jpeg)

Model predicts one of 10 CIFAR-10 classes:

airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

Displays prediction confidence

Shows probability distribution as a bar chart


->Model Details

Base model: ResNet-18 (from torchvision)

Last fully connected layer modified for 10 CIFAR-10 classes

Trained separately and weights saved as resnet18_cifar10.pth


->Acknowledgements

Pytorch - https://pytorch.org/

Streamlit - https://streamlit.io/

CIFAR-10 Dataset - https://www.cs.toronto.edu/~kriz/cifar.html
