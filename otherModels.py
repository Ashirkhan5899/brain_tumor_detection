import torchvision.models as models
import torch.nn as nn

num_classes = 4

def vgg_model():
    vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    for param in vgg16.parameters():
        param.requires_grad = False
    # Add custom classification layers
    vgg16.classifier[6] = nn.Sequential(
    nn.Linear(vgg16.classifier[6].in_features, 128),
    nn.ReLU(),
    nn.Linear(128, num_classes),
    nn.Softmax(dim=1)
    )
    return vgg16

def resNet_model():
    # Load pre-trained ResNet50
    resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    for param in resnet50.parameters():
        param.requires_grad = False
    # Add custom classification layers
    resnet50.fc = nn.Sequential(
        nn.Linear(resnet50.fc.in_features, 128),
        nn.ReLU(),
        nn.Linear(128, num_classes),
        nn.Softmax(dim=1)
    )
    return resnet50

def mobileNet_model():
    # Load pre-trained mobilenet_v3_small
    #mobilenet_v3_small = models.mobilenet_v3_small(pretrained=True)
    mobilenet_v3 = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
    for param in mobilenet_v3.parameters():
        param.requires_grad = False

    # Add custom classification layers
    mobilenet_v3.classifier[3] = nn.Sequential(
        nn.Linear(mobilenet_v3.classifier[3].in_features, 128),
        nn.ReLU(),
        nn.Linear(128, num_classes),
        nn.Softmax(dim=1)
    )
    return mobilenet_v3