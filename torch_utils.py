import torch
import torch.nn as nn
import torchvision
from torchvision import datasets,transforms,models
from torchvision.transforms.transforms import ToPILImage
from PIL import Image
import numpy as np
import torch.optim as optim
import cv2
import io

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)


def Transform_image(img_path):
    img_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean,std)
    ])
    img = Image.open(img_path).convert("RGB")
    return img_transform(img).unsqueeze(0)

def Net_Effb0():
    model = models.efficientnet_b0(pretrained=False)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier[1] = nn.Linear(1280,4)
    return model

class Net_CNN(nn.Module):
    def __init__(self,num_classes):
        super(Net_CNN,self).__init__()
    
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
 
            nn.Conv2d(32, 64, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
 
            nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

 
            nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
 
            nn.Conv2d(128, 128, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
 
            nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
 
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)  
        )
    def forward(self,xb):
        return self.network(xb)

def Prediction(img_path,model):
    img_tensor = Transform_image(img_path)
    output = model(img_tensor)
    output = nn.functional.softmax(output[0],dim=0)
    confidence,index = torch.max(output,0)

    return index.item()
