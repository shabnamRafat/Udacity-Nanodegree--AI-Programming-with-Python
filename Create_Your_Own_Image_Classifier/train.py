# Imports here
import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import OrderedDict
import json
from PIL import Image
import numpy as np
import os
import argparse

print("Please select your preferred architecture from below options:")
print("1. vgg16")
print("2. densenet 16")

#architecture = input("Please select your preferred architecture from these(vgg16 or densenet121) : ")

architecture = "densenet121"

if architecture is "vgg16":
    model = models.vgg13(pretrained=True)
elif architecture is "densenet121":
    model = models.densenet121(pretrained=True)

print(model)




data_dir = 'flowers'

train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# TODO: Define your transforms for the training, validation, and testing sets
training_data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

# no need to perform randomization on validation/test samples; only need to normalize
validation_data_transforms = transforms.Compose([transforms.Resize(255),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

testing_data_transforms  = transforms.Compose([transforms.Resize(255),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

# TODO: Load the datasets with ImageFolder
training_datasets = datasets.ImageFolder(train_dir, transform = training_data_transforms)
validation_datasets = datasets.ImageFolder(valid_dir, transform = validation_data_transforms)
testing_datasets  = datasets.ImageFolder(test_dir,  transform = testing_data_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
training_dataloader = torch.utils.data.DataLoader(training_datasets, batch_size = 64, shuffle = True)
validation_dataloader = torch.utils.data.DataLoader(validation_datasets, batch_size = 64, shuffle = True)
testing_dataloader  = torch.utils.data.DataLoader(testing_datasets,  batch_size = 64, shuffle = True)


with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

for param in model.parameters():
    param.requires_grad = False

model.classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(1024, 512)),
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(0.2)),
                          ('fc2', nn.Linear(512, 256)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)

model.to(device);

from time import time
epochs = 1
steps  = 0
running_loss = 0
print_every  = 5
for epoch in range(epochs):
    for inputs, labels in training_dataloader:
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
                
        optimizer.zero_grad()
        
        logps = model.forward(inputs)
        loss  = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in validation_dataloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    
                    test_loss += batch_loss.item()
                    
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            print(f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Validation loss: {test_loss/len(validation_dataloader):.3f}.. "
                  f"Validation accuracy: {accuracy/len(validation_dataloader):.3f}")
            running_loss = 0
            model.train()
            
            
model.to(device);

accuracy = 0
model.eval()

with torch.no_grad():
    
    for inputs, labels in testing_dataloader:

        inputs, labels = inputs.to(device), labels.to(device)
        logps = model.forward(inputs)

        # Calculate accuracy
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

print(f"Test accuracy: {accuracy/len(testing_dataloader):.3f}")
        
model.train();

model.class_to_idx = training_datasets.class_to_idx
torch.save(model.state_dict(), 'checkpoint.pth')

print("Saved the checkpoint successfully")








