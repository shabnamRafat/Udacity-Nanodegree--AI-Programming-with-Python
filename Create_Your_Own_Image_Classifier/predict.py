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



data_dir = 'flowers'

train_dir = data_dir + '/train'

training_data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

training_datasets = datasets.ImageFolder(train_dir, transform = training_data_transforms)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

architecture = "densenet121"

if architecture is "vgg16":
    model = models.vgg13(pretrained=True)
elif architecture is "densenet121":
    model = models.densenet121(pretrained=True)
    
    
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    
    if checkpoint['arch'] == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif checkpoint['arch'] == 'vgg13':
        model = models.vgg13(pretrained = True)
    else:
        raise ValueError('Model arch error.')

    for param in model.parameters():
        param.requires_grad = False
        
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    
    return model



state_dict = torch.load('checkpoint.pth')
model.to(device)



def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    img = Image.open(image)
    convert_tensor = transforms.Compose([transforms.Resize(255),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
    img = np.array(convert_tensor(img))
    return img



def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = np.array(image).transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

test_image_processing_function = process_image('flowers/test/10/image_07090.jpg')



def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    model.class_to_idx = training_datasets.class_to_idx
    model.to(device)
    image = process_image(image_path)
    image = torch.from_numpy(image).type(torch.FloatTensor).to(device).unsqueeze(0)
  
    with torch.no_grad ():
        output = model.forward(image)
        
    probs, item_list = torch.exp(output).topk(topk)
    probs = F.softmax(output.data,dim=1)
    item_list = item_list.to(device)
    item_list = np.array(item_list).tolist()[0]
    
    
    mapping = {val: key for key, val in model.class_to_idx.items()}
    classes = []
    for item in range(0, len(item_list)):
        get_class = mapping[item]
        classes.append(get_class)
    
    
    return probs, classes

image_path = ('flowers/test/10/image_07090.jpg')
probs, classes = predict(image_path, model, 5)


names = [cat_to_name[key] for key in classes]

print("Probabilities: ", probs)
print("Classes : ", classes)
print("Names: ", names)

image_to_display = imshow(test_image_processing_function)
plt.show()