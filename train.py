# TODO: Do validation on the test set

# Imports here
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models
import matplotlib as plt
import argparse

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


# TODO: Define your transforms for the training, validation, and testing sets
parser = argparse.ArgumentParser()
parser.add_argument('data_dir',type=str, help='Gets location of image directory')
parser.add_argument('-a','--arch',action='store',type=str, help='Choose network.')
parser.add_argument('-H','--hidden_units',action='store',type=int, help='Type number of hidden units.')
parser.add_argument('-l','--learning_rate',action='store',type=float, help='Assign float number of learning rate')
parser.add_argument('-e','--epochs',action='store',type=int, help='Assign number of epochs. ')
parser.add_argument('-s','--save_dir',action='store', type=str, help='Type name of file to save trained checkpint')
parser.add_argument('-g','--gpu',action='store_true',help='Use GPU if available')


learning_rate = 0.01
arch = 'vgg16'
save_dir = './'
hidden_units = 200
epochs = 30


args = parser.parse_args()


    
# Check if user has cuda on his/her computer
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# User can pick GPU or cpu
if args.gpu == 'gpu':
    device = 'cuda'
else:
    device = 'cpu'

learning_rate = args.learning_rate

hidden_units = args.hidden_units

epochs = args.epochs



train_transforms = transforms.Compose([transforms.Resize(224),
                              transforms.RandomResizedCrop(224),
                              transforms.RandomHorizontalFlip(),
                              transforms.ToTensor(),
                              transforms.Normalize([0.485, 0.456, 0.406],
                                                   [0.229, 0.224, 0.225])
                              ])

validation_transforms = transforms.Compose([transforms.Resize(224),
                              transforms.CenterCrop(224),
                              transforms.ToTensor(),
                              transforms.Normalize([0.485, 0.456, 0.406],
                                                  [0.229, 0.224, 0.225])
                                           ])

test_transforms = transforms.Compose([transforms.Resize(224),
                              transforms.CenterCrop(224),
                              transforms.ToTensor(),
                              transforms.Normalize([0.485, 0.456, 0.406],
                                                  [0.229, 0.224, 0.225])
                              ])

# TODO: Load the datasets with ImageFolder
image_dataset = datasets.ImageFolder('flowers/train', transform=train_transforms)
valadiation_datasets = datasets.ImageFolder(data_dir + '/valid', transform = validation_transforms)
test_dataset = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
data_loader = torch.utils.data.DataLoader(image_dataset, batch_size=32, shuffle=True)
validation_loader = torch.utils.data.DataLoader(valadiation_datasets, batch_size=32, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 32, shuffle = True)

import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
print(cat_to_name)


model = models.vgg16(pretrained = True)

if args.arch == 'vgg16':
    model = models.vgg16(pretrained=True)
    for paraam in model.parameters():
        param.requires_grad = False
    classifier = nn.Sequential(OrderedDict([
    ('fc', nn.Linear(25088, 500)),
    ('relu', nn.ReLU()),
    ('drp', nn.Dropout(0.4)),
    ('fc2', nn.Linear(500, 102)),
    ('output', nn.LogSoftmax(dim=1))
]))
elif args.arch == 'densenet121':
    model = models.densenet121(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    classifier = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(1024, hidden_units)),
    ('relu', nn.ReLU()),
    ('fc2', nn.Linear(hidden_units, 100)),
    ('drop', nn.Dropout(p=0.5)),
    ('output', nn.LogSoftmax(dim=1))
    
    ]))
else:
    raise Exception("Architecture not accepted")
    
# Freeze paramewters so we don't backpropagate through them.
for param in model.parameters():
    param.requires_grad = False

# TODO: Build and train your network


model.classifier = classifier

# Define our loss
critreion = nn.NLLLoss()

#Define optimizer using prameters from our classifier
optimizer = optim.Adam(model.classifier.parameters(), args.learning_rate) # fc means fully connected
model.to(device)
      
    
 # Defining validation pass
def validation(model, validLoader, criterion):
    
    model.to(device)
    valid_loss = 0
    accuracy = 0
    
    
    for images, labels in validLoader:
        
        # moving images and labels to cuda or gpu if avaliable
        images, labels = images.to(device), labels.to(device)
        
        # Pass image through model using forward for accruracy
        output = model.forward(images)
        
        # find loss
        valid_loss += criterion(output, labels).item()
        
        # Finding Probability
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return valid_loss, accuracy   
    
    
# setting vairables for training network
epochs = args.epochs
steps = 0
running_loss = 0
print_every = 5

#loop through epochs
for epoch in range(epochs):
    
    #unpacking training dataset.
    for images, labels in data_loader:
        steps+= 1
        
        # moving images and labels to cuda or gpu if avaliable
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        
        
        logps = model.forward(images)
        loss = critreion(logps, labels)
        loss.backward()
        optimizer.step()
        
        #incrementing are running loss so we can keep track of our
        # training loss as we are going through more data.
        running_loss += loss.item()
        
        if steps % print_every == 0:
            
            model.eval()
            
            # Turn off gradients to save memory and cpu or cuda power
            with torch.no_grad():
                valid_loss, accuracy = validation(model, validation_loader, critreion)
                
                
                
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Test loss: {valid_loss/len(validation_loader):.3f}.. "
                  f"Test accuracy: {accuracy/len(validation_loader):.3f}")
            running_loss = 0
            model.train()

                    
    



accurate = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        
        model.eval()
        images, labels = images.to(device), labels.to(device)
 
        output = model(images)
        _,predicted = torch.max(output.data, 1)
        total += labels.size(0)
        accurate += (predicted == labels).sum().item()



print(f"Test accuracy: %", int((100 * accurate / total)))           
    

# TODO: Save the checkpoint 

model.class_to_idx = image_dataset.class_to_idx
 
checkpoint = {
              'classifier' : model.classifier,
    
              'index': model.class_to_idx,
 
              'optimizer': optimizer.state_dict
             }
 
torch.save(checkpoint, 'my_model.pth')

# TODO: Write a function that loads a checkpoint and rebuilds the model
def loadCheckpoint(path):
    checkpoint = torch.load(path)
 
    #Parameters are freezed out.
 
    model = models.vgg16(pretrained=True)
 
    model.classifier = classifier
 
    model.idx_to_class = checkpoint['index']
 
 
    model.optimizer = checkpoint['optimizer']

    

    return model

model_check = loadCheckpoint('my_model.pth')

model_check


from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    image_on = Image.open(image_path)
    image_on.thumbnail((255,255))
    
    # getting dimensions of image
    width, height = image_on.size
    
    # getting side lengths of each image
    left_side = (width - 224)/2
    right_side = left_side + 224
    top_side = (height-224)/2
    bottom_side = top_side + 224
    
    #cropping image
    image_on = image_on.crop((left_side, top_side, right_side, bottom_side))
    
    
    #normalizing image
    np_image = np.array(image_on)
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    np_image = (np_image-mean)/std
    
    np_image = np.array(image_on)/254
    
    # transposing to match dimensions
    np_image = np_image.transpose((2, 0, 1))

    
    # TODO: Process a PIL image for use in a PyTorch model
    return np_image
    

