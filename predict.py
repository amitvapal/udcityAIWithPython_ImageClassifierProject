# Imports here
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models
import matplotlib as plt
import argparse

# Set up parameters for entry in command line
parser = argparse.ArgumentParser()

parser.add_argument('-g','--gpu',action='store_true',help='Use GPU if available')
parser.add_argument('-in','--input_image', type=str, help='Image path')
parser.add_argument('-ck','--checkpoint', type=str, help='Gets checkpoint of trained model')
parser.add_argument('-k','--top_k', type=int, help='Returns most likely classes')
parser.add_argument('-c','--classes', type=str, help='Compare categories to clases from json file.')
parser.add_argument('--cat_to_name', action='store',help='Input image path.')


    
args = parser.parse_args()


cat_to_name = 'cat_to_name.json'
if args.cat_name_dir:
    cat_to_name = args.cat_name_dir
    
with open(cat_names, 'r') as f:
    cat_to_name = json.load(f)

    
image_path = args.input_image
checkpoint = args.checkpoint

top_k = 1
if args.top_k:
    top_k = args.top_k
    classes = None
    
if args.classes:
    classes = args.classes
    cuda = False

# Check if user has cuda on his/her computer
device = 'cuda' if args.gpu else 'cpu'


def predict(image_path, model, topk=5, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    
  
    model.to(device)
    # in inference mode
    model.eval
     
    with torch.no_grad():   
        # Get image from image_path
        image = process_image(image_path)

        # Creating tensor for image to match dimensions
        image = torch.from_numpy(np.array([image])).float()
        image.to(device)

        # Create prediction
        logps = model(image)
        ps = torch.exp(logps)

        # Get the categories of flowers stored as list
        prob, classes = ps.topk(topk, dim=1)
        model.train()
        
        # Get the top five categories
        top_p = prob.tolist()[0]
        top_classes = classes.tolist()[0]
        
        # Reeverse categories from dictionary
        idx_to_class = {v:k for k, v in model.idx_to_class.items()}
        
        
        labels = []
        for c in top_classes:
            labels.append(cat_to_name[idx_to_class[c]])
    
        return top_p, labels
    
probability, classes = predict('flowers/test/1/image_06743.jpg', model, device=device)
print(probability)
print(classes)



# TODO: Display an image along with the top 5 classes

def check_sanity():
    fig = plt.figure(figsize = [10,5])
    # Axes for flower image 
    ax = fig.add_axes([.5, .4, .5, .5])

    # Displaay image
    result = process_image('flowers/test/1/image_06743.jpg')
    ax = imshow(result, ax)
    ax.axis('off')
    index = 77
    ax.set_title(cat_to_name[str(index)])

    # Prediction of image
    predictions, classes = predict('flowers/test/1/image_06743.jpg', model, device = device)


    # Create bar graph
    # Axis x and y
    ax1 = fig.add_axes([0, -.4, .888, .888])

    # Classes probability
    y_pos = np.arange(len(classes))

    # Horizontal bar chart to see it better
    plt.barh(y_pos, predictions, align='center', alpha=0.5)
    plt.yticks(y_pos, classes)
    plt.xlabel('probabilities')
    plt.show()
    
check_sanity()


