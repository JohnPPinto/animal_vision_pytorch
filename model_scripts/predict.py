"""
Contains code to predict a single image class using a saved model.
"""
import os
import torch
import torchvision
import argparse
import model_builder

# creating a parser
parser = argparse.ArgumentParser()

# get an image_path
parser.add_argument('--image_path',
                    help='target image filepath to predict on')
# get a saved model
parser.add_argument("--model_path",
                    default='models/model.pth',
                    type=str,
                    help='target model to use for prediction filepath')
args = parser.parse_args()

# setup classs names
class_names = sorted([i for i in os.listdir('data/lion_tiger_wolf/train')])

# Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"

# get image path
IMG_PATH = args.image_path
print(f'[INFO] Predicting on image: {IMG_PATH}')

# Function to load the model
def load_model(filepath=args.model_path):
    # Need to use hyperparameters for saved model
    load_model = model_builder.TinyVGG(input_shape=3,
                                       output_shape=len(class_names),
                                       hidden_units=128).to(device)
    print(f'[INFO] Loading in model from: {filepath}')
    load_model.load_state_dict(torch.load(filepath))
    return load_model

def predict_on_image(image_path=IMG_PATH,
                     filepath=args.model_path):
    model=load_model(filepath)
    image = torchvision.io.read_image(str(image_path)).type(torch.float32)
    image = image/255.
    transform = torchvision.transforms.Resize(size=(64, 64))
    image = transform(image)
  
    model.eval()
    with torch.inference_mode():
        image=image.to(device)
        pred = model(image.unsqueeze(dim=0))
        pred_probs = torch.softmax(pred, dim=1)
        pred_label = torch.argmax(pred_probs, dim=1)
        pred_label_class = class_names[pred_label]
    print(f'[INFO] Pred class: {pred_label_class} and Pred Prob: {pred_probs.max():.3f}')

if __name__ == '__main__':
    predict_on_image()
