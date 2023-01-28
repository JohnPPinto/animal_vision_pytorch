import gradio as gr
import os
import torch

from model import create_effnet_model
from timeit import default_timer as timer
from typing import Tuple, Dict

# setup class names
with open('class_names.txt', 'r') as f:
    class_names = [i.strip() for i in f.readlines()]

# Create model
effnetb2, effnetb2_transforms = create_effnet_model(num_classes=len(class_names))

# Load saved weights
effnetb2.load_state_dict(torch.load(f='pretrained_effnetb2_animal_vision_big.pth',
                                    map_location=torch.device('cpu')))

# Create Predict function
def predict(img) -> Tuple[Dict, float]:
    """
    Transforms the Input Image and returns prediction and time taken to predict.
    """
    # Start the timer
    start_time = timer()
    
    # Transform the image and add a batch
    img = effnetb2_transforms(img).unsqueeze(dim=0)
    
    # Put the model in evaluation and inference mode
    effnetb2.eval()
    with torch.inference_mode():
        # Get prediction probabilities
        pred_probs = torch.softmax(effnetb2(img), dim=1)
    
    # Create a prediction label and probability dictionary for each prediction class as per the gradio requirement
    pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}
    
    # Calculate the prediction time
    pred_time = round(timer() - start_time, 5)
    
    return pred_labels_and_probs, pred_time

# Create gradio app
# Title, description and article in gradio app
title = 'Animal Vision Big 🦁 🐯 🐺 👀'
description = 'A Animal Vision Big project created and build using EfficientNetB2 feature extractor model to clasify images of 30 different animals #Github link of class_names.txt#.'
article = 'Created by John Pinto.'

# Create example list
example_list = [['examples/' + example] for example in os.listdir('examples') if example.endswith('.jpg')]

# Creating a gradio demo
demo = gr.Interface(fn=predict,
                    inputs=gr.Image(type='pil'),
                    outputs=[gr.Label(num_top_classes=5, label='Predictions'),
                             gr.Number(label='Prediction Time (seconds)')],
                    examples=example_list,
                    title=title,
                    description=description,
                    article=article)
# Launch the app
demo.launch()
