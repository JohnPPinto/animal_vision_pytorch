import gradio as gr
import os
import torch
from model import create_effnetb2_model
from timeit import default_timer as timer
from typing import Tuple, Dict

# Defining class names
class_names = ['LION', 'TIGER', 'WOLF']

# Model and transform preparation using create_effnetb2_model function
effnetb2, effnetb2_transforms = create_effnetb2_model(num_classes=len(class_names))

# Load saved weights from the saved model
effnetb2.load_state_dict(torch.load(f='pretrained_effnetb2_animal_vision_mini.pth',
                                   map_location=torch.device('cpu')))

# create a predict function to connect inputs and outputs
def predict(img) -> Tuple[Dict, float]:
    """
    Transforms the Input Image and returns prediction and time taken to predict.
    """
    # Start the timer
    start_time = timer()
    
    # Transform and batch the image
    img = effnetb2_transforms(img).unsqueeze(0)
    
    # Put model in evaluation and inference mode for prediction
    effnetb2.eval()
    with torch.inference_mode():
        # get the prediction probabilities
        pred_prob = torch.softmax(effnetb2(img), dim=1)
    
    # Creating a dictionary with prediction probabilities and labels as per the gradio format
    pred_labels_and_probs = {class_names[i]: float(pred_prob[0][i]) for i in range(len(class_names))}
    
    # Calculate the prediction time
    pred_time = round(timer() - start_time, 5)
    
    return pred_labels_and_probs, pred_time

# Building the gradio interface
# creating the title, description and article
title = 'Animal Vision Mini Demo 🦁 🐯 🐺'
description = 'A Animal Vision Mini project created and build using EfficientNetB2 feature extractor model to clasify images of Lion, Tiger, and Wolf.'
article = 'Created by John Pinto.'

# Creating the example list
example_list = [['examples/' + example] for example in os.listdir('examples') if example.endswith('.jpg')]

# Creating the gradio demo
demo = gr.Interface(fn=predict,
                    inputs=gr.Image(type='pil'),
                    outputs=[gr.Label(num_top_classes=3, label='Predictions'),
                             gr.Number(label='Prediction Time (s)')],
                    examples=example_list,
                    title=title,
                    description=description,
                    article=article)

# Launch the demo app
demo.launch()
