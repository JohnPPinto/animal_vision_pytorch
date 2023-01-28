import torch
import torchvision
from torch import nn

def create_effnetb2_model(num_classes: int=3,
                          seed: int=42):
    """
    Create a feature extraction efficientNetB2 model and transforms.
    Args: 
          num_classes: A integer number for classes in the classifier head.
          seed: A random seed.
    Returns:
          model: A model of EfficientNetB2.
          transforms: A torchvision image transforms based out of the EfficientNetb2.
    """
    # Setup pretrained weights
    weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
    
    # Get EfficientNetB2 transforms
    transforms = weights.transforms()
    
    # Setup the EfficientNetB2 model
    model = torchvision.models.efficientnet_b2(weights=weights)
    
    # Freeze the parameters of the model
    for params in model.parameters():
        params.requires_grad = False
    
    # Change the head classifier with a random seed
    torch.manual_seed(seed)
    model.classifier[1] = nn.Linear(in_features=1408, out_features=num_classes)
    
    return model, transforms
