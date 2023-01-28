import torch
import torchvision
from torch import nn

def create_effnet_model(num_classes: int=30,
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
    # Create weights, transform and model of effnetb2
    weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
    transform = weights.transforms()
    model = torchvision.models.efficientnet_b2(weights=weights)
    
    # Freeze all the layers of the model
    for param in model.parameters():
        param.requires_grad = False
    
    # Change the head of the model
    torch.manual_seed(seed)
    model.classifier[1] = nn.Linear(in_features=1408, out_features=num_classes)
    
    return model, transform
