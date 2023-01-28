"""
Contains PyTorch Model code to 
initiate a custom model to train and test a dataset 
"""
import torch
from torch import nn

class TinyVGG(nn.Module):
    """
    Creates a neural network architecture similar to the TinyVGG architecture from
    here: https://github.com/poloclub/cnn-explainer/tree/master/tiny-vgg and for visuals 
    here: https://poloclub.github.io/cnn-explainer/
  
    Args:
      input_shape: An integer for the input units. E.g. 3 for RBG and 1 for grayscale.
      output_shape: An integer for the output units. E.g. Any number that represents your total labels.
      hidden_units: An integer for the hidden units. E.g. Any number to change the filters in the layer.
    
    Return:
      A tensor of logits with the length provided as per the output_shape.
    """
    def __init__(self, input_shape: int, output_shape: int, hidden_units: int):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(input_shape, hidden_units, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_units*13*13,
                      output_shape)
        )

    def forward(self, x: torch.Tensor):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.classifier(x)
        return x
