"""
Contains functionality for creating 
PyTorch datasets and dataloaders for
Image classification data
"""
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()

def create_dataloaders(train_dir: str,
                       test_dir: str,
                       transform: transforms.Compose,
                       batch_size: int,
                         num_workers: int = NUM_WORKERS):
    """
    Create training and testing dataloaders.
    
    Args:
      train_dir: Train directory path,
      test_dir: Test directory path,
      transform: torchvision transforms compose class to transform the data,
      batch_size: Total sample per batch,
      num_workers: An integer for the number of workers durning dataloaders
    
    Return:
      Returns a tuple of (train_dataloaders, test_dataloaders, class_names).
    """
    # Create datasets using ImageFolder
    train_data = datasets.ImageFolder(root=train_dir, 
                                      transform=transform)
    test_data = datasets.ImageFolder(root=test_dir, 
                                     transform=transform)
    
    # Get class name
    class_names = train_data.classes
  
    # Create dataloaders
    train_dataloaders = DataLoader(dataset=train_data,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=num_workers,
                                   pin_memory=True)
    test_dataloaders = DataLoader(dataset=test_data,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_workers,
                                  pin_memory=True)
    
    return train_dataloaders, test_dataloaders, class_names
