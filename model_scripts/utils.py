"""
Contains various utilities function for PyTorch model training and saving.
"""
import torch
from pathlib import Path

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    """
    Saves a PyTorch Model to a directory.
    Args:
      model: A PyTorch model of nn.Module type.
      target_dir: A string of directory path.
      model_name: A filename for the saved model.
                  Should include .pth or .pt as a file extention.
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)
  
    # Create model save path
    assert model_name.endswith('.pth') or model_name.endswith('.pt'), 'model_name should end with .pth or .pt'
    model_save_path = target_dir_path / model_name
  
    # save the model state_dict
    print(f'\n[INFO] Saving Model to: {model_save_path}')
    torch.save(obj=model.state_dict(),
               f=model_save_path)
