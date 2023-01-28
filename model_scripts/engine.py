"""
Contains training and testing function in PyTorch
"""
import torch
from torch.utils.tensorboard.writer import SummaryWriter
from datetime import datetime
import os
from tqdm.auto import tqdm
from typing import List, Dict, Tuple

# Function for training the model
def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    """
    Trains a PyTorch model.
    Turns the model to a training mode and applies steps
    like loss, accuracy metrics, optimizer.
    Args:
        model: A PyTorch model to be trained.
        dataloader: A Dataloader instance that needs to be trained on.
        loss_fn: A PyTorch Loss function for calculating the loss durning the training.
        optimizer: A Pytorch Optimizer to help reduce the loss function.
        device: A target device either 'cuda' or 'cpu'.
    Returns:
        A tuple of training loss and training accuracy metrics. 
    """
    # model in train mode
    model.train()
    # setup train_loss and train_acc
    train_loss, train_acc = 0, 0
    # Loop through dataloader
    for batch, (X, y) in enumerate(dataloader):
        # define the target device to data
        X, y = X.to(device), y.to(device)
        # forward pass
        y_pred = model(X)
        # loss and accuracy
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        # backward propagation    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # calculate accuracy metrics
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)
    # Calculate the loss and accuracy for the model
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    return train_loss, train_acc

# function for testing the model
def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
    """
    Test a PyTorch model.
    Turns the model to a evaluation mode and applies steps
    like loss and accuracy metrics.
    Args:
        model: A PyTorch model to be trained.
        dataloader: A Dataloader instance that needs to be trained on.
        loss_fn: A PyTorch Loss function for calculating the loss durning the training.
        device: A target device either 'cuda' or 'cpu'.
    Returns:
        A tuple of testing loss and testing accuracy metrics.  
    """
    model.eval()
    # setup train_loss and train_acc
    test_loss, test_acc = 0, 0
    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through dataloader
        for batch, (X, y) in enumerate(dataloader):
            # define the target device to data
            X, y = X.to(device), y.to(device)
            # forward pass
            test_pred_logits = model(X)
            # calculate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()
            # calculate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))
            # Calculate the loss and accuracy for the model
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc

# Function for tracking different experiment
def create_writer(experiment_name: str,
                  model_name: str,
                  extra: str=None) -> torch.utils.tensorboard.writer.SummaryWriter():
    """
    Creates a log directory using tensorboard, in a format runs/timestamp/experiment_name/model_name/extra.
    Args: 
      experiment_name: Name of experiment,
      model_name: Name of the model,
      extra: Anything extra to add to the directory. Default to None.
    Returns:
      torch.utils.tensorboard.writer.SummaryWriter(): Instance of a writer saving to log_dir
    """
    timestamp = datetime.now().strftime("%d-%m-%Y") # time in format DD-MM-YYYY
    if extra:
        log_dir = os.path.join('runs', timestamp, experiment_name, model_name, extra)
    else:
        log_dir = os.path.join('runs', timestamp, experiment_name, model_name)
    print(f'\n[INFO] Creating SummaryWriter, saving to: {log_dir}...')
    return SummaryWriter(log_dir=log_dir)

# function for training and testing the model for n epochs.
def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          epochs: int,
          device: torch.device,
          writer: torch.utils.tensorboard.writer.SummaryWriter) -> Dict[str, List]:
    """
    Train and test a PyTorch Model
  
    Passing a model through train_step and test_step for certain
    number of epochs and training the model for that epoch loop.
    Calculate, print and stores all the metrics durning the training loop.
  
    Args:
        model: A pytorch model to be trained.
        train_dataloader: A dataloader only for train_step.
        test_dataloader: A dataloader only for test_step.
        loss_fn: A PyTorch Loss function for calculating the loss durning the training and testing.
        optimizer: A Pytorch Optimizer to help reduce the loss function while training the model.
        epochs: An integer indicating how much epochs to train the model for.
        device: A target device either 'cuda' or 'cpu'.
        writer: A SummaryWriter to save all the experiments.  
  
    Returns:
        A Dict containing all the values returned by train_step and test_step.
        results = {'train_loss': [],
                   'train_acc': [],
                   'test_loss': [],
                   'test_acc': []}
    """
    # Creating a result dict.
    results = {'train_loss': [],
               'train_acc': [],
               'test_loss': [],
               'test_acc': []}
  
    # Training and evaluation loop
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           device=device)
        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn,
                                        device=device)
        print(f'Epoch: {epoch+1} | train_loss: {train_loss:.4f} , train_acc: {train_acc:.4f} | test_loss: {test_loss:.4f}, test_acc: {test_acc:.4f}')
    
        # store every epoch results in the results Dict
        results['train_loss'].append(train_loss)
        results['train_acc'].append(train_acc)
        results['test_loss'].append(test_loss)
        results['test_acc'].append(test_acc)
    
        # Use the writer to track experiment
        if writer:
            writer.add_scalars(main_tag='loss',
                               tag_scalar_dict={'train_loss': train_loss,
                                                'test_loss': test_loss},
                               global_step=epoch)
            writer.add_scalars(main_tag='Accuracy',
                               tag_scalar_dict={'train_acc': train_acc,
                                                'test_acc': test_acc},
                               global_step=epoch)
            # close the writer
            writer.close()
        else:
            pass
  
    return results
