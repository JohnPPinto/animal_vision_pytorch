"""
Contains code to display the result
obtained after training the model.
"""
import matplotlib.pyplot as plt

# Plot loss and accuracy curves
def plot_curves(model_result: dict):
    """
    Plots the loss and accuracy curves for the training and evaluation.
    Args:
      model_result: A dict in format of {'train_loss': [],
                                         'train_acc': [],
                                         'test_loss': [],
                                         'test_acc': []} 
    """
    # Get loss values
    train_loss = model_result['train_loss']
    test_loss = model_result['test_loss']
  
    # get accuracy values
    train_accuracy = model_result['train_acc']
    test_accuracy = model_result['test_acc']
  
    # get total epochs
    epochs = range(len(model_result['train_loss']))
  
    # plot the curves
    plt.figure(figsize=(15, 7))
    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, test_loss, label='Test Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracy, label='Train Accuracy')
    plt.plot(epochs, test_accuracy, label='Test Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend();
