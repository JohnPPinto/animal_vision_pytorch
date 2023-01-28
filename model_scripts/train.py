"""
Trains, evaluate and saves a PyTorch image classification model. 
"""

import os
import argparse
import torch
from torch import nn
import data_setup, engine, model_builder, utils
from torchvision import transforms
from datetime import datetime

# create a parser
parser = argparse.ArgumentParser(description='Get some hyperparameters')

# get experiment name
parser.add_argument('--exp_name',
                    default='experiment',
                    type=str,
                    help='Name of the experiment')
# get model name
parser.add_argument('--model_name',
                    default='model',
                    type=str,
                    help='Name of the model')
# get an arg for num_epochs
parser.add_argument("--num_epochs",
                    default=10,
                    type=int,
                    help='The number of epochs to train for')
# get an arg for batch_size
parser.add_argument("--batch_size",
                    default=32,
                    type=int,
                    help='The number of sample for batch_size')
# get an arg for hidden_units
parser.add_argument("--hidden_units",
                    default=10,
                    type=int,
                    help='The number of units for hidden layers')
# get an arg for learning_rate
parser.add_argument("--learning_rate",
                    default=0.001,
                    type=float,
                    help='learning rate for optimizer')
# get an arg for training directory
parser.add_argument("--train_dir",
                    default='data/lion_tiger_wolf/train',
                    type=str,
                    help='The path for training data')
# get an arg for testing directory
parser.add_argument("--test_dir",
                    default='data/lion_tiger_wolf/val',
                    type=str,
                    help='The path for test data')
# get our arg from the parser
args=parser.parse_args()

# setup hyperparameters
NUM_EPOCHS = args.num_epochs
BATCH_SIZE=args.batch_size
NUM_WORKERS = os.cpu_count()
HIDDEN_UNITS=args.hidden_units
LEARNING_RATE= args.learning_rate
print(f'\n[INFO] Training a model for {NUM_EPOCHS} epochs with batch size {BATCH_SIZE} using {HIDDEN_UNITS} hidden units with a learning rate of {LEARNING_RATE}')

# setup directories
train_dir = args.train_dir
test_dir = args.test_dir
print(f'[INFO] Training data directory: {train_dir}')
print(f'[INFO] Testing data directory: {test_dir}')

# setup target device
device='cuda' if torch.cuda.is_available() else 'cpu'

# Create transform
data_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Create dataloaders using data_setup script
train_dataloaders, test_dataloaders, class_names =  data_setup.create_dataloaders(train_dir=train_dir,
                                                                                  test_dir=test_dir,
                                                                                  transform=data_transform,
                                                                                  batch_size=BATCH_SIZE,
                                                                                  num_workers=NUM_WORKERS)

# Create model using model_builder script
model = model_builder.TinyVGG(input_shape=3,
                              output_shape=len(class_names),
                              hidden_units=HIDDEN_UNITS).to(device)

# set loss, accuracy and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

# define the summary writer and track our result through tensorboard
writer = engine.create_writer(args.exp_name,
                              args.model_name)

# start training using engine script
print(f'\n[INFO] Starting Model Training...')
engine.train(model=model,
             train_dataloader=train_dataloaders,
             test_dataloader=test_dataloaders,
             loss_fn=loss_fn,
             optimizer=optimizer,
             epochs=NUM_EPOCHS,
             device=device,
             writer=writer)

# save the model using utils script
utils.save_model(model=model,
                 target_dir='models',
                 model_name=args.model_name + '.pth')
