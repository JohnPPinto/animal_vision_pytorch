"""
Contains code to download the animal data
and process into the format required as per the 
torchvision ImageFolder function.
Data is downloaded through kaggle dataset, for this
kaggle.json api file is required.
"""
import os
from pathlib import Path
import shutil
from pip._internal.cli.main import main
package_names=['split-folders', 'opendatasets'] #packages to install
main(['install'] + package_names + ['--upgrade']) 
import splitfolders
import opendatasets as od

def download_data():
    """
    Download data from kaggle.
    Kaggle username and key will be needed to download the data.
    You can find your username and key from your account 
    Kaggle.com -> Account -> API -> click on 'Create New API Token'
    """
    # Download the dataset
    kaggle_dataset_url = 'https://www.kaggle.com/datasets/jerrinbright/cheetahtigerwolf'
    temp_data_path = Path('temp_data')
    if temp_data_path.exists():
        print(f'{temp_data_path} directory exist')
    else:
        print('Downloading data...')
        od.download_kaggle_dataset(kaggle_dataset_url,
                                   temp_data_path)

    # Splitting the data in Train Test Split in ratio(80:20).
    raw_data = Path('temp_data/cheetahtigerwolf/ANIMAL-N30/ANIMALS')
    data_path = Path('data')
    animal30_path = data_path / 'animal30_classes'
    if animal30_path.exists():
        print(f'\nSplit directory exist - {animal30_path}')
    else:
        print('\nSpliting data in Train and Test set')
        splitfolders.ratio(raw_data, str(animal30_path), 42, (0.80, 0.20), move=True)

    shutil.rmtree(temp_data_path)
    print(f'\n{temp_data_path} directory is deleted')
