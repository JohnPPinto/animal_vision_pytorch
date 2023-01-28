"""
Contains code to download the animal data
and process into the format required as per the 
torchvision ImageFolder function.
Data is downloaded through kaggle dataset, for this
kaggle.json file is required.
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
  
    # Copying three classes data from the raw data
    raw_data = Path('temp_data/cheetahtigerwolf/ANIMALS/ANIMALS')
    data_extracted_path = Path('temp_data/animal_data')
    animal_list = ['LION', 'TIGER', 'WOLF']
    if data_extracted_path.exists():
        print(f'\n{raw_data} directory exist')
    else:
        print('\nCopying raw data...')
        for i in animal_list:
            shutil.copytree(raw_data / i, data_extracted_path / i)
            print(f'{i} directory created and files copied in "{data_extracted_path/i}"')
  
    # Splitting the data in Train Test Split in ratio(80:20).
    data_path = Path('data')
    lion_tiger_wolf_path = data_path / 'lion_tiger_wolf'
    if lion_tiger_wolf_path.exists():
        print(f'\nSplit directory exist - {lion_tiger_wolf_path}')
    else:
        print('\nSpliting data in Train and Test set')
        splitfolders.ratio(data_extracted_path, str(lion_tiger_wolf_path), 42, (0.80, 0.20))
  
    shutil.rmtree(temp_data_path)
    print(f'\n{temp_data_path} directory is deleted')
