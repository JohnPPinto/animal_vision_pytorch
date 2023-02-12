# Animal Vision

This is a **[PyTorch](https://pytorch.org) Project to Classify Images of Animals (30 Different Animals)**.

Basic understanding of PyTorch and Convolution Neural Networks is needed to understand this project.

Questions, suggestions or corrections can be posted as a issues.

---

# Contents

[***Objective***](https://github.com/JohnPPinto/animal_vision_pytorch/edit/main/README.md#objective)

[***Overview***](https://github.com/JohnPPinto/animal_vision_pytorch/edit/main/README.md#overview)

[***Experiment***](https://github.com/JohnPPinto/animal_vision_pytorch/edit/main/README.md#experiment)

[***Scripting***](https://github.com/JohnPPinto/animal_vision_pytorch/edit/main/README.md#scripting)

[***Deployment***](https://github.com/JohnPPinto/animal_vision_pytorch/edit/main/README.md#deployment)

---

## Objective

**To build a model and a web application that can classify images of an animal.**

This project will have all the necessary worflow needed to conduct a deep learning project based on Image Classification using PyTorch library. 

## Overview

To Keep things simple, we will be seprating the workflow in two Phase: 

1. Experiment Phase
2. Deployment Phase.

For this project, the dataset is collected from the "Kaggle dataset" you can check the dataset from [here](https://www.kaggle.com/datasets/jerrinbright/cheetahtigerwolf), the basic worflow is to conduct some experiments on the dataset and models and once the method and model is finalized then move onto the deployment phase. 

This project explains the use of different methods performed durning a deep learning project and how to conclude your project by taking your model in the open world for a real use case.

## Experiment Phase

***[Experiment Notebook](https://github.com/JohnPPinto/animal_vision_pytorch/blob/main/Animal%20Vision%20-%20Experiment.ipynb)***

This is the starting point of the project workflow, we initiate the project with the necessary steps like Importing libraries and Data Collection, selecting only three classes, a small dataset for experimenting purpose, a phrase used in Machine Learning "Start with a Small Dataset".

Then we quickly move towards modelling a Baseline model and fit it to the dataset, performing this step helps in understanding the dataset and the behaviour patters of the dataset. To spice it up, we perform a data augmentation on the dataset and do not change to much of the model architecture.

Now moving forward and performing modelling  experiments by using state of the art models, which are pre-trained on some of the huge dataset. This method is called as **Transfer Learning**, complex models that are capable to produce great results on pretty much any dataset. So, performing feature extraction on my dataset using different pre-trained models and comparing the results.

After comparing and finalizing the model, time to collect the complete 30 classes dataset and clean the whole dataset, the reason we didn't clean the dataset initially was because cleaning is a rigorous process, visualization is the most critical factor durning data cleaning and modelling helps in channelling the workflow.

## Scripting

***[Helper Script Notebook](https://github.com/JohnPPinto/animal_vision_pytorch/blob/main/Animal%20Vision%20-%20Helper%20Script.ipynb) and [Model Script Notebook](https://github.com/JohnPPinto/animal_vision_pytorch/blob/main/Animal%20Vision%20-%20Model%20Script.ipynb)***

This section is not a part of a workflow but has a equal importance durning a project, my most preferable method to do any project is to first try all my experiments on a jupyter notebook and move certains programs on the python script this helps in using this scripts to quickly perform some additional experiments.

As the name of the notebook same are the purpose of those scripts are produced, scripts are easy to understand if you have a closer look to it.

## Deployment Phase

***[Deployment Notebook](https://github.com/JohnPPinto/animal_vision_pytorch/blob/main/Animal%20Vision%20-%20Deployment.ipynb)***

The Final Phase of the project pipeline, here we have all the necessary tools ready to conduct our final experiments and deploy our model in the real world.

Similar to the experiment notebook, performing steps like importing libraries and data collection. Then using the same pre-trained neural networks on the big dataset and comparing the results.

Once everything looks good and perfect to move forward, we start the deployment experiment using the [Gradio library](https://gradio.app/), again using the machine learning methodology performing a small scale deployment first on the notebook and then on the open world environment, for this we use the [HuggingFace](https://huggingface.co/) which stramlines the web application space for the model. After concluding with the small scale application, we move to the bigger dataset and perform the same process and deploy it on the HuggingFace spaces.

Links of the web application: [Animal Vision Mini](https://huggingface.co/spaces/JohnPinto/animalvision_mini) and [Animal Vision Big](https://huggingface.co/spaces/JohnPinto/Animal_Vision_Big)
