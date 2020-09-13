# udacity-AI-app-flower-classification
Jupyter notebook for image classification of flowers

## purpose
AI Application for image classification, predicting types flowers, using transfer learning and on ImageNet pretrained PyTorch modes
(Udacity project within Nanodegree program 'AI programming with Python')

## installation
* install [Python 3] (https://www.python.org/downloads/)
* install [JupyterLab] (https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html)
* ensure that torch, torchvision, pandas, matplotlib, numpy, PIL modules are available within Python installation

## disclaimer
* Please use cloud services (as AWS GPUs) or hardware with a high performance graphics processing unit (GPU)
* Training of the network will take much longer on regular CPUs

## preparation
* Before any image can be classified you need to train your model
* If you use default settings (e.g. vgg16 as architecture) you just can start training by choosing 'Run All Cells'
  from the 'Cell'-menu of your Juypter app
* Training will start with default setting (CNN architecture 'vgg16', 102 output classes, one hidden layer with 4096 
  neurons...)
* If training has started successfully you can see the output of the training process below cell ln[10]
  (Training on a GPU environment can take up to one hour, one CPU environment much more)
  After each batch of 16 images you will get a printed update of current loss states and model accuracy 
* When training and validation will be finished you will see a message that training is finished
* Below cell ln[11] a plot will show you development of accuracy over training period
* In ln[12] you can test your trained model with a test data set of 8 pictures
* Your model parameters will be saved as 'checkpoint_jupyther.pth') in the same folder like your Jupyter notebook file

## loading your pretrained model
* Before any image can be classified and your application was shut down you need to load your model
* In ln[13] you can load your trained model
* Your model parameters will be loaded from 'checkpoint_jupyther.pth') in the same folder like your Jupyter notebook file
* Please run ln[16],ln[17],ln[18] once for initialization

## input
* upload a test image of a flower you would like to classify
* **edit the image_path and name in ln[19] and put in your path and your name of the image you would like to classify**

## how to classify your flower
* run ln[19]

## output
* output is shown below cell ln[19]
* **You will see a plot with the five flower types your network thinks are the right ones with the respecitve possibilities**
