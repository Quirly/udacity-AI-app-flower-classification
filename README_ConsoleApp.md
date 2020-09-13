# udacity-AI-app-flower-classification
Jupyter notebook for image classification of flowers

## purpose
**AI Python Console Application** for image classification, predicting types flowers, using transfer learning and on ImageNet pretrained PyTorch modes
(Udacity project within Nanodegree program 'AI programming with Python')

## installation
* install [Python 3] (https://www.python.org/downloads/)
* ensure that torch, torchvision, pandas, matplotlib, numpy, PIL modules are available within Python installation

## disclaimer
* Please use cloud services (as AWS GPUs) or hardware with a high performance graphics processing unit (GPU)
* Training of the network will take much longer on regular CPUs

## preparation
* Before any image can be classified you need to train your model
* If you use default settings (e.g. vgg16 as architecture) you just can start training by typing in 'python train.py' in your terminal
* Training will start with default setting (CNN architecture 'vgg16', 102 output classes, one hidden layer with 4096 
  neurons...)
* If training has started successfully you can see the output of the training process in your terminal output window
  (Training on a GPU environment can take up to one hour, one CPU environment much more)
  After each batch of 16 images you will get a printed update of current loss states and model accuracy 
* When training and validation will be finished you will see a message that training is finished
* Your model parameters will be saved as 'checkpoint_for_trained_model.pth') in the same folder like your train.py file

## input
* upload a test image of a flower you would like to classify
* **edit the image_path and name in ln[19] and put in your path and your name of the image you would like to classify**

## loading your pretrained model and classify image
* Before any image can be classified and your application was shut down you need to load your model
* You can load your trained model by typing in 'Python predict.py' in your terminal window
* Your model parameters will be loaded from 'checkpoint_for_trained_model.pth') in the same folder like your python file

## output
* output is shown in your terminal window
* **You will see a list with the five flower types your network thinks are the right flower types with the respecitve possibilities**
