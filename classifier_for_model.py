
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torchvision import datasets,transforms,models
#Import NumPy
import numpy as np
#Import Json
import json
#Import Time
import time
#Keep session
#import workspace_utils
from workspace_utils import active_session

class classifier_for_model():
    
    def __init__(self):
        self.model=None
        
    def prepare_data(self):
        
        #Load the data
        data_dir = 'flowers'
        train_dir = data_dir + '/train'
        valid_dir = data_dir + '/valid'
        test_dir = data_dir + '/test'
        
        #Preprocess images in datasets
        data_train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                                    transforms.RandomResizedCrop(224),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                                         [0.229, 0.224, 0.225])])

        data_validation_transforms = transforms.Compose([transforms.Resize(255),
                                                         transforms.CenterCrop(224),
                                                         transforms.ToTensor(),
                                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                                         [0.229, 0.224, 0.225])])

        data_test_transforms = transforms.Compose([transforms.Resize(255),
                                                   transforms.CenterCrop(224),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize([0.485, 0.456, 0.406],
                                                                         [0.229, 0.224, 0.225])])

        #Load the datasets with ImageFolder
        data_train_image_datasets = datasets.ImageFolder(train_dir,transform=data_train_transforms)
        data_validation_image_datasets = datasets.ImageFolder(valid_dir,transform=data_validation_transforms)
        data_test_image_datasets = datasets.ImageFolder(test_dir,transform=data_test_transforms)
        
        # Define dataloaders
        dataloaders_train = torch.utils.data.DataLoader(data_train_image_datasets,batch_size=64,shuffle=True)
        dataloaders_validation = torch.utils.data.DataLoader(data_validation_image_datasets,batch_size=16,shuffle=True)
        dataloaders_test = torch.utils.data.DataLoader(data_test_image_datasets,batch_size=8,shuffle=True)
        
        #Create a new dictionary for the dataset to map flower names to pictures as labels instead of integer values
        train_dict = data_train_image_datasets.class_to_idx
        validation_dict = data_validation_image_datasets.class_to_idx
        test_dict = data_test_image_datasets.class_to_idx
        
        print("Step 1 PASS: Images loaded to dataloaders")
        return dataloaders_train,dataloaders_validation

    def model_definition(self, pretrained_model_name):
        model_inputs={'vgg16': 25088, 'vgg19':25088, 'densenet161':2208, 'alexnet':9216}
        if pretrained_model_name in model_inputs:
            if pretrained_model_name=='vgg16':
                self.model=models.vgg16(pretrained=True)
            else:
                if pretrained_model_name=='vgg19':
                    self.model=models.vgg19(pretrained=True)
                else:
                    if pretrained_model_name=='densenet161':
                        self.model=models.densenet161(pretrained=True)
                    else:
                        if pretrained_model_name=='alexnet':
                            self.model=models.alexnet(pretrained=True)
            input_size=model_inputs.get(pretrained_model_name)
            for param in self.model.parameters():
                param.requires_grad=False
            print("Step 2 PASS: Model found and initialized!")
        else:
            print("Step 2 FAIL: Modelname is not part of list of models that can be chosen!")
        return input_size
            
    def model_foward_pass(self, input_size, hidden_units, learning_rate, num_classes):
        num_classes=102

        #Initialize model
        classifier=nn.Sequential(
                    nn.Linear(input_size, hidden_units),
                    nn.ReLU(True),
                    nn.Dropout(0.5),
                    nn.Linear(hidden_units, hidden_units),
                    nn.ReLU(True),
                    nn.Dropout(0.5),
                    nn.Linear(hidden_units, num_classes),
                    nn.LogSoftmax(dim=1))
        
        self.model.classifier=classifier

        #Define Loss function
        criterion=nn.NLLLoss()

        #Define Optimizer/Update
        optimizer=optim.Adam(self.model.classifier.parameters(),lr=learning_rate)
        
        device=torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
        self.model.to(device);
        print("Step 3 PASS: Model set up!")
        
        return classifier,criterion, optimizer

    def model_train_validate(self, epochs,device,dataloaders_train,dataloaders_validation,criterion,optimizer):
        
        steps=0
        steps_valid=0
        print_every=16
        running_loss=0
        validation_loss=0
        training_loss_sum=[]
        validation_loss_sum=[]
        print("-----------------------------------------------------")        
        print("STEP 4 PASS: Training Preparation ready")        
        print("I will start training...")
        print("-----------------------------------------------------")
        self.model.train()
        
        with active_session():
            for epoch in range(epochs-1):
            
                running_loss=0
            
                for ii, (images, labels) in enumerate(dataloaders_train):

                    #Increment Steps
                    steps+=1
                    #Move input and label tensors to the default device
                    images,labels=images.to(device),labels.to(device)
                    #Training Loop
                    start=time.time()

                    #Clear the gradients
                    optimizer.zero_grad()
                    #print(images.shape)
                    logps=self.model.forward(images)
                    loss=criterion(logps,labels)
                    loss.backward()
                    optimizer.step()

                    #Calculate Loss
                    running_loss+=loss.item()
                    training_loss=running_loss/(len(dataloaders_train))

                    #Append Loss data for visualization of training progress
                    training_loss_sum.append(training_loss)
                    validation_loss_sum.append(validation_loss)
                    print(f"Training Loss: {training_loss:.5f} Loop step({steps})")

                    if steps%print_every==0:

                        validation_loss=0
                        accuracy=0
                        self.model.eval()

                        with torch.no_grad():
                            validation_loss=0
                            accuracy=0
                            for images,labels in dataloaders_validation:
                                steps_valid+=1
                                #Move input and label tensors to the default device
                                images,labels=images.to(device),labels.to(device)
                                #Learning Step
                                logps=self.model.forward(images)
                                batch_loss=criterion(logps,labels)
                                validation_loss+=batch_loss.item()
                                #Append list for graphics
                                validation_loss_sum.append(validation_loss/(len(dataloaders_validation)))
                                #Calculate accuracy
                                ps=torch.exp(logps)
                                top_p,top_class=ps.topk(1,dim=1)
                                equals=top_class==labels.view(*top_class.shape)
                                accuracy+=torch.mean(equals.type(torch.FloatTensor)).item()

                        validation_loss_print=validation_loss/(len(dataloaders_validation))
                        validation_accuracy_print=(accuracy/(len(dataloaders_validation))*100)
                        print("------------------------------------------------------------")                    
                        print("Epoch {}/{}".format(epoch+1,epochs))
                        print("Training Loss: {:.5f}".format(training_loss/print_every))
                        print("Validation Loss: {:.5f}".format(validation_loss/len(dataloaders_validation)))
                        print("Accuracy Training State: {:.2f}%".format(accuracy))                                   
                        print("------------------------------------------------------------")

    def model_save(self, filepath, dataloaders_train,optimizer,arch_name,input_size,num_classes):
        #self.model.class_to_idx=dataloaders_train.class_to_idx
        checkpoint={'input size':input_size,
                    'output size':num_classes,
                    'state dict':self.model.state_dict(),
                    'arch':arch_name,
                    'classifier':self.model.classifier,
                    #'class_to_idx':self.model.class_to_idx,
                    'optimizer':optimizer.state_dict()}
        torch.save(checkpoint, filepath+"checkpoint-trained-model.pth")
        print("Model successfully saved!")          
               
        
        
