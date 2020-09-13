
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torchvision import datasets,transforms,models
#Import NumPy
import numpy as np
#Import Json
import json
#import Pandas
import pandas as pd
#Import Time
import time
#Keep session
#import workspace_utils
from workspace_utils import active_session
from image_processing import ImageProcessing


class prediction_model():
    
    def __init__(self):
        self.model=None
        
    def load_model(self, device, filepath, modelname):
        
        model_inputs={'vgg16': 25088, 'vgg19':25088, 'densenet161':2208, 'alexnet':9216}
        
        #Choose device
        if torch.cuda.is_available():
            map_location=lambda storage, loc: storage.cuda()
        else:
            map_location='cpu'
            
        checkpoint=torch.load(str(filepath+"/"+modelname),map_location)
        
        pretrained_model_name=checkpoint['arch']
        
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

        self.model.classifier=checkpoint['classifier']
        self.model.load_state_dict(checkpoint['state dict'])
        #self.model.optimizer.state_dict(checkpoint['optimizer'])
        #self.model.class_to_idx=checkpoint['class_to_idx']

        for param in self.model.parameters():
            param.requires_grad=False
        
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model.eval()                                              
        self.model.to(device)
                                              
        print("Model successfully loaded!")                  

    def predict(self, np_image, device, topk):
        
        #Bring model to eval()-mode
        self.model.eval()
        #move model to activated device
        self.model.to(device)
        # Convert image from numpy to torch
        pre_np_array_prediction_image=np_image
        np_array_prediction_image=torch.from_numpy(pre_np_array_prediction_image).to(device)
        np_array_prediction_image = np_array_prediction_image.unsqueeze(0)
        #print(np_array_prediction_image.shape)
        self.model.eval()
        self.model.to(device)
        predict_log_ps = self.model(np_array_prediction_image.float())
        predict_ps=torch.exp(predict_log_ps)
        top_p,top_class=predict_ps.topk(5,dim=1)

        return top_p, top_class

    def show_me_probs(self, top_p, top_class, flower_name_path):
        
        with open(flower_name_path,'r') as f:
            cat_to_name=json.load(f)
        
        probs_np, labs_np = top_p.detach().cpu().numpy(), top_class.detach().cpu().numpy()
        new_probs=list(probs_np[0]*100)
        new_probs_floats=[round(float(i),2) for i in new_probs]
        #cares for labels
        new_labs=list(labs_np[0])
        labs_names=[]
        
        for flowernumber in new_labs:
            for nummer, bez in cat_to_name.items():
                if str(flowernumber)==str(nummer):
                    labs_names.append(bez)
                if str(flowernumber)==str(0):
                    labs_names.append(bez)
                    break
        data_probs={'Probabilities %':pd.Series(new_probs_floats,index=labs_names)}
        df = pd.DataFrame(data_probs)
        print("----------------------------------------------------")
        print(df)
        print("----------------------------------------------------")