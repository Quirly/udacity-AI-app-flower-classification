#Import modules
import argparse
import os
from classifier_for_model import classifier_for_model

def get_input_args():
    
    # Create Parse using ArgumentParser
    parser=argparse.ArgumentParser(description='Input of system parameters')
    parser.add_argument('--imagedir', type=str, default='flowers/', help='Image Folder')
    parser.add_argument('--savedir', type=str, default='saved_models/', help='Directory for saving trained model') 
    parser.add_argument('--arch', type=str, default='vgg16', help='CNN Model Architecture')
    parser.add_argument('--indexfile', type=str, default='cat_to_name.json', help='Json File with Flower Names')
    parser.add_argument('--l_r', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--h_u', type=int, default=4096, help='CNN Model Hidden Units')
    parser.add_argument('--epochs', type=int, default=2, help='Number of Training Epochs')  
    args=parser.parse_args()
    return args

if __name__=='__main__':
    
    print("-----------------------------------------------------")
    print("Train.py application: Welcome!")
    print("-----------------------------------------------------")
    print("Provided pretrained models: vgg16 (default)")
    print("Select by --arch option: alexnet,densenet161,vgg19")
    print("-----------------------------------------------------")    
    print("Neurons in hidden layer: 4096 (default)")
    print("Select other by --h_u option")
    print("-----------------------------------------------------")
    print("Epochs: 5 (default)")
    print("Select other by --epochs option")
    print("-----------------------------------------------------")
    print("Learning Rate: 0.001 (default)")
    print("Select other by --l_r option")
    print("-----------------------------------------------------")
    print("Index file with names: cat_to_name.json (default)")
    print("Select other by --indexfile option")
    print("-----------------------------------------------------")
    print("Image directory: /flowers (default)")
    print("Select other by --imagedir option")
    print("-----------------------------------------------------")
    print("Save directory: Current directory (default)")
    print("Select other by --savedir option")
    print("-----------------------------------------------------")    
    
    #read current directory
    active_dir=os.getcwd()
    
    #define 
    parameter_in=get_input_args()
    image_directory=parameter_in.imagedir
    save_directory=parameter_in.savedir
    pretrained_model=parameter_in.arch
    file_with_image_label_names=parameter_in.indexfile
    learning_rate=parameter_in.l_r
    neurons_hidden_layer=parameter_in.h_u
    epochs=parameter_in.epochs
    
    print(parameter_in)
    print(active_dir)
    
    num_classes=102
    device='cuda'
    nn_object=classifier_for_model()
    dataloaders_train,dataloaders_validiation=nn_object.prepare_data()


    input_size=nn_object.model_definition(str(pretrained_model))
    classifier,criterion, optimizer=nn_object.model_foward_pass(input_size,neurons_hidden_layer, learning_rate, num_classes)
    nn_object.model_train_validate(epochs,device,dataloaders_train,dataloaders_validiation,criterion,optimizer)
    print("-----------------------------------------------------")
    print("Train.py application: Work is done - I have finished!")
    print("Network used for Transfer learning:",pretrained_model)
    print("Parameters Epochs:",epochs)
    print("-----------------------------------------------------")
    
    #Save Model
    nn_object.model_save(save_directory,dataloaders_train,optimizer,pretrained_model,input_size,num_classes)
     
    print("-----------------------------------------------------")