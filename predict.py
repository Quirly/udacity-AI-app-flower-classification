#Import modules
import argparse
import os
from image_processing import ImageProcessing
from prediction_model import prediction_model

def get_input_predict_args():
    # Create Parse using ArgumentParser
    parser=argparse.ArgumentParser(description='Input of system parameters')
    parser.add_argument('--filepath', type=str, default='flowers/test/24/image_06849.jpg', help='Image to predict')
    parser.add_argument('--loaddir', type=str, default='saved_models', help='Directory for trained model') 
    parser.add_argument('--modelname', type=str, default='checkpoint-trained-model.pth', help='Modelname')
    parser.add_argument('--topk', type=int, default=5, help='Number of Top Classes')
    parser.add_argument('--c_n', type=str, default='cat_to_name.json', help='File with Class Names')
    parser.add_argument('--dev', type=str, default='cuda', help='GPU or CPU mode')  
    args=parser.parse_args()
    return args

if __name__=='__main__':
    
    print("-----------------------------------------------------")
    print("Predict.py application: Welcome!")
    print("-----------------------------------------------------")
    print("Folder with test image: /flowers/test/14/image_06052.jpg")
    print("Select by --filepath option: Path/Folder")
    print("-----------------------------------------------------")
    print("Folder with saved models: /saved_models")
    print("Select by --loaddir option: Path/Folder")
    print("-----------------------------------------------------")    
    print("Modelname: 'checkpoint-trained-model.pth' (default)")
    print("Select other by --modelname option")
    print("-----------------------------------------------------")
    print("Top K: 5 (default)")
    print("Select other by -topk option")
    print("-----------------------------------------------------")
    print("Device: GPU (default)")
    print("Select other by --gpu option: CPU")
    print("-----------------------------------------------------")
    print("Index file with names: cat_to_name.json (default)")
    print("Select other by --indexfile option")
    print("-----------------------------------------------------")
   
    
    active_dir=os.getcwd()
    
    parameter_in=get_input_predict_args()
    filepath=parameter_in.filepath
    load_directory=parameter_in.loaddir
    modelname=parameter_in.modelname
    topk=parameter_in.topk
    cat_names=parameter_in.c_n
    dev=parameter_in.dev
    
    print(parameter_in)
    print(active_dir)
    
    p_object=prediction_model()
    p_object.load_model(dev,load_directory,modelname)
    
    np_test_img=ImageProcessing(filepath)
    np_image=np_test_img.process_image()
    top_p, top_class=p_object.predict(np_image,dev,topk)

    print(top_p)
    print(top_class)
    
    p_object.show_me_probs(top_p,top_class,cat_names)
    
    
    