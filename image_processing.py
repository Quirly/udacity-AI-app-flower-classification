from PIL import Image
import numpy as np

class ImageProcessing():
    
    def __init__(self, test_image_path):
        self.pil_im=Image.open(test_image_path)

    #Function that processes a PIL image for use in a PyTorch model
    def process_image(self):
        ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
            returns an Numpy array
        '''
        #Image Processing Parameters Project
        #project_px_size is the smallest side (height or width) of the scaled picture
        #project_px_crop is the size of the centered square cut out of the image
        project_px_size=256
        project_px_crop=224

        width,height=self.pil_im.size

        #Calculate new width and new height depending upon which side is shorter
        if width>height:
            new_height=project_px_size
            if new_height>height:
                new_width=int(width*(height/new_height))
            else:
                new_width=int(width*(new_height/height))
        else:
            new_width=project_px_size
            if new_width>width:
                new_height=int(height*(width/new_width))
            else:
                new_height=int(height*(new_width/width))

        #Resize image
        self.pil_im=self.pil_im.resize((new_width,new_height))

        #Crop square of 224x224 px from center
        self.pil_im=self.pil_im.crop(((new_width-project_px_crop)//2, (new_height-project_px_crop)//2,
                        (new_width+project_px_crop)//2,(new_height+project_px_crop)//2))

        #Create NumPy-Array from Image
        np_image=np.array(self.pil_im)

        #Do Normalization on NumPy-Array
        channels_means=[0.485, 0.456, 0.406]
        channels_stds=[0.229, 0.224, 0.225]
        np_image=np_image/256
        np_image = (np_image - channels_means)/channels_stds

        #Transpose ColorChannel
        np_image = np_image.transpose((2, 0, 1))

        return np_image


