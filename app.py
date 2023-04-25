from flask import Flask, request
import base64
from PIL import Image
from preprocess4 import Pr1
import torch
from torchvision import models
import torch.nn as nn
import numpy as np
import cv2
from torchvision.transforms import Resize, ToTensor

def get_net():
    finetune_net = nn.Sequential()
    finetune_net.features = models.resnet18(weights='ResNet18_Weights.DEFAULT')


    finetune_net.output_new = nn.Sequential(nn.Linear(1000, 256),
                                            nn.ReLU(),
                                            nn.Linear(256, 26))

    finetune_net = finetune_net.to('cpu')

    for param in finetune_net.features.parameters():
        param.requires_grad = False
    return finetune_net


# Load the saved parameters
saved_params = torch.load('my_model61.pt', map_location=torch.device('cpu'))

# Create a new instance of the model and load the parameters
model_test = get_net()
model_test.load_state_dict(saved_params)
model_test.eval()
classes = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload():
    try:
        print(0)
        # Get the base64 image string from the request
        base64_image = request.json['image']
        print(1)
        # Decode the base64 image string to bytes
        image_bytes = base64.b64decode(base64_image)
        print(2)
        
        # Convert the image bytes to a numpy array
        nparr = np.fromstring(image_bytes, np.uint8)
        print(3)
        # Decode the numpy array to an image using OpenCV
        frame = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        print(4)
        
        # Process the image as needed
        p1 = Pr1(frame)
        print(5)
        processed_frame = p1.detect_crop_and_segment_hands(p1.image)
        print(6)
        if processed_frame is not None: 
            print(7)
            cropped_hand_array = Image.fromarray(processed_frame)
            print(cropped_hand_array)
            print(8)
            
            # Resize the image to a specific size
            resize = Resize((224, 224))
            print(8.5)
            cropped_hand_pil = resize(cropped_hand_array)
            #print(9)
            # Convert the image to a PyTorch tensor
            to_tensor = ToTensor()
            print(9.25)
            #cropped_hand_tensor = to_tensor(cropped_hand_pil)
            print(9.5)
            #Make a prediction using the model
            #prediction = model_test(cropped_hand_tensor[None].to("cpu")) 
            print(10)
            # Get the predicted label
            #pred_label = classes[torch.max(prediction, dim=1)[1]]
            #print(pred_label)
        return {'status': 'success'}
    
    except Exception as e:
        return {'status': 'error', 'message': str(e)}


if __name__ == '__main__':
    app.run()
