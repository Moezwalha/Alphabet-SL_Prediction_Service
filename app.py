from flask import Flask, request
import base64
from PIL import Image
from io import BytesIO
from preprocess4 import Pr1
from preprocess2 import test_transforms
import torch
from torchvision import models
import torch.nn as nn
import numpy as np
import cv2

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
        return {'status': 'success'}
    
    except Exception as e:
        return {'status': 'error', 'message': str(e)}

if __name__ == '__main__':
    app.run()
