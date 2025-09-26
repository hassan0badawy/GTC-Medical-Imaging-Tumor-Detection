import torch
import numpy as np
import matplotlib as plt
import streamlit as st
import cv2
from PIL import Image
from torch import nn
import torch.nn.functional as F
#processing the image
def processImage(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #contrast the image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    #resize the image an normalize it
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    #sharpen the image
    kernel = np.array([[-1, -1, -1],
                   [-1,  9, -1],
                   [-1, -1, -1]])
    img = cv2.filter2D(img, -1, kernel)
    img = np.expand_dims(img, axis=0)
    image = []
    image.append(img)
    image = np.array(img)
    image = torch.FloatTensor(image)
    return image
#the model
class CNN(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(1,6,3,1)
    self.conv2 = nn.Conv2d(6,16,3,1)
    self.fc1 = nn.Linear(5*5*16,480)
    self.fc2 = nn.Linear(480,240)
    self.fc3 = nn.Linear(240,120)
    self.fc4 = nn.Linear(120,84)
    self.fc5 = nn.Linear(84,4)
  def forward(self,x):
    x = F.relu(self.conv1(x))
    x = F.max_pool2d(x,2,2)
    x = F.relu(self.conv2(x))
    x = F.max_pool2d(x,2,2)
    x = x.view(-1,16*5*5)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = F.relu(self.fc3(x))
    x = F.relu(self.fc4(x))
    x = self.fc5(x)
    return F.log_softmax(x,dim=1)
#loading the model
model = CNN()
model.load_state_dict(torch.load("brain_tumor_detection.py"))
#extracting the label
def predictLabel(image):
    image = processImage(image)
    labels = ["glioma","meningioma","no tumor","pituitary"]
    value = model(image)
    value = value.argmax()
    return labels[value]
#lauching the web app
st.title("Brain Tumor Detection")
uploaded_file = st.file_uploader("Upload MRI scan of the brain", type=["jpg","png","jpeg"])
if uploaded_file:
    #read image as byte
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    #convert image to normal
    image = cv2.imdecode(file_bytes,cv2.IMREAD_COLOR)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    label = predictLabel(image)
    st.subheader(f"Prediction: {label}")
    #st.progress(float(confidence))  # progress bar as confidence
    #st.write(f"Model Confidence: {confidence.item()*100:.2f}%")