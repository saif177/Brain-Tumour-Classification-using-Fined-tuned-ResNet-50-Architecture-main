import os
import time
import streamlit as st
from streamlit.elements import progress
from streamlit.elements.image_proto import ImageFormatWarning
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms, models

device_name = "cpu"
device = torch.device(device_name)
@st.cache
def model():
    resnet_model = models.resnet50(pretrained=True)
    for param in resnet_model.parameters():
        param.requires_grad = True
    n_inputs = resnet_model.fc.in_features
    resnet_model.fc = nn.Sequential(nn.Linear(n_inputs, 64),
                                    nn.SELU(),
                                    nn.Dropout(p=0.4),
                                    nn.Linear(64,64),
                                    nn.SELU(),
                                    nn.Dropout(p=0.4),
                                    nn.Linear(64, 4),
                                    nn.LogSigmoid())
    for name, child in resnet_model.named_children():
        for name2, params in child.named_parameters():
            params.requires_grad = True
    resnet_model.to(device)
    resnet_model.load_state_dict(torch.load('bt_resnet50_model.pt',map_location=torch.device('cpu')))
    resnet_model.eval()
    return resnet_model

@st.cache
def predict(img_file_buffer):
    transform = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])
    LABELS = ['None','Meningioma','Glioma','Pitutary']
    if img_file_buffer is not None:
        image = Image.open(img_file_buffer)
        img = transform(image)
        img = img[None, ...]
        with torch.no_grad():
            y_hat = model().forward(img.to(device))
            predicted = torch.argmax(y_hat.data, dim=1)
            caption=LABELS[predicted.data]
            return caption
           
def main():
    """ Brain tumor classifier"""
    html_temp="""
     <style>
        .title h2{
          user-select: none;
          font-size: 43px;
          color: white;
          background: repeating-linear-gradient(-45deg, red 0%, yellow 7.14%, rgb(0,255,0) 14.28%, rgb(0,255,255) 21.4%, cyan 28.56%, blue 35.7%, magenta 42.84%, red 50%);
          background-size: 600vw 600vw;
          -webkit-text-fill-color: transparent;
          -webkit-background-clip: text;
          animation: slide 10s linear infinite forwards;
        }
        @keyframes slide {
          0%{
            background-position-x: 0%;
          }
          100%{
            background-position-x: 600vw;
          }
        }
        
    </style> 
    <div class="title">
          <h2><strong>     Deep Mind    </strong> &#129504;</h2>
    </div>
    """
    st.markdown(html_temp,True)  
    st.write("")
    st.header("A Brain Tumor Classifier.")
    st.write("")
    st.header("This Web App perform the following tasks:")
    st.write("")
    st.subheader("1. Transform the uploaded images into Vector form. ")
    st.subheader("2. Classify the MRI Image into [ Meningioma Tumor, Glioma Tumor, pitutary Tumor ] by forwarding transformed image to our model. ")
    st.subheader("3. Display the tumour image with the tumor type. ")
    st.write("")
    st.write("")
    st.write("")
    st.subheader(" ------------  *Upload the MRI Image*  ------------ ")
    img = st.file_uploader("",type=["png", "jpg", "jpeg"])
    caption=predict(img)
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.write("")
    
    if st.button("Classify"):
       
        st.image(img, caption="Uploaded", width=400)
        st.write("")
        st.write("")
        progress=st.progress(0)
        for i in range(100):
            time.sleep(0.1)
            progress.progress(i+1)
        st.success("Successfully classifying MRI")
        st.header("The MRI scan has " + caption + " tumor")
        
            
if __name__=='__main__':
    main()