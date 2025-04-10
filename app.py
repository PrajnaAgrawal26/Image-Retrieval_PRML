import streamlit as st
st.set_page_config(page_title="CIFAR-10 Classifier", layout="centered")

import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import random
import warnings
import os
import pickle
from torchvision import models
import torch.nn as nn

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

st.title("CIFAR-10 Image Classifier")
st.markdown("Upload an image and see the predicted class, along with 5 similar CIFAR-10 samples.")

# Model choice
model_choice = st.selectbox("Choose a model", ["resnet32", "resnet50+LDA+LR"])

# CIFAR-10 classes
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if model_choice == "resnet32":
    from Model.resnet_model import model, device as resnet_device
    checkpoint_path = 'Checkpoints/resnet32.ckpt'
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

elif model_choice == "resnet50+LDA+LR":
    with open("Checkpoints/LR.pkl", "rb") as f:
        pipeline = pickle.load(f)

    lda = pipeline['lda']
    clf = pipeline['classifier']

    resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    resnet50.fc = nn.Identity()
    resnet50.to(device)
    resnet50.eval()

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "jfif"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="🖼 Uploaded Image", width=200)

    # Preprocessing
    if model_choice == "resnet32":
        preprocess = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2470, 0.2435, 0.2616))
        ])
    else:  # resnet50+LDA+LR
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    input_tensor = preprocess(image).unsqueeze(0).to(device)

    # Prediction
    with torch.no_grad():
        if model_choice == "resnet32":
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)
            pred_class = classes[predicted.item()]
        else:
            features = resnet50(input_tensor)
            features = features / features.norm(dim=1, keepdim=True)
            reduced_feat = lda.transform(features.cpu().numpy())
            pred_ind = clf.predict(reduced_feat)[0]
            pred_class = classes[pred_ind]

    st.success(f"✅ Predicted Class: **{pred_class.upper()}**")

    # Load test set
    transform_simple = transforms.Compose([transforms.ToTensor()])
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform_simple)

    target_class_idx = classes.index(pred_class)
    matching_images = [img for img, label in test_set if label == target_class_idx][:5]

    st.markdown(f"### 🔍 5 CIFAR-10 Test Images Predicted as: `{pred_class}`")

    cols = st.columns(5)
    for i in range(5):
        npimg = matching_images[i].numpy()
        npimg = np.transpose(npimg, (1, 2, 0))
        cols[i].image(npimg, caption=pred_class, use_container_width=True)
