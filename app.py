import streamlit as st
st.set_page_config(page_title="CIFAR-10 Image Retrieval", layout="centered")

import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import warnings
import pickle
from torchvision import models
import torch.nn as nn
import random
import joblib

# Suppress future warnings for clean output
warnings.filterwarnings("ignore", category=FutureWarning)

# Streamlit UI setup
st.title("CIFAR-10 Image Retrieval")
st.markdown("Upload an image and see the predicted class, along with 5 similar CIFAR-10 samples predicted by the model.")

# Dropdown menu to choose model
model_choice = st.selectbox("Choose a model", ["Vortex", "Aether", "Orion", "Nuvora"])

# CIFAR-10 class labels
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load selected model
if model_choice == "Vortex":
    # Custom ResNet model for CIFAR-10 (32x32)
    from Model.resnet_model import model, device as resnet_device
    checkpoint_path = 'Checkpoints/resnet32.ckpt'
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

elif model_choice == "Aether":
    # Logistic Regression pipeline with LDA
    with open("Checkpoints/LR.pkl", "rb") as f:
        pipeline = pickle.load(f)
    lda = pipeline['lda']
    clf = pipeline['classifier']
    # Use ResNet50 for feature extraction
    resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    resnet50.fc = nn.Identity()
    resnet50.to(device)
    resnet50.eval()

elif model_choice == "Orion":
    # Clustering-based pipeline with KMeans and LDA
    with open("./Checkpoints/kmeans+lda+cnn.pkl", "rb") as f:
        pipeline = pickle.load(f)
    lda = pipeline['lda']
    kmeans = pipeline['kmeans']
    label_map = pipeline['label_map']
    resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    resnet50.fc = nn.Identity()
    resnet50.to(device)
    resnet50.eval()

elif model_choice == "Nuvora":
    # Random Forest pipeline with LDA
    pipeline = joblib.load("./Checkpoints/RF.pkl")
    lda = pipeline['lda']
    clf = pipeline['rf']
    resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    resnet50.fc = nn.Identity()
    resnet50.to(device)
    resnet50.eval()

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "jfif"])


if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", width=200)

# Define preprocessing based on model type
    if model_choice == "Vortex":
        preprocess = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2470, 0.2435, 0.2616))
        ])
    else:
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    input_tensor = preprocess(image).unsqueeze(0).to(device)

    # Prepare test dataset for retrieval of similar images
    with torch.no_grad():
        if model_choice == "Vortex":
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)
            pred_class = classes[predicted.item()]

        elif model_choice == "Orion":
            features = resnet50(input_tensor)
            features = features / features.norm(dim=1, keepdim=True)
            reduced_feat = lda.transform(features.cpu().numpy())
            cluster = kmeans.predict(reduced_feat)[0]
            pred_class = classes[label_map[cluster]]

        else:  # LR and RF
            features = resnet50(input_tensor)
            features = features / features.norm(dim=1, keepdim=True)
            reduced_feat = lda.transform(features.cpu().numpy())
            pred_ind = clf.predict(reduced_feat)[0]
            pred_class = classes[pred_ind]

    st.success(f"Predicted Class: **{pred_class.upper()}**")

    # Load test set
    if model_choice == "Vortex":
        test_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2470, 0.2435, 0.2616))
        ])
    else:
        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    raw_transform = transforms.ToTensor()
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    raw_test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=raw_transform)

    # Find 5 test images that the model predicts as the same class
    matching_images = []
    indices = list(range(len(test_set)))
    random.shuffle(indices)

    with torch.no_grad():
        for i in indices:
            input_img, _ = test_set[i]
            input_tensor = input_img.unsqueeze(0).to(device)

            if model_choice == "Vortex":
                output = model(input_tensor)
                _, pred = torch.max(output, 1)
                pred_class_idx = pred.item()

            elif model_choice == "Orion":
                feat = resnet50(input_tensor)
                feat = feat / feat.norm(dim=1, keepdim=True)
                reduced_feat = lda.transform(feat.cpu().numpy())
                cluster = kmeans.predict(reduced_feat)[0]
                pred_class_idx = label_map[cluster]

            else:  # LR and RF
                feat = resnet50(input_tensor)
                feat = feat / feat.norm(dim=1, keepdim=True)
                reduced_feat = lda.transform(feat.cpu().numpy())
                pred_class_idx = clf.predict(reduced_feat)[0]

            if classes[pred_class_idx] == pred_class:
                matching_images.append(raw_test_set[i][0])
            if len(matching_images) >= 5:
                break

    # Display retrieved matching images
    st.markdown(f"5 CIFAR-10 Test Images the Model Predicted as: `{pred_class}`")
    cols = st.columns(5)
    for i in range(5):
        npimg = matching_images[i].numpy()
        npimg = np.transpose(npimg, (1, 2, 0))
        cols[i].image(npimg, caption=pred_class, use_container_width=True)
