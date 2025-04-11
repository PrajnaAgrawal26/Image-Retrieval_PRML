import numpy as np
import tqdm
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision import models
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib 

# Step 1: Load features and labels
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=False, num_workers=2)

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Step 2: Extract features from ResNet50
resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
resnet50.fc = nn.Identity()  # Remove final classification layer
resnet50 = resnet50.to(device)
resnet50.eval()

cnn_features = []
labels = []

with torch.no_grad():
    for images, targets in tqdm(trainloader, desc="Extracting CNN Features"):
        images = images.to(device)
        feats = resnet50(images)
        feats = feats / feats.norm(dim=1, keepdim=True)  # Normalize features
        cnn_features.append(feats.cpu().numpy())
        labels.append(targets.numpy())

X_train = np.concatenate(cnn_features, axis=0)
y_train = np.concatenate(labels, axis=0)
print("Extracted feature shape:", X_train.shape)


testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

# Step 3: Extract features for test set using ResNet50
cnn_features_test = []
labels_test = []

with torch.no_grad():
    for images, targets in tqdm(testloader, desc="Extracting Test CNN Features"):
        images = images.to(device)
        feats = resnet50(images)
        feats = feats / feats.norm(dim=1, keepdim=True)
        cnn_features_test.append(feats.cpu().numpy())
        labels_test.append(targets.numpy())

X_test = np.concatenate(cnn_features_test, axis=0)
y_test = np.concatenate(labels_test, axis=0)

# Build pipeline with LDA and RandomForestClassifier
model = Pipeline([
    ("lda", LinearDiscriminantAnalysis(n_components=9)),
    ("rf", RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    ))
])

# Fit the model
model.fit(X_train, y_train)

# Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Random Forest Train Accuracy after LDA: {train_accuracy:.4f}")
print(classification_report(y_train, y_train_pred))

print(f"Random Forest Test Accuracy after LDA: {test_accuracy:.4f}")
print(classification_report(y_test, y_test_pred))

# Save the model
joblib.dump(model, "RF.pkl")
print("Model saved as 'RF.pkl'")
