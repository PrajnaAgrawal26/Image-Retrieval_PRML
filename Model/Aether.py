"""
Aether: An image classification and retrieval pipeline using ResNet50, LDA, and Logistic Regression.

Pipeline Overview:
1. Data preprocessing: Loads and normalizes the CIFAR-10 dataset.
2. Feature extraction: Leverages pretrained ResNet50 to generate deep feature embeddings.
3. Dimensionality Reduction: Applies Linear Discriminant Analysis (LDA) to project features into a lower-dimensional space.
4. Classification: Trains a logistic regression model on the reduced features for label prediction.
5. Retrieval & Visualization: Predicts the class of a query image and displays visually similar images from the test set.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import pickle
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# step 1: data handling
class DataHandler:
    def __init__(self, batch_size=128):
        self.transform = transforms.Compose([                   # apply transformation
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.batch_size = batch_size                            # download dataset - cifar10
        self.train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=self.transform)
        self.test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=self.transform)

    def get_loaders(self):                                      # get dataloaders
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
        return train_loader, test_loader

    def get_datasets(self):
        return self.train_dataset, self.test_dataset


#step 2: feature extractor
class FeatureExtractor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")                  # initialize device
        # self.device = torch.device("mps" if torch.backends.mps.is_available() else self.device)   # for mac users
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)                       # load resnet50 weights
        self.model.fc = nn.Identity()                                                               # remove last fc layer                 
        self.model.to(self.device)
        self.model.eval()                                                                           # set model to eval mode

    def extract(self, dataloader):
        all_feats = []
        all_labels = []
        with torch.no_grad():                                                                       # iterate over dataloader
            for imgs, labels, *_ in tqdm(dataloader):
                imgs = imgs.to(self.device)
                feats = self.model(imgs)                                                                 
                feats = feats / feats.norm(dim=1, keepdim=True)                                     # normalize features
                all_feats.append(feats.cpu().numpy())                                               # append features to list
                all_labels.extend(labels.numpy())                                                   # append labels to list
        return np.vstack(all_feats), np.array(all_labels)


# step 3: dimentionality reduction
class LDAReducer:
    def __init__(self, n_components=9):
        self.lda = LDA(n_components=n_components)

    def fit_transform(self, X, y):
        return self.lda.fit_transform(X, y)

    def transform(self, X):
        return self.lda.transform(X)


# step 4: classifier
class Classifier:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs')   # 1000 iterations with lbfgs solver

    def train(self, X, y):
        self.model.fit(X, y)                                                # fit using train data

    def evaluate(self, X, y, class_names):
        preds = self.model.predict(X)                                       # predict using trained model     
        acc = accuracy_score(y, preds)                                      # accuracy
        print(f"Test Accuracy: {acc:.4f}")
        print(classification_report(y, preds, target_names=class_names))    # classification report


# step 5: visualizer
class PredictorVisualizer:
    def __init__(self, extractor, lda, clf, transform, test_dataset):
        self.extractor = extractor                                          # initialize parameters
        self.lda = lda
        self.clf = clf
        self.transform = transform
        self.test_dataset = test_dataset
        self.device = extractor.device

    def predict_and_show(self, img_path):
        img = Image.open(img_path).convert('RGB')                           # load + preprocess image         
        input_tensor = self.transform(img).unsqueeze(0).to(self.device)     # convert to tensor

        with torch.no_grad():
            feat = self.extractor.model(input_tensor)                       # extract features
            feat = feat / feat.norm()                                       # normalize

        lda_feat = self.lda.transform(feat.cpu().numpy())                   # reduce dimensionality

        pred = self.clf.model.predict(lda_feat)[0]                          # predict using classifier       
        pred_class = self.test_dataset.classes[pred]                        # get class name           

        print(f"\nPredicted class for query image: {pred_class}")

        matching_images = []
        for i in tqdm(range(len(self.test_dataset))):
            test_img, _ = self.test_dataset[i]
            input_tensor = test_img.unsqueeze(0).to(self.device)
            with torch.no_grad():
                feat = self.extractor.model(input_tensor)
                feat = feat / feat.norm()
            lda_feat = self.lda.transform(feat.cpu().numpy())           # preprocess using lda
            test_pred = self.clf.model.predict(lda_feat)[0]             # predict using classifier

            if test_pred == pred:                                       # if predicted class matches target class
                matching_images.append(test_img)                        # append to list
                if len(matching_images) == 5:
                    break

        fig, axes = plt.subplots(1, 6, figsize=(15, 4))                 # plot images
        axes[0].imshow(img)
        axes[0].set_title("Query Image")
        axes[0].axis('off')

        for i, example_img in enumerate(matching_images):
            example_img = example_img.permute(1, 2, 0).numpy()
            example_img = example_img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
            example_img = np.clip(example_img, 0, 1)

            axes[i+1].imshow(example_img)
            axes[i+1].axis('off')
            axes[i+1].set_title(f"{pred_class}")

        plt.tight_layout()
        plt.show()

# step 6: main function
def main():
    data_handler = DataHandler()                                        # load data
    train_loader, test_loader = data_handler.get_loaders()              # get dataloaders
    train_dataset, test_dataset = data_handler.get_datasets()           # get datasets

    extractor = FeatureExtractor()
    train_feats, train_labels = extractor.extract(train_loader)         # extract features for train and test
    test_feats, test_labels = extractor.extract(test_loader)

    lda = LDAReducer(n_components=9)                                    # lda for train and test sets
    X_train = lda.fit_transform(train_feats, train_labels)
    X_test = lda.transform(test_feats)

    clf = Classifier()
    clf.train(X_train, train_labels)                                    # train classifier
    clf.evaluate(X_test, test_labels, train_dataset.classes)

    with open("LR.pkl", "wb") as f:                                     # save lda and classifier in pkl file
        pickle.dump({
            'lda': lda.lda,
            'classifier': clf.model,
            'classes': train_dataset.classes,
        }, f)
    print("Saved all components to LR.pkl")

    visualizer = PredictorVisualizer(extractor, lda, clf, data_handler.transform, test_dataset) # visualizer
    visualizer.predict_and_show("test_images/plane1.jpg")

if __name__ == '__main__':
    main()