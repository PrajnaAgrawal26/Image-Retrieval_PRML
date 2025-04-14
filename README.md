# PRML CSL2050 Course Project

## Topic: Image Retrieval

This project implements an image retrieval system using machine learning techniques. The system is designed to classify images from the CIFAR-10 dataset and retrieve visually similar images based on the predicted class.

---

## Team Members:
- **Harshita Vachhani (B23EE1026)**
- **Prajna Agrawal (B23CS1054)**
- **Sonam Sikarwar (B23CM1060)**
- **Sreenitya Thatikunta (B23CS1072)**
- **Nishkarsh Verma (B23CM1028)**

---

## Demo Videos:
<!-- - [Demo Video 1: Overview of the Project](#)
- [Demo Video 2: How to Use the Web Application](#)
- [Demo Video 3: Model Training and Evaluation](#) -->

---

## Repo Structure:
```
Directory structure:
└── prajnaagrawal26-image-retrieval_prml/
    ├── README.md
    ├── app.py
    ├── requirements.txt
    ├── Checkpoints/
    │   ├── LR.pkl
    │   ├── RF.pkl
    │   ├── kmeans+lda+cnn.pkl
    │   └── resnet32.ckpt
    ├── Experiments/
    │   ├── CNN+SVM.ipynb
    │   ├── CNN_DT.ipynb
    │   ├── CNN_Kmeans.ipynb
    │   ├── DT.ipynb
    │   ├── HOG+SVM.ipynb
    │   ├── HoG_DT.ipynb
    │   ├── HoG_Kmeans.ipynb
    │   ├── HoG_LDA_Kmeans.ipynb
    │   ├── KNN.ipynb
    │   ├── LDA+CNN+DT.ipynb
    │   ├── PCA+CNN+DT.ipynb
    │   └── PCA+DT.ipynb
    └── Model/
        ├── Aether.py
        ├── Nuvora.py
        ├── __init__.py
        ├── kmeans.py
        └── resnet_model.py
```
The ```Experiments/``` folder contains Jupyter notebooks for various experiments conducted during the project.

The ```Checkpoints/``` folder contains pre-trained model weights for all the implemented pipelines.

The ```Model/``` folder contains the implementation of the final models namely Vortex, Aether, Orion and Nuvora.

## How to Use:

### Option 1: Use the Deployed Application
The application is already deployed and can be accessed at the following link:

[ImageRetrieval](http://34.131.53.70:8501/)

### Option 2: Run Locally

#### 1. Clone the Repository
```bash
git clone https://github.com/PrajnaAgrawal26/Image-Retrieval_PRML
cd Image-Retrieval_PRML
```

#### 2. Create a Virtual Environment
```
python3 -m venv env

source env/bin/activate  # On Mac/Linux
.\env\Scripts\activate   # For Windows
```

#### 3. Install Dependencies
```
pip install -r requirements.txt
```

#### 4. Run the application
```
streamlit run app.py
```

### Using the application

1. **Select a Model**: Choose one of the available models (`Vortex`, `Aether`, `Orion`, `Nuvora`) from the dropdown menu.
2. **Upload an Image**: Upload an image in `.jpg`, `.jpeg` or `.png` format.
3. **View Predictions**:
   - The application will display the predicted class for the uploaded image.
   - It will also show 5 visually similar images from the CIFAR-10 dataset that belong to the same predicted class.


### About the Models:

1. **Vortex**: This is a implementation of a ResNet-32 model for image classification. It uses `ResidualBlock` to enable residual learning, which helps mitigate the vanishing gradient problem in deep networks. The model consists of three layers, each with multiple residual blocks, and includes downsampling for feature map reduction. The final output is passed through an average pooling layer and a fully connected layer to classify images into 10 classes.

The following models utilize ResNet50, a deep convolutional neural network, to extract high-level features from input data, capturing complex patterns and representations. These features are then passed through Linear Discriminant Analysis (LDA), which reduces dimensionality while preserving class-discriminative information, enhancing both efficiency and classification performance.

2. **Aether**: This is a supervised classification model based on multinomial **logistic regression**, which extends logistic regression to handle multiple classes. It models the relationship between input features and class probabilities using the softmax function. The model learns weight vectors for each class by minimizing the difference between predicted and actual class labels. During prediction, it assigns a sample to the class with the highest computed probability.

3. **Nuvora**: This is a multi-class classification model that uses ensemble learning via a **Random Forest** Classifier, combining 100 decision trees trained on bootstrapped subsets of data. Each tree is constrained in depth and minimum sample size to reduce overfitting and enhance generalization. During prediction, class labels are determined by majority voting across all trees, ensuring robust and accurate results.

4. **Orion**: Orion is an unsupervised learning model that uses the **K-Means clustering** algorithm to group data into k distinct clusters based on similarity. It iteratively assigns data points to the nearest centroid and updates centroids as the mean of assigned points until convergence. Although it doesn’t use labels during training, clusters are later mapped to class labels using majority voting for evaluation. New data is classified by assigning it to the nearest cluster and using the corresponding label.