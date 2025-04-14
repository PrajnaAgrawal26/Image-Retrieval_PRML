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
- [Demo 1](assets/Demo1.mp4)
- [Demo 2](assets/Demo2.mp4)
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
The  application  is  already  deployed  and  can  be  accessed  at  the  following  link:

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


### Models

1. **Vortex**  
   A ResNet-32 based image classification model using `ResidualBlock` for efficient residual learning. It includes downsampling, average pooling, and a fully connected layer to classify images into 10 classes.

2. **Aether**  
   A multinomial logistic regression model for supervised multi-class classification. It uses the softmax function to estimate class probabilities and assigns the class with the highest probability.

3. **Nuvora**  
   An ensemble model using a Random Forest Classifier with 100 decision trees. It improves generalization through bootstrapping and majority voting for robust multi-class classification.

4. **Orion**  
   An unsupervised learning model using K-Means clustering to group data into *k* clusters. After clustering, labels are assigned via majority voting, allowing class prediction for new data.

---

**Note:**  
Models 2–4 utilize **ResNet50** for feature extraction and apply **Linear Discriminant Analysis (LDA)** for dimensionality reduction, preserving class-discriminative information and improving overall performance.
