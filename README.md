# Facial Expression Recognition using Classical and Deep Learning Techniques

## 📌 Project Goal

The aim of this project is to develop and evaluate multiple approaches for classifying facial expressions using the FER2013 dataset. We explore both traditional feature-based methods (HOG, LBP) combined with machine learning models, as well as a custom Convolutional Neural Network (CNN). The ultimate goal is to determine which technique offers the best performance in recognizing emotions from facial images.

---

## 📂 Dataset Used

- **Dataset**: [FER2013](https://www.kaggle.com/datasets/msambare/fer2013)
- **Description**: A publicly available dataset consisting of 48x48 grayscale images of faces. Each image is labeled with one of the following 7 emotions:
  - Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
- **Split**:
  - Training set
  - Validation set (created from training set)
  - Test set

---

## 🧰 Libraries Used

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from skimage.feature import hog, local_binary_pattern
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
