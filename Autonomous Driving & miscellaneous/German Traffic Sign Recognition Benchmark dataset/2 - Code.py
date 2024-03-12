# Section 1: Data Preprocessing

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Load the dataset
train_data = pd.read_csv('Train.csv')
test_data = pd.read_csv('Test.csv')

# Preprocess the images
train_images = []
train_labels = []

for i in range(len(train_data)):
    img = cv2.imread('Train/' + train_data['Path'][i])
    img = cv2.resize(img, (32, 32))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype('float32') / 255.0
    train_images.append(img)
    train_labels.append(train_data['ClassId'][i])
    
test_images = []
test_labels = []

for i in range(len(test_data)):
    img = cv2.imread('Test/' + test_data['Path'][i])
    img = cv2.resize(img, (32, 32))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype('float32') / 255.0
    test_images.append(img)
    test_labels.append(test_data['ClassId'][i])
    
train_images = np.array(train_images)
train_labels = np.array(train_labels)
test_images = np.array(test_images)
test_labels = np.array(test_labels)

# Section 2: Feature Extraction

from skimage.feature import hog

# Extract HOG features from the images
train_hog = []

for i in range(len(train_images)):
    hog_feature = hog(train_images[i], orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
    train_hog.append(hog_feature)
    
test_hog = []

for i in range(len(test_images)):
    hog_feature = hog(test_images[i], orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
    test_hog.append(hog_feature)
    
train_hog = np.array(train_hog)
test_hog = np.array(test_hog)

# Section 3: Model Selection and Training

from sklearn.svm import SVC

# Train an SVM classifier on the HOG features
model = SVC(kernel='rbf', C=10, gamma=0.1)
model.fit(train_hog, train_labels)

# Section 4: Hyperparameter Tuning

from sklearn.model_selection import GridSearchCV

# Tune the hyperparameters of the SVM using grid search
param_grid = {'C': [1, 10, 100], 'gamma': [0.1, 0.01, 0.001]}
grid_search = GridSearchCV(model, param_grid, cv=3)
grid_search.fit(train_hog, train_labels)
best_params = grid_search.best_params_

# Retrain the SVM with the best hyperparameters
model = SVC(kernel='rbf', C=best_params['C'], gamma=best_params['gamma'])
model.fit(train_hog, train_labels)

# Section 5: Model Evaluation

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Section 5: Model Evaluation (continued)

from sklearn.metrics import confusion_matrix, classification_report

# Print the confusion matrix and classification report
conf_mat = confusion_matrix(test_labels, test_pred)
class_report = classification_report(test_labels, test_pred)
print('Confusion Matrix:\n', conf_mat)
print('Classification Report:\n', class_report)

# Visualize the performance of the SVM using a heatmap
import seaborn as sns

sns.heatmap(conf_mat, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.title('Confusion Matrix')
plt.show()

# Section 6: Model Visualization

# Visualize the HOG features of a sample image
sample_index = 100
sample_img = test_images[sample_index]
sample_hog = test_hog[sample_index]

fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].imshow(sample_img, cmap='gray')
ax[0].set_title('Sample Image')
ax[1].imshow(sample_hog, cmap='gray')
ax[1].set_title('HOG Features')
plt.show()

# Section 7: Deployment

# Save the trained SVM model for deployment
import joblib

joblib.dump(model, 'svm_model.pkl')
