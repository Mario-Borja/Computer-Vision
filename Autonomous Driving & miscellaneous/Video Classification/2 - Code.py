# import required libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, precision_recall_fscore_support

# load the dataset
df = pd.read_csv('YouTube8M.csv')

# split the dataset into features and labels
X = df.iloc[:, 1:-4800].values
y = df.iloc[:, -4800:].values

# split the dataset into training, validation, and testing sets
X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=42)

# normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# build the model
model = keras.Sequential([
    keras.layers.Dense(1024, activation='relu', input_shape=(2048,)),
    keras.layers.Dense(4800, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# train the model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=64)

# evaluate the model on the validation set
y_pred = model.predict(X_val)
mAP = average_precision_score(y_val, y_pred, average='macro')
print('Validation mAP:', mAP)

# fine-tune the model
model.fit(X_val, y_val, epochs=10, batch_size=64)

# evaluate the model on the test set
y_pred = model.predict(X_test)
mAP = average_precision_score(y_test, y_pred, average='macro')
print('Test mAP:', mAP)

# analyze the performance of the model on individual categories
precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred > 0.5)
print('Category Precision:', precision)
print('Category Recall:', recall)
print('Category F1-score:', f1)
