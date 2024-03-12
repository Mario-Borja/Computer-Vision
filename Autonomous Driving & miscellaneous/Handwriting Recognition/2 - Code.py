# Load the dataset
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1)

# Split into features and labels
X = mnist['data']
y = mnist['target']

# Split into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a support vector machine (SVM) classifier
from sklearn.svm import SVC
svm_clf = SVC(kernel='poly', degree=3, gamma='scale')
svm_clf.fit(X_train, y_train)

# Evaluate the performance of the model on the test set
from sklearn.metrics import accuracy_score
y_pred = svm_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")


# Load the dataset
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1)

# Split into features and labels
X = mnist['data']
y = mnist['target']

# Split into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a support vector machine (SVM) classifier
from sklearn.svm import SVC
svm_clf = SVC(kernel='poly', degree=3, gamma='scale')
svm_clf.fit(X_train, y_train)

# Evaluate the performance of the model on the test set
from sklearn.metrics import accuracy_score
y_pred = svm_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

