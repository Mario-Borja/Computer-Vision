# Load and preprocess the dataset
data = pd.read_csv('food_dataset.csv')

# Extract the features and labels
X = data.drop(['id', 'name', 'calories'], axis=1)
y = data['calories']

# Normalize the features
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Select the best machine learning model
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor(),
    'XGBoost': XGBRegressor()
}

# Evaluate the performance of each model
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-scores)
    print(f'{name}: Mean RMSE: {rmse_scores.mean()}, Standard deviation: {rmse_scores.std()}')

# Tune the hyperparameters of the selected model
model = XGBRegressor()

param_grid = {
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'n_estimators': [50, 100, 200]
}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

# Evaluate the performance of the final model
y_pred = best_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f'RMSE: {rmse}, R^2: {r2}')

# Load the image and extract its features
image = load_image('new_image.jpg')
features = extract_features(image)

# Scale the features
features = scaler.transform(features.reshape(1, -1))

# Use the trained model to predict the calorie count
calories = best_model.predict(features)
print(f'The estimated calorie count is: {calories[0]}')

# Helper function to load an image
def load_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    return img

# Helper function to extract features from an image
def extract_features(image):
    # Load the pre-trained VGG16 model
    model = VGG16(weights='imagenet', include_top=False)

    # Preprocess the image
    image = preprocess_input(image)

    # Extract the features
    features = model.predict(image)
    features = features.flatten()

    return features

