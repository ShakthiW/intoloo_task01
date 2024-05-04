# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix
import numpy as np


# Step 1: Load the dataset
data = pd.read_csv("Crop_Dataset.csv")

# Display the first few rows of the dataset to understand its structure
print(data.head())

# Step 2: Handle missing values (if any)
# Check for missing values in the dataset
print(data.isnull().sum())

# Since there are no missing values, we can proceed without any imputation

# Step 3: Encode categorical variables (if any)
# Check if there are any categorical variables that need encoding
print(data.dtypes)

# There are no categorical variables that need encoding in this dataset

# Step 4: Scale numerical features
# Separate features and labels
X = data.drop(['Label', 'Label_Encoded'], axis=1)
y = data['Label_Encoded']

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Visualize numerical features
numerical_features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'Total_Nutrients', 'Temperature_Humidity', 'Log_Rainfall']

# fig, ax = plt.subplots(nrows=5, ncols=2, figsize=(15, 20))
# for i, feature in enumerate(numerical_features):
#     sns.histplot(data=data, x=feature, ax=ax[i//2, i%2], kde=True)
#     ax[i//2, i%2].set_title(f'Distribution of {feature}')
# plt.tight_layout()
# plt.show()

# # Visualize correlation between features
# plt.figure(figsize=(12, 8))
# sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
# plt.title('Correlation Heatmap')
# plt.show()

# # Visualize target variable distribution
# plt.figure(figsize=(6, 4))
# sns.countplot(data=data, x='Label')
# plt.title('Distribution of Crop Labels')
# plt.show()


# Now, we have preprocessed the data. We can proceed to model training.

# Step 6: Model Training
# Choose a machine learning algorithm (Random Forest Classifier)
model = RandomForestClassifier(random_state=42)

# Train the model using the training data
model.fit(X_train, y_train)

# Step 7: Model Evaluation
# Evaluate the trained model's accuracy using the testing dataset
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the model: {accuracy}")

# Provide insights into model performance
# For classification tasks, accuracy is one of the metrics to evaluate the model's performance.
# Additionally, you can explore other metrics like precision, recall, and F1-score for each class.
# Compute precision, recall, and F1-score
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Compute confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# numerical_features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'Total_Nutrients', 'Temperature_Humidity', 'Log_Rainfall']

# Plot feature importances
feature_importances = model.feature_importances_
sorted_indices = np.argsort(feature_importances)[::-1]
sorted_features = [numerical_features[i] for i in sorted_indices]

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances[sorted_indices], y=sorted_features, palette='viridis')
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Feature Importances')
plt.show()


# Now, let's proceed to save the trained model for future use.

# Step 8: Save the trained model
joblib.dump(model, 'crop_recommendation_model.joblib')
print("Model saved successfully!")
