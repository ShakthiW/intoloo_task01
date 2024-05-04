import numpy as np
import joblib

# Load the saved model
model = joblib.load('crop_recommendation_model.joblib')

# Define a function to make predictions
def predict_crop(environmental_conditions):
    """
    Function to predict the crop based on environmental conditions.
    Args:
    - environmental_conditions (list or numpy array): Environmental conditions including
                                                      N, P, K, temperature, humidity, pH, rainfall.
    Returns:
    - prediction (str): Predicted crop.
    """
    # Reshape the input array for prediction
    environmental_conditions = np.array(environmental_conditions).reshape(1, -1)
    # Use the loaded model to make predictions
    prediction = model.predict(environmental_conditions)
    return prediction

# Test the model with some example environmental conditions
example_environment = [90,42,43,20.87974371,82.00274423,6.502985292,202.9355362,175,1712.196283,5.317803945]

predicted_crop = predict_crop(example_environment)
print("Predicted Crop:", predicted_crop)
