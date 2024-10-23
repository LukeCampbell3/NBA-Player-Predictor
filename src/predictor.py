from tensorflow.keras.models import load_model
import joblib
import numpy as np

# Load the model and scaler
model = load_model('points_model.h5')
scaler = joblib.load('scaler.pkl')  # Load the feature scaler, not the target scaler

# Prepare new data (ensure it's preprocessed in the same way as during training)
# Replace this with your actual new data in the correct shape: (1, sequence_length, num_features)
new_sequence = np.array([[
    [9.0,15.0,23.0,0.652,3.0,5.0,0.6,7.0,8.0,0.875,4.0,0.754,35.7,0.717,140.0,131.0,36.7,16.4,115.2], # 40.0,
    [10.0,10.0,14.0,0.714,2.0,5.0,0.4,3.0,3.0,1.0,2.0,0.816,27.2,0.786,153.0,106.0,26.3,13.3,119.4], # 25.0,
    [6.0,7.0,15.0,0.467,1.0,3.0,0.333,5.0,7.0,0.714,8.0,0.553,30.5,0.5,87.0,97.0,10.4,-3.2,113.8],  # 20.0,
    [10.0,8.0,19.0,0.421,2.0,4.0,0.5,8.0,8.0,1.0,3.0,0.577,27.2,0.474,124.0,132.0,23.1,0.2,118.0],  # 26.0,
    [12.0,8.0,14.0,0.571,0.0,1.0,0.0,7.0,11.0,0.636,4.0,0.61,28.3,0.571,125.0,111.0,24.0,3.2,114.3],    # 23.0,
    [8.0,6.0,12.0,0.5,0.0,3.0,0.0,4.0,4.0,1.0,5.0,0.581,23.6,0.5,98.0,113.0,13.6,1.5,118.0]     # 16.0,
]])  # Shape: (1, 5, 5) --> 1 sample, 5 time steps, 5 features

# Scale the new sequence data using the feature scaler (used during training)
new_sequence_scaled = scaler.transform(new_sequence.reshape(-1, new_sequence.shape[2])).reshape(new_sequence.shape)

# Make prediction
prediction_scaled = model.predict(new_sequence_scaled)

# Load target scaler to inverse transform the prediction back to original scale (PTS)
scaler_target = joblib.load('scaler_target.pkl')
prediction = scaler_target.inverse_transform(prediction_scaled)

# Output the predicted points
print(f'Predicted points: {prediction[0][0]:.2f}')
