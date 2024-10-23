from keras.models import load_model

# pip install joblib
import joblib

def predict_stat_value(def_rating, model_path='stat_prediction_model.h5', scaler_X_path='scaler_X.pkl', scaler_y_path='scaler_y.pkl'):
    model = load_model(model_path)
    # Load scalers
    scaler_X = joblib.load(scaler_X_path)
    scaler_y = joblib.load(scaler_y_path)

    # Scale the input
    def_rating_scaled = scaler_X.transform([[def_rating]])

    # Make a prediction
    predicted_stat_scaled = model.predict(def_rating_scaled)

    # Inverse scale the prediction
    predicted_stat = scaler_y.inverse_transform(predicted_stat_scaled)

    return predicted_stat[0][0]

# Example usage
if __name__ == "__main__":
    example_def_rating = 104.5
    predicted_stat = predict_stat_value(example_def_rating)
    print(f'Predicted Stat Value: {predicted_stat}')
