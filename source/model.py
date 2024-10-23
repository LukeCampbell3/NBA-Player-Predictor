from keras.models import Sequential
from keras.layers import Dense
from data_prep import load_and_prepare_data

def build_and_train_model(csv_file):
    X_train, X_test, y_train, y_test, scaler_X, scaler_y = load_and_prepare_data(csv_file)

    model = Sequential([
        Dense(64, activation='relu', input_dim=1),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

    return model, scaler_X, scaler_y

def save_model_and_scalers(model, scaler_X, scaler_y):
    # Save your model and scalers for later use
    model.save('stat_prediction_model.h5')
    # Save scalers using joblib or pickle
    # Example: joblib.dump(scaler_X, 'scaler_X.pkl')
    # Example: joblib.dump(scaler_y, 'scaler_y.pkl')

# Example usage
if __name__ == "__main__":
    csv_file = 'your_data.csv'
    model, scaler_X, scaler_y = build_and_train_model(csv_file)
    save_model_and_scalers(model, scaler_X, scaler_y)
