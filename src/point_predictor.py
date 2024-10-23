import pandas as pd
import numpy as np
import keras_tuner as kt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import r2_score
from tensorflow import keras
from tensorflow.keras import layers
import joblib

# 1. Data Preparation Functions

def load_player_data(data_dir):
    """
    Loads and combines CSV files for a specific player.
    """
    df_list = [pd.read_csv(file) for file in data_dir]
    player_df = pd.concat(df_list, ignore_index=True)
    return player_df

def create_sequences(df, sequence_length, target_column):
    """
    Transforms the DataFrame into sequences suitable for RNN input.
    """
    X = []
    y = []
    for i in range(len(df) - sequence_length):
        X_sequence = df.iloc[i:i+sequence_length].drop(columns=[target_column]).values
        y_value = df.iloc[i+sequence_length][target_column]
        X.append(X_sequence)
        y.append(y_value)
    return np.array(X), np.array(y)

def preprocess_data(df, sequence_length, target_column):
    """
    Preprocesses the data by scaling features and creating sequences.
    Returns the scaled data and the fitted scalers.
    """
    df_model = df.select_dtypes(include=[np.number])

    # Initialize scalers
    scaler = StandardScaler()
    scaler_target = StandardScaler()

    # Scale features (everything except the target column)
    features = df_model.drop(columns=[target_column])  # Drop target column 'PTS'
    features_scaled = scaler.fit_transform(features)

    # Scale target column (PTS)
    df_scaled = pd.DataFrame(features_scaled, columns=features.columns)
    df_scaled[target_column] = scaler_target.fit_transform(df_model[[target_column]])

    # Create sequences for RNN
    X, y = create_sequences(df_scaled, sequence_length, target_column)

    # Save scalers for later use
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(scaler_target, 'scaler_target.pkl')

    return X, y, scaler, scaler_target

# 2. Model Building Function

def build_rnn_model(hp):
    """
    Builds and compiles the RNN model with hyperparameter tuning.
    """
    model = keras.Sequential([
        layers.Masking(mask_value=0., input_shape=(X.shape[1], X.shape[2])),
        layers.Bidirectional(layers.LSTM(hp.Int('units', min_value=64, max_value=256, step=64), return_sequences=True)),
        layers.LSTM(hp.Int('units2', min_value=64, max_value=256, step=64)),
        layers.Dropout(0.2),
        layers.Dense(hp.Int('dense_units', min_value=32, max_value=128, step=32), activation='relu'),
        layers.Dense(1)  # Output layer for points prediction
    ])
    
    optimizer = keras.optimizers.Adam(learning_rate=hp.Choice('learning_rate', [1e-3, 1e-4, 1e-5]))
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
    return model

# 3. Cross-Validation and Evaluation Function

def train_and_evaluate_model_with_cv(X, y, n_splits=5):
    """
    Performs K-fold cross-validation to evaluate model performance.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    mse_scores = []
    mae_scores = []
    r2_scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = build_rnn_model(kt.HyperParameters())  # Use default hyperparameters for CV
        model.fit(X_train, y_train, epochs=100, batch_size=16, verbose=1, 
                  validation_split=0.1, callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)])

        # Evaluate the model
        mse, mae = model.evaluate(X_test, y_test, verbose=1)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)

        mse_scores.append(mse)
        mae_scores.append(mae)
        r2_scores.append(r2)

    avg_mse = np.mean(mse_scores)
    avg_mae = np.mean(mae_scores)
    avg_r2 = np.mean(r2_scores)
    print(f'Average MSE from Cross-Validation: {avg_mse}, Average MAE: {avg_mae}, Average R2: {avg_r2}')
    return avg_mse, avg_mae, avg_r2, model

# 4. Save Model Function

def save_model(model, scaler, scaler_target, model_path='points_model.h5'):
    """
    Saves the trained model and scalers to disk.
    """
    model.save(model_path)
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(scaler_target, 'scaler_target.pkl')
    print(f'Model saved to {model_path}')
    print(f'Scalers saved to scaler.pkl and scaler_target.pkl')

# Main Execution

if __name__ == '__main__':
    # Parameters
    data_dir = ['Data\\2022\\Lebron_James_2022.csv', 'Data\\2023\\Lebron_James_2023.csv', 'Data\\2024\\Lebron_James_2024.csv']  # List of data files
    sequence_length = 20  # Sequence length
    target_column = 'PTS'  # Target column

    # Load and preprocess data
    df_player = load_player_data(data_dir)
    X, y, scaler, scaler_target = preprocess_data(df_player, sequence_length, target_column)

    # Split data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize tuner
    tuner = kt.Hyperband(
        build_rnn_model,
        objective='val_loss',
        max_epochs=50,
        factor=3,
        directory='src',  # Save tuning logs here
        project_name='nba_player_model'
    )

    # Search for best hyperparameters
    tuner.search(X_train, y_train, epochs=50, validation_data=(X_val, y_val), callbacks=[keras.callbacks.EarlyStopping(patience=5)])

    # Get the best model
    best_model = tuner.get_best_models(num_models=1)[0]

    # Optionally, further evaluate the model with cross-validation
    avg_mse, avg_mae, avg_r2, model = train_and_evaluate_model_with_cv(X, y)

    # Save the best model and scalers
    save_model(best_model, scaler, scaler_target)

    # Optional: Make predictions on the validation data
    y_pred = best_model.predict(X_val)
    y_pred = scaler_target.inverse_transform(y_pred)  # Inverse transform the predicted points
    print('Sample Predictions:')
    for i in range(5):
        print(f'Predicted: {y_pred[i][0]:.2f}, Actual: {scaler_target.inverse_transform([[y_val[i]]])[0][0]:.2f}')
