#pip install pandas
import pandas as pd

# pip install keras scikit-learn
# do this once and keras and sklearn will be downloaded
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def load_and_prepare_data(csv_file):
    df = pd.read_csv(csv_file)

    X = df[['DefRating']]  # Features
    y = df['StatValue']  # Target

    # Normalize the features and target
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, scaler_X, scaler_y
