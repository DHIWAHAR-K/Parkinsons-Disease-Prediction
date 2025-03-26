import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def load_data(path):
    df = pd.read_csv(path)
    features = df.loc[:, df.columns != 'status'].values[:, 1:]
    labels = df.loc[:, 'status'].values
    return features, labels

def preprocess_data(features, labels, test_size=0.15):
    scaler = MinMaxScaler((-1, 1))
    x_scaled = scaler.fit_transform(features)
    x_train, x_test, y_train, y_test = train_test_split(x_scaled, labels, test_size=test_size, random_state=42)
    return x_train, x_test, y_train, y_test