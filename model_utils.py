from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

def train_model(x_train, y_train):
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(x_train, y_train)
    return model

def evaluate_model(model, x_test, y_test):
    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions) * 100
    return accuracy