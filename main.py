from utils.data_loader import load_data, preprocess_data
from utils.model_utils import train_model, evaluate_model

def main():
    path = 'data/parkinsons.data'
    features, labels = load_data(path)
    x_train, x_test, y_train, y_test = preprocess_data(features, labels)
    
    model = train_model(x_train, y_train)
    accuracy = evaluate_model(model, x_test, y_test)
    
    print(f"Accuracy Score is: {accuracy:.2f}%")

if __name__ == "__main__":
    main()