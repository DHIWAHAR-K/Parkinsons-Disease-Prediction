# Parkinsons Disease Prediction

This project uses an XGBoost classifier to predict the presence of Parkinson's disease based on biomedical voice measurements.

## Project Structure:

- `data/`: Contains the dataset
- `utils/`: Contains helper scripts for data processing and model building
- `main.py`: Main script to run the pipeline

## Model:

- Model: XGBoost Classifier
- Preprocessing: Min-Max Scaling (range: -1 to 1)
- Evaluation: Accuracy Score

## How to Run:

```bash
pip install -r requirements.txt
```

```bash
python main.py
```


## Dataset:

Dataset Source: [UCI Parkinson’s Disease Dataset](https://www.kaggle.com/datasets/vikasukani/parkinsons-disease-data-set)


## License:

This project is intended for educational and research purposes. Feel free to use, modify, and build upon this work with credit.