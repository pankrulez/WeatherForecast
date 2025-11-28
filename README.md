### Logistic Regression on Australian Weather Dataset

This project implements logistic regression to predict rainfall using the Australian Weather dataset. The workflow covers data loading, cleaning, exploratory analysis, feature engineering, model training, and evaluation.

### Dataset

The dataset includes daily weather observations from multiple Australian locations. Key variables include temperature, humidity, wind measurements, atmospheric pressure, and rainfall indicators.

### Model

A logistic regression model is used to predict the binary target variable:
RainTomorrow — whether it will rain the next day.

### Steps:

- Handle missing values

- Encode categorical features

- Scale numeric features

- Split data into training and test sets

- Train logistic regression

- Evaluate accuracy, precision, recall, F1-score, and ROC-AUC

### Project Structure

```
WeatherForecast/
├── data/
│   └── raw/
│       └── AUS_Weather.csv
├── notebooks/
│   ├── exploration.ipynb
│   └── exploration2.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── features.py
│   ├── train.py
│   └── predict.py
├── models/
│   └── weather_logreg.joblib       
├── app.py                          
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE
```

### How to Run

- Install dependencies:
```
pip install -r requirements.txt
```

- Run the model script:
```
python src/exploration.ipynb
```
- Output

The script logs model performance metrics and generates plots for:

- Feature distributions

Correlation matrix

ROC curve

### Purpose

This repository is intended for learning and demonstrating end-to-end logistic regression modeling on a real-world dataset.
