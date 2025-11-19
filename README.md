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

├── data/

│   └── AUS_weather.csv

├── notebooks/

│   └── EDA_and_Model.ipynb

├── src/

│   └── logistic_regression.ipynb

└── README.md

### How to Run

- Install dependencies:

pip install -r requirements.txt


- Run the model script:

python src/logistic_regression.py

- Output

The script logs model performance metrics and generates plots for:

- Feature distributions

Correlation matrix

ROC curve

### Purpose

This repository is intended for learning and demonstrating end-to-end logistic regression modeling on a real-world dataset.
