# Australian Weather Rain Prediction 🌧️

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Status](https://img.shields.io/badge/Status-Production%20Ready-success)
![License](https://img.shields.io/badge/License-Open%20Source-brightgreen)

This project builds an end-to-end **machine learning pipeline to predict whether it will rain tomorrow in Australia**, using historical weather data. The project includes:

- Data cleaning and preprocessing
- Feature engineering
- Model training with scikit-learn
- Model evaluation
- An interactive **Streamlit web application** for:
  - Data inspection & cleaning preview
  - Real-time rain prediction

---

## 🚀 Features

- Modular project structure (no notebook dependency for training)
- Automated data cleaning
- Missing value handling using imputers
- Logistic Regression classification model
- Streamlit-based UI with:
  - Data Cleaning & EDA page
  - Rain Prediction page
- Reproducible training pipeline
- Production-ready model saving & loading

---

## 📂 Project Structure

<<<<<<< HEAD
```text
=======
```
>>>>>>> abfe4776b492a546d64883fb560c132ed54befe9
WeatherForecast/
├── data/
│   └── raw/
│       └── AUS_Weather.csv
<<<<<<< HEAD
├── models/
│   └── weather_logreg.joblib
├── notebooks/
│   └── exploration.ipynb
=======
├── notebooks/
│   ├── exploration.ipynb
│   └── exploration2.ipynb
>>>>>>> abfe4776b492a546d64883fb560c132ed54befe9
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── features.py
│   ├── train.py
│   └── predict.py
<<<<<<< HEAD
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## 🧠 Model Details
Algorithm: Logistic Regression
=======
├── models/
│   └── weather_logreg.joblib       
├── app.py                          
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE
```
>>>>>>> abfe4776b492a546d64883fb560c132ed54befe9

Target Variable: RainTomorrow

<<<<<<< HEAD
Selected Features:

MinTemp

MaxTemp

Humidity3pm

Pressure3pm

WindSpeed3pm

RainToday

Preprocessing:

Numerical features → Median imputation + StandardScaler

Categorical features → Mode imputation + OneHotEncoder

### ⚙️ How to Run
1️⃣ Install dependencies
```
pip install -r requirements.txt
```
2️⃣ Train the model
```
python -m src.train
```
This will generate:

models/weather_logreg.joblib

3️⃣ Run the Streamlit app
```
streamlit run app.py
```
### 🖥️ Application Pages
✅ Data Cleaning & EDA

Displays raw dataset

Shows missing values
=======
- Install dependencies:
```
pip install -r requirements.txt
```

- Run the model script:
```
python src/exploration.ipynb
```
- Output
>>>>>>> abfe4776b492a546d64883fb560c132ed54befe9

Displays cleaned dataset preview

✅ Prediction Page

User inputs weather conditions

Outputs:

Rain / No Rain

Prediction probability

### 📈 Future Enhancements
Batch predictions via CSV upload

Advanced models (XGBoost, Random Forest)

Model performance visualizations

Deployment on Streamlit Cloud

### 📜 License
This project is open-source and free to use for learning and experimentation.