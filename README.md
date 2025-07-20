# 💼 Salary Predictor Web Application

An interactive, AI-powered salary prediction app built using **Streamlit** and **Scikit-learn**, designed to estimate salaries based on key professional and educational attributes.

---

## 📊 Overview

This project predicts salaries using a **trained machine learning model** (RandomForestRegressor) and a custom pipeline that includes:

- **Job Title Grouping**
- **Feature Engineering**
- **Education Encoding**
- **Categorical + Numerical Preprocessing**

The model is trained on Indian salary data and provides outputs in **INR**.

---

## 🚀 Features

- 🎯 Predict salaries with real-time input
- 📈 Display prediction breakdowns (like base, bonus, tax)
- 🧠 Uses custom transformer modules
- 🖥️ Fully interactive frontend via Streamlit
- 🔍 Displays model evaluation metrics and comparison
- 💾 Model: `salary_predictor_corrected.pkl`

---

## 📁 Project Structure
📦 salary_prediction_app/
│
├── Data/
│ └── Salary Data.csv
│
├── model/
│ └── salary_predictor_corrected.pkl
│
├── transformers/
│ ├── init.py
│ ├── education_encoder.py
│ ├── feature_engineer.py
│ └── job_grouper.py
│
├── streamlit_app.py # Main UI app
├── train_model.py # Model training + pipeline
├── requirements.txt
├── README.md # This file
├── TIPS.md # Optional: Usage tips
└── model_evaluation_table.xlsx

## 🧠 Input Features
├──Job Title
│
├──Years of Experience
│
├──Education Level
│
├──Age
│
├──Gender
│
├──Company Size

## 🔐 Model Details

Model Used: Random Forest Regressor
Framework: Scikit-learn
Encoding: Custom Transformer modules
Data Format: INR (Indian Rupees)

## 📌 Future Enhancements
💱 Add currency conversion (USD, EUR)
🌐 Deploy to HuggingFace or Streamlit Cloud
🧑‍💻 Add user login + history tracking

## 👨‍💻 Developer
Praneeth
LinkedIn: www.linkedin.com/in/praneethlakhineni
GitHub: [github.com/praneeth](https://github.com/PraneethL)




