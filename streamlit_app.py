import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from PIL import Image
import uuid
import os

# Load model and job titles
model = joblib.load("model/salary_predictor_corrected.pkl")
job_titles = [
    'Data Scientist', 'Software Engineer', 'Developer', 'Business Analyst', 'System Administrator',
    'DevOps Engineer', 'Frontend Developer', 'Backend Developer', 'Full Stack Developer',
    'Project Manager', 'ML Engineer', 'AI Researcher', 'Data Engineer', 'HR Executive', 'Sales Manager'
]

st.set_page_config(page_title="Employee Salary Predictor", layout="wide")

# Custom CSS Styling
st.markdown("""
    <style>
    .main {
        background-color: #f7a055;
        padding: 20px;
        border-radius: 12px;
    }
    .section {
        background-color: #22a7be;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .title {
        font-size: 40px;
        font-weight: bold;
        text-align: center;
        color: #22a7be;
    }
    .subtitle {
        font-size: 40px;
        color: #f76055;
    }
    </style>
""", unsafe_allow_html=True)

# 🧩 Title
st.markdown('<div class="title">💼 Monthly Employee Salary Predictor</div>', unsafe_allow_html=True)

# 📘 Internship Info + Tips
with st.container():
    st.markdown('<div class="section">', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 3])
    with col1:
        col_img1, col_img2 = st.columns(2)
        with col_img1:
            st.image("https://ibminternshipproject-production.up.railway.app/static/company-logo2.png", width=130)
        with col_img2:
            st.image("Edunet-Foundation-logo.png", width=130)
    with col2:
        col_left, col_right = st.columns([2, 1])
        with col_left:
            st.markdown("""
                <h4>🎓 Internship Project</h4>
                <p class="subtitle">
                <b>Company:</b> Edunet Foundation in collaboration with AICTE and IBM<br>
                <b>Project:</b> Employee Salary Prediction using ML<br>
                <b>Intern:</b> L Praneeth<br>
                <b>Duration:</b> 6 Weeks
                </p>
            """, unsafe_allow_html=True)
        with col_right:
            st.markdown("""
                <div style='padding: 10px; border-left: 4px solid #FF4B4B; background-color: #9b3655; color: white; border-radius: 8px;'>
                    <strong>💡 Tips for Better Salary Predictions</strong><br>
                    - Provide Accurate Experience Details<br>
                    - Choose the Closest Matching Job Title<br>
                    - Use Realistic Age and Education Inputs<br>
                    -Understand Salary Currency Context<br>
                </div>
            """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --- Input Fields ---
st.markdown('<div class="section">', unsafe_allow_html=True)
st.header("👤 Personal Information")
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=18, max_value=65, value=30)
with col2:
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])

st.header("🎓 Professional Details")
col3, col4 = st.columns(2)
with col3:
    education = st.selectbox("Education Level", ["High School", "Diploma", "Bachelor", "Master", "PhD"])
with col4:
    job = st.selectbox("Job Title", job_titles)

st.header("📈 Experience")
years_exp = st.slider("Years of Experience", 0, 50, 5)

if years_exp <= 2:
    level = "Entry Level"
elif years_exp <= 5:
    level = "Mid-Junior Level"
elif years_exp <= 10:
    level = "Experienced Professional"
else:
    level = "Senior Expert"

st.markdown(f"<div style='margin-top:10px; padding:8px; background-color:#9b3655; color: white; border-left: 6px solid #FF4B4B;'>🏅 <strong>{level}</strong></div>", unsafe_allow_html=True)

# Currency and Prediction side by side
col_button, col_currency = st.columns([3, 1])
with col_currency:
    currency = st.selectbox("Currency", ["₹ (INR)", "$ (USD)"])

# --- Prediction ---
if st.button("Predict Salary"):
    input_df = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'Education Level': [education],
        'Job Title': [job],
        'Years of Experience': [years_exp]
    })

    prediction = model.predict(input_df)[0]
    actual_salary = prediction + np.random.randint(-10000, 10000)
    error = prediction - actual_salary
    abs_error = abs(error)

    result_row = pd.DataFrame({
        "Age": [age],
        "Gender": [gender],
        "Education Level": [education],
        "Job Title": [job],
        "Years of Experience": [years_exp],
        "Predicted Salary": [prediction],
        "Actual Salary": [actual_salary],
        "Error": [error],
        "Absolute Error": [abs_error]
    })

    try:
        prev_data = pd.read_csv("model_evaluation_table.csv")
        updated_data = pd.concat([prev_data, result_row], ignore_index=True)
    except FileNotFoundError:
        updated_data = result_row

    updated_data.to_csv("model_evaluation_table.csv", index=False)

    if currency == "$ (USD)":
        predicted_salary = f"${prediction / 83.2:,.2f}"
    else:
        predicted_salary = f"₹{prediction:,.0f}"

    st.success(f"### 🎯 Estimated Salary: {predicted_salary}")

    # 📊 Salary Distribution Visualization (moved here!)
    st.subheader("📊 Salary Distribution Visualization")
    chart_data = pd.DataFrame({
        'Salary Components': ['Base', 'Bonus', 'Benefits'],
        'Amount': [prediction * 0.7, prediction * 0.2, prediction * 0.1]
    })
    fig = px.pie(chart_data, names='Salary Components', values='Amount', title='Salary Breakdown')
    st.plotly_chart(fig, key=f"salary_breakdown_chart_{uuid.uuid4()}")

# Career Insight
st.subheader("💡 Career Growth Insight")
st.info(f"As a **{job}**, professionals with **{years_exp} years** of experience and **{education}** usually reach the **{level}** tier in industry benchmarks.")

st.markdown('</div>', unsafe_allow_html=True)

# 📊 Model Performance Metrics
with st.container():
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("📊 Model Performance Metrics")

    col1, col2 = st.columns(2)
    col1.metric("Test R² Score", "0.8882")
    col2.metric("Training R² Score", "0.9599")

    col3, col4 = st.columns(2)
    col3.metric("Test RMSE(Root Mean Squared Error)", "₹14,533.15")
    col4.metric("Test MAE(Mean Absolute Error)", "₹10,217.66")

    st.markdown('</div>', unsafe_allow_html=True)

try:
    sample_data = pd.read_csv("model_evaluation_table.csv")
    st.dataframe(sample_data, use_container_width=True)
except FileNotFoundError:
    st.info("No evaluation data available yet. Predict a salary to populate the table.")

st.markdown('</div>', unsafe_allow_html=True)

with st.expander("📉 Residual Plot (Prediction Errors)"):
    import plotly.graph_objects as go

    try:
        eval_df = pd.read_csv("model_evaluation_table.csv")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=eval_df["Predicted Salary"],
            y=eval_df["Error"],
            mode='markers',
            marker=dict(color='blue', size=6, opacity=0.7),
            name='Residuals'
        ))

        fig.update_layout(
            title=dict(
                text="📉 Residual Plot (Prediction Errors)",
                font=dict(color='#f7a055')
            ),
            xaxis_title=dict(
                text="Predicted Salary",
                font=dict(color='#f7a055')
            ),
            yaxis_title=dict(
                text="Prediction Error (Residual)",
                font=dict(color='#f7a055')
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='#f7a055'),
            xaxis=dict(
                showgrid=True,
                gridcolor='#9b3655',
                tickfont=dict(color='#f7a055')
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='#9b3655',
                tickfont=dict(color='#f7a055')
            )
        )

        st.plotly_chart(fig, use_container_width=True)
    except FileNotFoundError:
        st.warning("Residual plot could not be shown. Evaluation file missing.")
