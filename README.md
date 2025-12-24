Submitted by: Anzuman Ara
Student ID: 20241266
Module: Introduction to Data Handling, Exploration & Applied Machine Learning
Programme: MSc Business Data Science
# Trash Wheel — Monthly Trash Weight Prediction
Business Goal
The purpose of this project is to predict the amount of trash collected each month by the Mr. Trash Wheel system. The dataset comes from the TidyTuesday project (Week 10, 2024). Being able to forecast these values can help organizations plan resources and understand seasonal trends.

Project Steps:
1. Data Cleaning & Feature Engineering
I cleaned the dataset and created lag features so the model could use the previous month’s trash weight as context. This seemed important because waste collection shows seasonal and temporal patterns.

2. Exploratory Data Analysis
I generated three main visualizations to understand trends, detect anomalies, and observe potential correlations. This helped decide which features were worth keeping.

3. Train/Test Split (time-aware)
Since the data is time series, I avoided random splitting and instead kept the last part of the timeline as the test set.

4. Modeling
I compared a simple Linear Regression baseline with a Random Forest model. The Random Forest clearly performed better, especially in capturing nonlinear patterns.

5. Evaluation
I evaluated both models using RMSE, MAE, and R² to get a full picture of both accuracy and error magnitude.

6. Explainability
I included a SHAP summary plot to see how each feature affects the predictions. This made it easier to interpret the Random Forest results.

7. Streamlit App
Finally, I built a Streamlit interface so the model can be used interactively. The app allows the user to enter month/year values and experiment with a "what-if" slider for the lagged feature.

Reproducibility
I used the official dataset URL, set all random seeds, and added a requirements.txt file to make the project runnable on another machine.

