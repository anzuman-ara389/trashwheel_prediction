import streamlit as st
import pandas as pd
import joblib, json
import numpy as np
import matplotlib.pyplot as plt

# Basic page setup
st.set_page_config(page_title="Trash Wheel Predictor", layout="centered")
st.title("Trash Wheel — Monthly Weight Prediction")

# Load model & metadata
model = joblib.load("model.joblib")
meta  = json.load(open("model_meta.json"))
features = meta["features"] 

st.caption("Model: Random Forest • Features: " + ", ".join(features))

# Sidebar inputs 
st.sidebar.header("Enter feature values")

input_data = {}

for f in features:
    # Month
    if f.lower() == "month":
        input_data[f] = st.sidebar.slider("Month (1–12)", 1, 12, 6)

    # Year
    elif f.lower() == "year":
        input_data[f] = st.sidebar.number_input("Year", 2014, 2035, 2024, step=1)

    # Lag features (e.g. weight_tons_lag1, weight_tons_lag2, weight_tons_lag3)
    elif "lag" in f.lower():
        input_data[f] = st.sidebar.number_input(
            f"{f} (tons)",
            min_value=0.0,
            max_value=50.0,
            value=5.0,
            step=0.5
        )

    # Rain / precipitation type features
    elif "rain" in f.lower() or "precip" in f.lower():
        input_data[f] = st.sidebar.number_input(
            f"{f} (mm)",
            min_value=0.0,
            max_value=500.0,
            value=50.0,
            step=1.0
        )

    # Default numeric feature
    else:
        input_data[f] = st.sidebar.number_input(
            f,
            min_value=0.0,
            max_value=1000.0,
            value=0.0,
            step=1.0
        )

# Build dataframe in the correct feature order
X_base = pd.DataFrame([[input_data[f] for f in features]], columns=features)

#  Prediction 
pred_base = float(model.predict(X_base)[0])
st.subheader(f"Predicted Trash Weight: **{pred_base:.2f} tons**")

st.divider()

# Global Feature Importance 
st.markdown("### Feature Importance (Global)")
if hasattr(model, "feature_importances_"):
    importances = pd.Series(model.feature_importances_, index=features).sort_values()
    st.bar_chart(importances)
    st.caption("Higher bar ⇒ bigger impact on predictions (global view).")
else:
    st.info("Feature importances not available for this model.")

st.divider()

# What-if analysis on one lag feature (if available)
# Pick the first lag-like feature
lag_features = [f for f in features if "lag" in f.lower()]

if lag_features:
    target_lag = lag_features[0]  # e.g. weight_tons_lag1
    st.markdown(f"### What-if: change `{target_lag}`")

    current_value = float(input_data[target_lag])
    delta = st.slider(
        f"Change in {target_lag} (± tons)",
        -5.0,
        5.0,
        1.0,
        0.1
    )

    X_whatif = X_base.copy()
    X_whatif.loc[0, target_lag] = max(0.0, current_value + delta)
    pred_whatif = float(model.predict(X_whatif)[0])

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Baseline", f"{pred_base:.2f} t")
    with col2:
        st.metric("What-if", f"{pred_whatif:.2f} t",
                  delta=f"{(pred_whatif - pred_base):+.2f} t")

    # Simple bar plot
    fig, ax = plt.subplots()
    ax.bar(["Baseline", "What-if"], [pred_base, pred_whatif])
    ax.set_ylabel("Predicted weight (tons)")
    st.pyplot(fig)
else:
    st.info("No lag feature found for what-if analysis.")
