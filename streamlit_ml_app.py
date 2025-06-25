import streamlit as st
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, r2_score
import pandas as pd
import matplotlib.pyplot as plt

# App title
st.title("ML Demo: Classification & Regression")

# Choose model type
model_type = st.selectbox("Choose model type:", ["Classification", "Regression"])

# Generate data
if model_type == "Classification":
    X, y = make_classification(n_samples=200, n_features=5, n_classes=2, random_state=42)
else:
    X, y = make_regression(n_samples=200, n_features=5, noise=10.0, random_state=42)

# Convert to DataFrame for display
df = pd.DataFrame(X, columns=[f"Feature {i+1}" for i in range(X.shape[1])])
df["Target"] = y
st.subheader("Sample Data")
st.dataframe(df.head())

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
if model_type == "Classification":
    model = LogisticRegression()
else:
    model = LinearRegression()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Show results
st.subheader("Results")
if model_type == "Classification":
    score = accuracy_score(y_test, y_pred.round())
    st.write(f"Accuracy: **{score:.2f}**")
else:
    score = r2_score(y_test, y_pred)
    st.write(f"R² Score: **{score:.2f}**")

# Plot true vs predicted (regression only)
if model_type == "Regression":
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.7)
    ax.set_xlabel("True Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title("True vs Predicted")
    st.pyplot(fig)