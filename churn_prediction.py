# Customer Churn Prediction using Machine Learning
# GUI Application using Tkinter

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ---------------------------
# Sample Dataset
# ---------------------------

data = {
    'SeniorCitizen': ['No', 'Yes', 'No', 'No', 'Yes'],
    'Partner': ['Yes', 'No', 'Yes', 'No', 'Yes'],
    'Dependents': ['Yes', 'No', 'No', 'Yes', 'No'],
    'PhoneService': ['Yes', 'Yes', 'No', 'Yes', 'Yes'],
    'MultipleLines': ['Yes', 'No', 'No', 'Yes', 'Yes'],
    'InternetService': ['DSL', 'Fiber optic', 'DSL', 'No', 'DSL'],
    'OnlineSecurity': ['Yes', 'No', 'Yes', 'No', 'Yes'],
    'OnlineBackup': ['Yes', 'No', 'Yes', 'Yes', 'No'],
    'DeviceProtection': ['Yes', 'No', 'Yes', 'No', 'Yes'],
    'TechSupport': ['Yes', 'No', 'Yes', 'No', 'Yes'],
    'StreamingTV': ['Yes', 'No', 'Yes', 'No', 'Yes'],
    'StreamingMovies': ['Yes', 'No', 'Yes', 'No', 'Yes'],
    'Contract': ['One year', 'Month-to-month', 'Two year', 'Month-to-month', 'One year'],
    'PaymentMethod': ['Mailed check', 'Electronic check', 'Bank transfer', 'Credit card', 'Mailed check'],
    'PaperlessBilling': ['Yes', 'No', 'Yes', 'Yes', 'No'],
    'Gender': ['Male', 'Female', 'Female', 'Male', 'Female'],
    'MonthlyCharges': [100, 70, 90, 50, 120],
    'TotalCharges': [1000, 200, 1500, 300, 2000],
    'Tenure': [12, 2, 24, 1, 36],
    'Churn': [0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

# ---------------------------
# Encoding Categorical Data
# ---------------------------

encoders = {}

for column in df.columns:
    if df[column].dtype == object:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        encoders[column] = le

# Features and Target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# ---------------------------
# Train Model
# ---------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ---------------------------
# GUI Window
# ---------------------------

root = tk.Tk()
root.title("Customer Churn Prediction")
root.geometry("1100x700")
root.configure(bg="lightblue")

title = tk.Label(
    root,
    text="CUSTOMER CHURN PREDICTION",
    font=("Times New Roman", 24, "bold"),
    bg="lightblue"
)

title.pack(pady=20)

frame = tk.Frame(root, bg="lightblue")
frame.pack()

# ---------------------------
# Input Fields
# ---------------------------

fields = {
    "SeniorCitizen": ["Yes", "No"],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["Yes", "No"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["Yes", "No"],
    "OnlineBackup": ["Yes", "No"],
    "DeviceProtection": ["Yes", "No"],
    "TechSupport": ["Yes", "No"],
    "StreamingTV": ["Yes", "No"],
    "StreamingMovies": ["Yes", "No"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
    "PaperlessBilling": ["Yes", "No"],
    "Gender": ["Male", "Female"]
}

entries = {}

row = 0
col = 0

# Dropdown Fields
for field, values in fields.items():

    label = tk.Label(
        frame,
        text=field,
        bg="lightblue",
        font=("Arial", 10, "bold")
    )

    label.grid(row=row, column=col, padx=10, pady=10)

    combo = ttk.Combobox(frame, values=values, width=18)
    combo.current(0)

    combo.grid(row=row + 1, column=col, padx=10)

    entries[field] = combo

    col += 1

    if col > 3:
        col = 0
        row += 2

# Numeric Fields
numeric_fields = ["MonthlyCharges", "TotalCharges", "Tenure"]

for field in numeric_fields:

    label = tk.Label(
        frame,
        text=field,
        bg="lightblue",
        font=("Arial", 10, "bold")
    )

    label.grid(row=row, column=col, padx=10, pady=10)

    entry = tk.Entry(frame, width=20)

    entry.grid(row=row + 1, column=col, padx=10)

    entries[field] = entry

    col += 1

# ---------------------------
# Result Label
# ---------------------------

result_label = tk.Label(
    root,
    text="",
    font=("Arial", 12, "bold")
)

result_label.pack(pady=20)

# ---------------------------
# Prediction Function
# ---------------------------

def predict_churn():

    try:

        input_data = []

        # Encode categorical fields
        for field in fields.keys():

            value = entries[field].get()

            encoded_value = encoders[field].transform([value])[0]

            input_data.append(encoded_value)

        # Add numeric values
        input_data.append(float(entries["MonthlyCharges"].get()))
        input_data.append(float(entries["TotalCharges"].get()))
        input_data.append(float(entries["Tenure"].get()))

        # Convert to numpy array
        input_array = np.array(input_data).reshape(1, -1)

        # Prediction
        prediction = model.predict(input_array)[0]

        probability = model.predict_proba(input_array)[0][1] * 100

        # Display result
        if prediction == 1:

            result_label.config(
                text=f"Customer is likely to CHURN\nConfidence: {probability:.2f}%",
                bg="pink",
                fg="red",
                width=50,
                height=3
            )

        else:

            result_label.config(
                text=f"Customer is likely to CONTINUE\nConfidence: {100 - probability:.2f}%",
                bg="lightgreen",
                fg="green",
                width=50,
                height=3
            )

    except ValueError:

        messagebox.showerror(
            "Input Error",
            "Please enter valid numeric values."
        )

    except Exception as e:

        messagebox.showerror(
            "Error",
            str(e)
        )

# ---------------------------
# Predict Button
# ---------------------------

predict_btn = tk.Button(
    root,
    text="PREDICT",
    bg="red",
    fg="white",
    font=("Arial", 12, "bold"),
    command=predict_churn
)

predict_btn.pack(pady=20)

# ---------------------------
# Run Application
# ---------------------------

root.mainloop()