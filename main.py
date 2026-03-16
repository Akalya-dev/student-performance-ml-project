import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import joblib


# 1️⃣ Load dataset
df = pd.read_csv("data/student-mat.csv", sep=";")

# 2️⃣ Select numeric columns
df = df.select_dtypes(include=["int64"])

# 3️⃣ Features and target
X = df.drop("G3", axis=1)
y = df["G3"]

# 4️⃣ Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5️⃣ Train model
model = LinearRegression()
model.fit(X_train, y_train)

# 6️⃣ Predictions
predictions = model.predict(X_test)

# 7️⃣ Evaluation
mse = mean_squared_error(y_test, predictions)

print("Model trained successfully")
print("Mean Squared Error:", mse)

# 8️⃣ Save model
joblib.dump(model, "student_model.pkl")
print("Model saved successfully!")

# --------------------------------
# 9️⃣ ADD INPUT SYSTEM HERE
# --------------------------------

g1 = int(input("Enter G1 marks: "))
g2 = int(input("Enter G2 marks: "))
absences = int(input("Enter absences: "))

# Create input dataframe
input_data = X.iloc[0:1].copy()

input_data["G1"] = g1
input_data["G2"] = g2
input_data["absences"] = absences

prediction = model.predict(input_data)

print("Predicted Final Grade (G3):", prediction[0])