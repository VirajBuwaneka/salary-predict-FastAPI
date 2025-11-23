from sklearn.linear_model import LinearRegression
import joblib

# Training dataset (experience, age, score)
X = [
    [1, 22, 65],
    [2, 25, 70],
    [3, 26, 75],
    [4, 29, 80],
    [5, 32, 85],
    [6, 35, 90],
    [7, 38, 92],
    [8, 40, 95]
]

# Salary labels (target)
y = [40000, 52000, 60000, 70000, 80000, 90000, 100000, 110000]

# Train the model
model = LinearRegression()
model.fit(X, y)

# Save model
joblib.dump(model, "salary_model.joblib")

print("Model saved as salary_model.joblib")
