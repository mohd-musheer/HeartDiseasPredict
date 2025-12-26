import joblib

model = joblib.load("../model/heart_disease_model.pkl")

# Example data - replace with user input
user_data = [[45,0,0,118,180,0,0,160,0,0.0,0,0,0]]

pred = model.predict(user_data)

if pred[0] == 1:
    print("⚠️ High Chance of Heart Disease")
else:
    print("💚 No Heart Disease Detected")
