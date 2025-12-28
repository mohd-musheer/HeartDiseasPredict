# ❤️ Heart Disease Prediction (Machine Learning)

This project predicts whether a person has heart disease based on medical data using
**Random Forest (99.4% validated accuracy)**.

---API in Progress

## 📌 Dataset
- **Rows:** 1000+
- **Target:** `HeartDisease` (0 = No Disease, 1 = Disease)
- All attributes are numeric & medically relevant.

Dataset: `data/heart_disease.csv`

---

## 🚀 Project Structure



    user_input = pd.DataFrame([[d.Age, d.Sex, d.ChestPainType, d.RestingBP, d.Cholesterol, d.FastingBS, d.RestingECG, d.MaxHR, d.ExerciseAngina, d.Oldpeak, d.ST_Slope, d.NumMajorVessels, d.Thalassemia]],columns=[
    "Age","Sex","ChestPainType","RestingBP","Cholesterol","FastingBS","RestingECG",
    "MaxHR","ExerciseAngina","Oldpeak","ST_Slope","NumMajorVessels","Thalassemia"])
