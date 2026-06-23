from fastapi.testclient import TestClient

import api


client = TestClient(api.app)


class DummyModel:
    def __init__(self, prediction):
        self.prediction = prediction

    def predict(self, user_input):
        return [self.prediction]


def test_predict_returns_fit_and_fine(monkeypatch):
    monkeypatch.setattr(api.joblib, "load", lambda _: DummyModel(0))

    response = client.post(
        "/predict",
        json={
            "Age": 45,
            "Sex": 1,
            "ChestPainType": 2,
            "RestingBP": 180,
            "Cholesterol": 250,
            "FastingBS": 1,
            "RestingECG": 1,
            "MaxHR": 120,
            "ExerciseAngina": 1,
            "Oldpeak": 2.2,
            "ST_Slope": 2,
            "NumMajorVessels": 3,
            "Thalassemia": 2,
        },
    )

    assert response.status_code == 200
    assert response.json() == {"message": "fit and fine"}


def test_predict_returns_hospital_message(monkeypatch):
    monkeypatch.setattr(api.joblib, "load", lambda _: DummyModel(1))

    response = client.post(
        "/predict",
        json={
            "Age": 60,
            "Sex": 0,
            "ChestPainType": 1,
            "RestingBP": 140,
            "Cholesterol": 220,
            "FastingBS": 0,
            "RestingECG": 0,
            "MaxHR": 150,
            "ExerciseAngina": 0,
            "Oldpeak": 1.0,
            "ST_Slope": 1,
            "NumMajorVessels": 0,
            "Thalassemia": 3,
        },
    )

    assert response.status_code == 200
    assert response.json() == {"message": "Hostpital ja laude"}