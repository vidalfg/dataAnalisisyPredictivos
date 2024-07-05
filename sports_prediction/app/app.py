from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle
import os

# Verificar si el archivo del modelo existe
model_path = 'D:\\WS_ANALITICA_DATOS\\sports_prediction\\model\\modelocatboost.pkl'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"No se encontró el archivo del modelo en la ruta: {model_path}")

# Cargar el modelo entrenado
with open(model_path, 'rb') as f:
    model = pickle.load(f)

app = FastAPI()

# Definir el modelo de datos para la entrada
class PredictionInput(BaseModel):
    games: float
    time: float
    assists: float
    shots: float
    key_passes: float
    yellow_cards: float
    red_cards: float

@app.get("/")
def read_root():
    return {"mensaje": "bienvenido Grupo 04"}

# Definir la ruta de la API para hacer predicciones
@app.post("/predict")
def predict(data: PredictionInput):
    # Convertir la entrada a un formato adecuado para el modelo
    input_data = np.array([[data.games, data.time, data.assists, data.shots, data.key_passes, data.yellow_cards, data.red_cards]])
    # Realizar la predicción
    prediction = model.predict(input_data)
    prediction = prediction.tolist()
    # Devolver la predicción
    return {"prediction": prediction[0]}
#pip install uvicorn
#python -m uvicorn app:app --reload
