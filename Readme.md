# RNN Traductor

Este proyecto implementa un ejemplo básico de un **traductor de texto** entrenado con una **Red Neuronal Recurrente (RNN)** usando TensorFlow y Keras.  
El modelo aprende a traducir frases simples de **inglés a español** a partir de un conjunto de pares de traducción almacenados en formato JSON.

Además del entrenamiento y la predicción, el proyecto incluye:

- guardado de modelo y tokenizadores
- registro de logs de entrenamiento y errores
- historial de entrenamiento en CSV
- evaluación automática del modelo
- generación de gráficas
- dashboard HTML con métricas, logs e imágenes

---

## Estructura del proyecto

```text
rnn_traductor/
│
├── data/
│   └── translation_pairs.json
│
├── logs/
│   ├── training.log
│   ├── errors.log
│   ├── training_history.csv
│   ├── evaluation_summary.json
│   ├── evaluation_predictions.csv
│   └── plots/
│
├── models/
│   ├── translation_model.keras
│   ├── input_tokenizer.pkl
│   ├── target_tokenizer.pkl
│   ├── translation_metadata.json
│   └── checkpoints/
│
├── train_translation.py
├── evaluate_translation.py
├── predict_translation.py
├── plot_results.py
├── generate_dashboard.py
├── requirements.txt
└── README.md
```
## Instalación
#bash

python -m venv venv

venv\Scripts\activate

pip install -r requirements.txt

## Entrenar el modelo

python train_translation.py

## Evaluar el modelo entrenado

python evaluate_translation.py

## Generar imágenes de resultados

python plot_results.py

## Generar el dashboard HTML

python generate_dashboard.py

## Probar traducciones manualmente

python predict_translation.py



