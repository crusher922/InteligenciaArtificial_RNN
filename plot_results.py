import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
LOGS_DIR = BASE_DIR / "logs"
PLOTS_DIR = LOGS_DIR / "plots"

HISTORY_CSV_PATH = LOGS_DIR / "training_history.csv"
EVAL_SUMMARY_PATH = LOGS_DIR / "evaluation_summary.json"
EVAL_PREDICTIONS_PATH = LOGS_DIR / "evaluation_predictions.csv"


def ensure_dir():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def plot_training_history():
    if not HISTORY_CSV_PATH.exists():
        print("No existe training_history.csv")
        return

    df = pd.read_csv(HISTORY_CSV_PATH)

    if "loss" in df.columns and "val_loss" in df.columns:
        plt.figure(figsize=(10, 6))
        plt.plot(df["loss"], label="loss")
        plt.plot(df["val_loss"], label="val_loss")
        plt.title("Pérdida de entrenamiento y validación")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "loss_curve.png")
        plt.close()

    if "accuracy" in df.columns and "val_accuracy" in df.columns:
        plt.figure(figsize=(10, 6))
        plt.plot(df["accuracy"], label="accuracy")
        plt.plot(df["val_accuracy"], label="val_accuracy")
        plt.title("Accuracy de entrenamiento y validación")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "accuracy_curve.png")
        plt.close()


def plot_evaluation_summary():
    if not EVAL_SUMMARY_PATH.exists():
        print("No existe evaluation_summary.json")
        return

    with open(EVAL_SUMMARY_PATH, "r", encoding="utf-8") as f:
        summary = json.load(f)

    metrics = {
        "Exact Match": summary.get("exact_match_accuracy", 0),
        "BLEU": summary.get("average_bleu_score", 0),
        "Token Accuracy": summary.get("average_token_accuracy", 0)
    }

    plt.figure(figsize=(8, 5))
    plt.bar(metrics.keys(), metrics.values())
    plt.title("Resumen de métricas de evaluación")
    plt.ylabel("Valor")
    plt.ylim(0, 1)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "evaluation_metrics.png")
    plt.close()


def plot_prediction_quality():
    if not EVAL_PREDICTIONS_PATH.exists():
        print("No existe evaluation_predictions.csv")
        return

    df = pd.read_csv(EVAL_PREDICTIONS_PATH)

    if "exact_match" in df.columns:
        counts = df["exact_match"].value_counts().sort_index()
        labels = ["Incorrecto", "Correcto"]
        values = [counts.get(0, 0), counts.get(1, 0)]

        plt.figure(figsize=(7, 5))
        plt.bar(labels, values)
        plt.title("Predicciones correctas vs incorrectas")
        plt.ylabel("Cantidad")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "prediction_quality.png")
        plt.close()

    if "bleu_score" in df.columns:
        plt.figure(figsize=(8, 5))
        plt.hist(df["bleu_score"], bins=10)
        plt.title("Distribución de BLEU Score")
        plt.xlabel("BLEU")
        plt.ylabel("Frecuencia")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "bleu_distribution.png")
        plt.close()


def main():
    ensure_dir()
    plot_training_history()
    plot_evaluation_summary()
    plot_prediction_quality()
    print(f"Gráficas guardadas en: {PLOTS_DIR}")


if __name__ == "__main__":
    main()