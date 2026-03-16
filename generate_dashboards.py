import json
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
LOGS_DIR = BASE_DIR / "logs"
PLOTS_DIR = LOGS_DIR / "plots"
DASHBOARD_PATH = BASE_DIR / "dashboard.html"

TRAIN_LOG_PATH = LOGS_DIR / "training.log"
ERROR_LOG_PATH = LOGS_DIR / "errors.log"
EVAL_SUMMARY_PATH = LOGS_DIR / "evaluation_summary.json"
EVAL_PREDICTIONS_PATH = LOGS_DIR / "evaluation_predictions.csv"


def read_text_file(path, max_lines=50):
    if not path.exists():
        return "Archivo no encontrado."
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return "".join(lines[-max_lines:])


def load_summary():
    if not EVAL_SUMMARY_PATH.exists():
        return {}
    with open(EVAL_SUMMARY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_predictions_table():
    if not EVAL_PREDICTIONS_PATH.exists():
        return "<p>No hay predicciones aún.</p>"

    df = pd.read_csv(EVAL_PREDICTIONS_PATH)
    preview = df.head(10)
    return preview.to_html(index=False, classes="table", border=0)


def get_plot_images():
    if not PLOTS_DIR.exists():
        return []

    images = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
        images.extend(PLOTS_DIR.glob(ext))

    return sorted(images)


def generate_html():
    summary = load_summary()
    training_log = read_text_file(TRAIN_LOG_PATH, max_lines=80)
    errors_log = read_text_file(ERROR_LOG_PATH, max_lines=40)
    predictions_table = load_predictions_table()
    images = get_plot_images()

    metrics_html = ""
    if summary:
        metrics_html = f"""
        <div class="metrics">
            <div class="card"><h3>Muestras evaluadas</h3><p>{summary.get("samples_evaluated", 0)}</p></div>
            <div class="card"><h3>Exact Match</h3><p>{summary.get("exact_match_accuracy", 0):.4f}</p></div>
            <div class="card"><h3>BLEU promedio</h3><p>{summary.get("average_bleu_score", 0):.4f}</p></div>
            <div class="card"><h3>Token Accuracy</h3><p>{summary.get("average_token_accuracy", 0):.4f}</p></div>
        </div>
        """
    else:
        metrics_html = "<p>No hay métricas disponibles todavía.</p>"

    images_html = ""
    for img in images:
        relative_path = img.relative_to(BASE_DIR).as_posix()
        images_html += f"""
        <div class="plot-card">
            <h4>{img.name}</h4>
            <img src="{relative_path}" alt="{img.name}">
        </div>
        """

    html = f"""
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <title>Dashboard - Traductor RNN</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background: #f4f6f9;
            color: #222;
        }}
        header {{
            background: #1f2937;
            color: white;
            padding: 24px;
            text-align: center;
        }}
        main {{
            max-width: 1200px;
            margin: 20px auto;
            padding: 20px;
        }}
        .section {{
            background: white;
            margin-bottom: 24px;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        }}
        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
        }}
        .card {{
            background: #eef2ff;
            padding: 16px;
            border-radius: 10px;
            text-align: center;
        }}
        .card h3 {{
            margin: 0 0 10px;
            font-size: 16px;
        }}
        .card p {{
            font-size: 24px;
            font-weight: bold;
            margin: 0;
        }}
        .plots {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 16px;
        }}
        .plot-card {{
            background: #fafafa;
            border: 1px solid #ddd;
            border-radius: 10px;
            padding: 12px;
        }}
        .plot-card img {{
            width: 100%;
            border-radius: 8px;
        }}
        pre {{
            background: #111827;
            color: #e5e7eb;
            padding: 16px;
            border-radius: 8px;
            overflow-x: auto;
            white-space: pre-wrap;
        }}
        .table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 16px;
        }}
        .table th, .table td {{
            border: 1px solid #ddd;
            padding: 10px;
            font-size: 14px;
        }}
        .table th {{
            background: #f3f4f6;
        }}
    </style>
</head>
<body>
    <header>
        <h1>Dashboard del Traductor RNN</h1>
        <p>Entrenamiento, evaluación, logs y visualizaciones</p>
    </header>

    <main>
        <section class="section">
            <h2>Métricas de evaluación</h2>
            {metrics_html}
        </section>

        <section class="section">
            <h2>Gráficas</h2>
            <div class="plots">
                {images_html if images_html else "<p>No hay imágenes generadas todavía.</p>"}
            </div>
        </section>

        <section class="section">
            <h2>Vista previa de predicciones</h2>
            {predictions_table}
        </section>

        <section class="section">
            <h2>Últimas líneas de training.log</h2>
            <pre>{training_log}</pre>
        </section>

        <section class="section">
            <h2>Últimas líneas de errors.log</h2>
            <pre>{errors_log}</pre>
        </section>
    </main>
</body>
</html>
"""
    with open(DASHBOARD_PATH, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Dashboard generado en: {DASHBOARD_PATH}")


if __name__ == "__main__":
    generate_html()