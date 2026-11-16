"""
Configuración de la aplicación web y de la consola.

El entrenamiento completo está en el notebook taller_pln_sentimientos.ipynb,
que guarda el modelo en modelo/ y las gráficas en resultados/.
"""

from pathlib import Path

RAIZ = Path(__file__).resolve().parent
CARPETA_MODELO = RAIZ / "modelo"
CARPETA_RESULTADOS = RAIZ / "resultados"
RUTA_MODELO = CARPETA_MODELO / "modelo.joblib"
RUTA_META = CARPETA_MODELO / "meta.json"

EMOCIONES = ["Alegría", "Tristeza", "Enojo", "Miedo", "Sorpresa", "Asco", "Neutral"]

# Polaridad del sentimiento asociada a cada emoción.
# La sorpresa es ambigua (puede ser buena o mala), por eso cuenta como neutral.
SENTIMIENTO = {
    "Alegría": "Positivo",
    "Sorpresa": "Neutral",
    "Neutral": "Neutral",
    "Tristeza": "Negativo",
    "Enojo": "Negativo",
    "Miedo": "Negativo",
    "Asco": "Negativo",
}

EMOJIS = {
    "Alegría": "😄", "Tristeza": "😢", "Enojo": "😠", "Miedo": "😨",
    "Sorpresa": "😮", "Asco": "🤢", "Neutral": "😐",
    "Positivo": "👍", "Negativo": "👎",
}


def sentimiento_de(emocion: str) -> str:
    return SENTIMIENTO.get(emocion, "Desconocido")
