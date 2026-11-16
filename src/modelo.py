"""
Uso del modelo entrenado en el notebook (taller_pln_sentimientos.ipynb).

El notebook guarda en modelo/ un Pipeline de scikit-learn
(TF-IDF de palabras + caracteres -> Regresión logística) que recibe el texto
ya preprocesado. Esta clase aplica el mismo preprocesamiento
(src/preprocesamiento.py) y devuelve emoción, confianza y sentimiento.

    analizador = AnalizadorSentimiento()
    r = analizador.analizar("¡Por fin aprobé el examen!")
    print(r["emocion"], r["confianza"], r["sentimiento"])
"""

import json
import re
import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RUTA_META, RUTA_MODELO, SENTIMIENTO, sentimiento_de  # noqa: E402
from src.preprocesamiento import preprocesar  # noqa: E402

_ORACIONES = re.compile(r"(?<=[.!?…])\s+|\n+")


def dividir_oraciones(texto: str) -> list[str]:
    """Separa un párrafo en oraciones (por puntos, signos o saltos de línea)."""
    return [p.strip() for p in _ORACIONES.split(str(texto)) if len(p.strip()) > 1]


class AnalizadorSentimiento:
    def __init__(self, ruta_modelo: str | Path = RUTA_MODELO):
        ruta_modelo = Path(ruta_modelo)
        if not ruta_modelo.is_file():
            raise FileNotFoundError(
                f"No se encontró el modelo en {ruta_modelo}. "
                "Ejecuta primero el notebook taller_pln_sentimientos.ipynb."
            )
        self.modelo = joblib.load(ruta_modelo)
        self.etiquetas: list[str] = list(self.modelo.classes_)
        ruta_meta = ruta_modelo.parent / RUTA_META.name
        self.metadatos = json.loads(ruta_meta.read_text(encoding="utf-8")) if ruta_meta.is_file() else {}

    def probabilidades(self, textos: list[str]) -> np.ndarray:
        """Matriz (n_textos x n_emociones) con la probabilidad de cada emoción."""
        return self.modelo.predict_proba([preprocesar(t) for t in textos])

    def _resultado(self, texto: str, probs: np.ndarray) -> dict:
        i = int(np.argmax(probs))
        emocion = self.etiquetas[i]
        por_sentimiento = {"Positivo": 0.0, "Negativo": 0.0, "Neutral": 0.0}
        for e, p in zip(self.etiquetas, probs):
            if SENTIMIENTO.get(e) in por_sentimiento:
                por_sentimiento[SENTIMIENTO[e]] += float(p)
        return {
            "texto": texto,
            "emocion": emocion,
            "confianza": float(probs[i]),
            "sentimiento_emocion": sentimiento_de(emocion),
            # Sentimiento global: el tipo cuyas emociones suman más probabilidad
            "sentimiento": max(por_sentimiento, key=por_sentimiento.get),
            "prob_sentimiento": por_sentimiento,
            "probabilidades": {e: float(p) for e, p in zip(self.etiquetas, probs)},
        }

    def analizar(self, texto: str) -> dict:
        if not str(texto).strip():
            raise ValueError("El texto está vacío.")
        return self._resultado(texto, self.probabilidades([texto])[0])

    def analizar_lote(self, textos: list[str]) -> list[dict]:
        textos = [str(t) for t in textos]
        return [self._resultado(t, p) for t, p in zip(textos, self.probabilidades(textos))]

    def analizar_por_oraciones(self, texto: str) -> list[dict]:
        return self.analizar_lote(dividir_oraciones(texto) or [texto])

    def explicar(self, texto: str, emocion: str | None = None, max_palabras: int = 60):
        """
        Qué palabras influyeron (técnica de oclusión): se quita cada palabra y
        se mide cuánto cambia la probabilidad de la emoción.
        Peso positivo = la palabra apoya esa emoción.
        """
        palabras = re.findall(r"\S+", str(texto))[:max_palabras]
        if not palabras:
            return []
        variantes = [" ".join(palabras)] + [
            " ".join(palabras[:i] + palabras[i + 1:]) for i in range(len(palabras))
        ]
        probs = self.probabilidades(variantes)
        j = self.etiquetas.index(emocion) if emocion else int(np.argmax(probs[0]))
        return [(p, float(probs[0, j] - probs[i + 1, j])) for i, p in enumerate(palabras)]
