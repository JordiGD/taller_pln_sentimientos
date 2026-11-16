# Análisis de sentimientos en texto con Machine Learning

Taller de **Inteligencia Computacional – UPTC**: aplicación de algoritmos de Machine Learning en el Procesamiento de Lenguaje Natural.

El sistema lee una frase, un párrafo o un documento en español y lo clasifica según la **emoción** que expresa (Alegría, Tristeza, Enojo, Miedo, Sorpresa, Asco o Neutral). A partir de la emoción obtiene el **sentimiento**: positivo, negativo o neutral.

## Contenido

| Archivo | Qué es |
|---|---|
| `taller_pln_sentimientos.ipynb` | **Producto principal del taller.** Notebook con el flujo completo: carga del dataset → limpieza → tokenización → stopwords y stemming → Bag of Words y TF-IDF → Naive Bayes, Regresión logística, Árbol de decisión y Random Forest → precisión, recall, F1, matriz de confusión, ROC/AUC → WordCloud → conclusiones. Ya viene ejecutado con todos los resultados. |
| `app.py` | Aplicación web: escribir un texto o subir un archivo y ver la emoción, el sentimiento y las palabras que influyeron. |
| `predecir.py` | Lo mismo desde la consola. |
| `src/preprocesamiento.py` | Limpieza, tokenización, stopwords y stemming (el mismo código del notebook). |
| `src/modelo.py` | Carga el modelo entrenado y analiza textos. |
| `modelo/` | Modelo final guardado por el notebook. |
| `resultados/` | Gráficas y tabla comparativa generadas por el notebook. |

## Resultados (2.722 textos de prueba)

| Modelo | Representación | Accuracy | F1 macro | AUC macro |
|---|---|---|---|---|
| **Regresión logística** | **TF-IDF** | **67,8 %** | **64,8 %** | **0,909** |
| Regresión logística | Bag of Words | 66,3 % | 63,6 % | 0,897 |
| Random Forest | TF-IDF | 64,9 % | 61,4 % | 0,893 |
| Random Forest | Bag of Words | 64,2 % | 60,6 % | 0,895 |
| Naive Bayes | TF-IDF | 61,4 % | 57,7 % | 0,877 |
| Naive Bayes | Bag of Words | 61,0 % | 57,7 % | 0,865 |
| Árbol de decisión | Bag of Words | 57,7 % | 54,5 % | 0,743 |
| Árbol de decisión | TF-IDF | 56,4 % | 52,2 % | 0,726 |

**Modelo final** (regresión logística con TF-IDF de palabras + caracteres): accuracy **68,1 %**, F1 macro **66,0 %**, AUC macro **0,920**. Acierta el sentimiento (positivo/negativo/neutral) en el **75,9 %** de los casos.

## Cómo ejecutarlo

### Opción 1: Google Colab (recomendada para el notebook)
1. Sube `taller_pln_sentimientos.ipynb` a [Google Colab](https://colab.research.google.com).
2. *Entorno de ejecución → Ejecutar todas*. Tarda unos 6 minutos; el notebook instala las librerías y descarga los datos solo.

### Opción 2: en tu computador
Requiere Python 3.10 o superior.
```bash
python -m venv venv
venv\Scripts\activate            # Windows   (Linux/Mac: source venv/bin/activate)
pip install -r requirements.txt

jupyter notebook taller_pln_sentimientos.ipynb    # el notebook
streamlit run app.py                               # la aplicación web
python predecir.py --texto "Me da miedo la tormenta" --explicar
```

La aplicación usa el modelo de `modelo/`, que ya viene entrenado. Si vuelves a ejecutar el notebook desde esta carpeta, el modelo y las gráficas se actualizan.

## Datos

- **EmoEvent** (Plaza-del-Arco et al., 2020): ~8.400 tuits en español sobre eventos de abril de 2019. https://github.com/fmplaza/EmoEvent
- **BRIGHTER / SemEval-2025 Tarea 11** (Muhammad et al., 2025), parte en español: ~3.900 comentarios de redes sociales. https://github.com/emotion-analysis-project/SemEval2025-task11

El notebook descarga ambos y corrige las etiquetas del archivo de prueba de EmoEvent, que en el repositorio no corresponden a los tuits.

## Limitaciones

- No detecta el sarcasmo: *"Pues qué maravilla, otra vez se cayó la aplicación"* sale como Alegría.
- Arrastra un sesgo temático de los datos: asocia hashtags de eventos de 2019 (*laliga*, *juegodetronos*, *venezuela*) con emociones concretas.
- Solo funciona en español.
- Interpreta cómo está escrito el texto, no lo que la persona siente realmente.
