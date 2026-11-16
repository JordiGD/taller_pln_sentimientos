import html
import re

import emoji
import nltk
from nltk.stem import SnowballStemmer

# ---------------------------------------------------------------------------
# 1. LIMPIEZA: quita enlaces, menciones y símbolos; normaliza emojis y risas
# ---------------------------------------------------------------------------
EMOTICONES = [
    (r"(?<!\S)[:;=]-?[)\]]+", " emoticono_feliz "),     # :)  ;)  =)
    (r"(?<!\S)[:;=]-?D+\b", " emoticono_risa "),        # :D
    (r"(?<!\S)[:;=]'?-?[(\[]+", " emoticono_triste "),  # :(  :'(
    (r"<3+", " emoticono_corazon "),                     # <3
]


def limpiar(texto: str) -> str:
    t = html.unescape(str(texto))                                     # &amp; -> &
    t = re.sub(r"https?://\S+|www\.\S+|\bURL\b", " ", t)             # enlaces
    t = re.sub(r"@<username>|@\w+|\bUSER\b|\bHASHTAG\b", " ", t)     # menciones
    t = re.sub(r"#(\w+)", r" \1 ", t)                                # #feliz -> feliz
    for patron, palabra in EMOTICONES:                               # :( -> emoticono_triste
        t = re.sub(patron, palabra, t)
    t = emoji.demojize(t, language="es", delimiters=(" emoji_", " "))  # 😭 -> emoji_cara_llorando_fuerte
    t = t.lower().replace("¡", "!").replace("¿", "?")
    t = re.sub(r"\b(?:a?j+[aeiou]+){2,}j?\b|\b(?:h+a+){2,}h?\b|\bj+s+j[js]*\b|\bxd+\b",
               " jaja ", t)                                          # jajajaja, jsjs, xD -> jaja
    t = re.sub(r"(\w)\1{2,}", r"\1\1", t)                            # siiiii -> sii
    t = re.sub(r"([!?])\1{2,}", r"\1\1", t)                          # !!!!! -> !!
    return t


# ---------------------------------------------------------------------------
# 2. TOKENIZACIÓN: divide el texto en palabras (tokens)
#    Se conservan las palabras (incluidos los emojis convertidos) y los signos
#    ! y ?, que expresan emoción. Números y demás símbolos se descartan.
# ---------------------------------------------------------------------------
def tokenizar(texto: str) -> list[str]:
    return re.findall(r"[a-záéíóúüñ_]+|[!?]", texto)


# ---------------------------------------------------------------------------
# 3. STOPWORDS: elimina palabras muy frecuentes que no aportan significado
#    Se usa la lista de NLTK para español (313 palabras), pero se CONSERVAN
#    las que cambian la emoción de la frase (negaciones, intensificadores...).
# ---------------------------------------------------------------------------
try:
    from nltk.corpus import stopwords
    _LISTA_NLTK = stopwords.words("spanish")
except LookupError:
    nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords
    _LISTA_NLTK = stopwords.words("spanish")

PALABRAS_A_CONSERVAR = {
    "no", "ni", "sin", "nada", "nunca", "pero", "contra",            # negación y contraste
    "muy", "mucho", "qué", "sí",                                     # intensidad y exclamación
    "siente", "sentido", "sentida", "sentidos", "sentidas", "sintiendo",
    "tengo", "tiene",                                                # "tengo miedo", "me siente mal"
}
STOPWORDS = set(_LISTA_NLTK) - PALABRAS_A_CONSERVAR


def quitar_stopwords(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t not in STOPWORDS]


# ---------------------------------------------------------------------------
# 4. STEMMING: reduce cada palabra a su raíz (algoritmo Snowball para español)
#    felicidad, felices -> felic ; tristeza -> tristez ; llorando -> llor
#    Los emojis convertidos, la risa y los signos no se modifican.
# ---------------------------------------------------------------------------
_stemmer = SnowballStemmer("spanish")
_SIN_STEMMING = ("emoji_", "emoticono_")


def aplicar_stemming(tokens: list[str]) -> list[str]:
    return [
        t if t.startswith(_SIN_STEMMING) or t in {"!", "?", "jaja"} else _stemmer.stem(t)
        for t in tokens
    ]


# ---------------------------------------------------------------------------
# PREPROCESAMIENTO COMPLETO: limpieza -> tokenización -> stopwords -> stemming
# Devuelve los tokens unidos por espacios (lo que recibe el vectorizador).
# ---------------------------------------------------------------------------
def preprocesar(texto: str, quitar_stop: bool = True, stemming: bool = True) -> str:
    tokens = tokenizar(limpiar(texto))
    if quitar_stop:
        tokens = quitar_stopwords(tokens)
    if stemming:
        tokens = aplicar_stemming(tokens)
    return " ".join(tokens)
