"""Pré-processamento de texto em português para as avaliações da Olist."""
import re
import unicodedata
from pathlib import Path

import pandas as pd

ARQUIVO = Path(__file__).with_name("avaliacoes_olist.csv.gz")

# Lista curta de palavras sem conteúdo. As NEGAÇÕES ficam de fora de propósito:
# "não chegou" e "chegou" têm sentidos opostos, e listas prontas de stopwords costumam apagar o "não".
STOPWORDS = set("""
a o as os um uma uns umas de do da dos das em no na nos nas por para pra com e que se ao aos
ele ela eles elas eu voce voces me te lhe meu minha seu sua isso isto esse essa este esta foi
ser ter tem era sao estou esta estava ja ainda mais muito tambem so pois como quando onde ate
""".split())
NEGACOES = {"nao", "nem", "nunca", "jamais", "sem"}


def normalizar(texto: str) -> str:
    """Minúsculas, sem acentos, sem números nem pontuação."""
    texto = unicodedata.normalize("NFKD", str(texto).lower())
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    texto = re.sub(r"[^a-z\s]", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()


def limpar(texto: str, manter_negacoes: bool = True) -> str:
    """Normaliza e remove stopwords; opcionalmente remove também as negações (para o experimento)."""
    remover = STOPWORDS if manter_negacoes else STOPWORDS | NEGACOES
    return " ".join(p for p in normalizar(texto).split() if p not in remover)


def carregar() -> pd.DataFrame:
    """Avaliações com texto. Rótulo: negativa (nota 1–2) = 1, positiva (4–5) = 0, neutra (3) = NaN."""
    d = pd.read_csv(ARQUIVO, parse_dates=["data"])
    d["negativa"] = d["nota"].map({1: 1, 2: 1, 4: 0, 5: 0})
    return d


# Temas de reclamação, definidos por expressões simples sobre o texto normalizado
TEMAS = {
    "Não recebeu o produto": r"\bnao (recebi|chegou|foi entregue|entregue)|ainda nao (recebi|chegou)",
    "Atraso na entrega": r"\batras|\bprazo|\bdemor",
    "Recebeu incompleto ou parte do pedido": r"\bfalt(a|ou|ando)|\bapenas um\b|\bsomente um\b|\bincomplet",
    "Produto errado ou diferente": r"\berrad|\bdiferente|\boutro produto|\bnao (era|e) o",
    "Defeito ou qualidade ruim": r"\bdefeit|\bquebrad|\bnao funciona|\bpessima qualidade|\bqualidade (ruim|baixa)",
    "Atendimento ou resposta": r"\batendiment|\bresposta|\bcontato|\bsac\b|\breclam",
}
