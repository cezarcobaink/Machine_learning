"""Funções do projeto de churn: carga, análise de sobrevivência e avaliação financeira da campanha."""
from pathlib import Path

import numpy as np
import pandas as pd

ARQUIVO = Path(__file__).with_name("Telco-Customer-Churn.csv")

# Premissas da campanha de retenção (explícitas para poderem ser discutidas e alteradas)
CUSTO_CONTATO = 50.0      # custo por cliente abordado (incentivo + operação), em US$
TAXA_SUCESSO = 0.30       # fração dos clientes que iriam sair e ficam após a oferta
MESES_VALOR = 12          # horizonte de receita preservada por cliente retido


def carregar() -> pd.DataFrame:
    """Lê a base e corrige tipos.

    TotalCharges vem como texto e tem 11 valores em branco, todos de clientes com tenure = 0
    (acabaram de entrar e ainda não pagaram nada), por isso o valor correto é zero.
    """
    d = pd.read_csv(ARQUIVO)
    d["TotalCharges"] = pd.to_numeric(d["TotalCharges"], errors="coerce").fillna(0.0)
    d["churn"] = (d["Churn"] == "Yes").astype(int)
    return d.drop(columns=["Churn"])


def kaplan_meier(tempo: pd.Series, evento: pd.Series) -> pd.DataFrame:
    """Estimador de Kaplan-Meier da curva de sobrevivência.

    tempo: meses de casa; evento: 1 se o cliente saiu, 0 se ainda é cliente (censurado).
    Retorna, para cada mês, a probabilidade de o cliente continuar ativo.
    """
    df = pd.DataFrame({"t": tempo.values, "e": evento.values})
    tabela = df.groupby("t").agg(saidas=("e", "sum"), total=("e", "size")).sort_index()
    em_risco = tabela["total"][::-1].cumsum()[::-1]           # clientes ainda ativos no início de cada mês
    sobrevivencia = (1 - tabela["saidas"] / em_risco).cumprod()
    return pd.DataFrame({"meses": tabela.index, "sobrevivencia": sobrevivencia.values})


def valor_esperado(prob: np.ndarray, mensalidade: np.ndarray) -> np.ndarray:
    """Ganho esperado de abordar cada cliente: P(sair) × sucesso × receita preservada − custo."""
    return prob * TAXA_SUCESSO * MESES_VALOR * mensalidade - CUSTO_CONTATO


def lucro_realizado(saiu: np.ndarray, mensalidade: np.ndarray, abordados: np.ndarray) -> float:
    """Lucro da campanha medido com o que de fato aconteceu no conjunto de teste."""
    receita = (saiu * abordados * TAXA_SUCESSO * MESES_VALOR * mensalidade).sum()
    return float(receita - CUSTO_CONTATO * abordados.sum())


def curva_lucro(saiu, mensalidade, pontuacao) -> pd.DataFrame:
    """Lucro ao abordar os k clientes com maior pontuação, para k de 0 até todos."""
    ordem = np.argsort(-pontuacao)
    ganho = (saiu[ordem] * TAXA_SUCESSO * MESES_VALOR * mensalidade[ordem]) - CUSTO_CONTATO
    lucro = np.concatenate([[0.0], np.cumsum(ganho)])
    return pd.DataFrame({"pct_abordados": np.arange(len(lucro)) / len(saiu) * 100, "lucro": lucro})
