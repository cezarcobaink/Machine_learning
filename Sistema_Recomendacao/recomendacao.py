"""Recomendação de filmes: dados, modelos e avaliação usados no notebook.

Dados: MovieTweetings (Simon Dooms, licença MIT), com avaliações de filmes publicadas no Twitter de 2013 a 2021.
Cada avaliação é tratada como "a pessoa assistiu ao filme" (sinal implícito); a nota não entra no alvo.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

PASTA = Path(__file__).resolve().parent
MIN_AVALIACOES_FILME = 20    # filmes com menos avaliações no treino ficam fora do catálogo do modelo
MIN_AVALIACOES_USUARIO = 5   # usuários com menos histórico não são avaliados (seria cold start)
K = 10                       # tamanho da lista recomendada


def carregar():
    """Lê avaliações e filmes. A data vem do timestamp do tweet."""
    av = pd.read_csv(PASTA / "avaliacoes.csv.gz")
    av["data"] = pd.to_datetime(av["ts"], unit="s")
    filmes = pd.read_csv(PASTA / "filmes.csv.gz").set_index("filme")
    return av, filmes


@dataclass
class Divisao:
    """Treino (tudo antes do corte) e alvo (filmes novos para cada usuário dentro da janela de teste)."""
    treino: pd.DataFrame        # avaliações do treino dentro do catálogo
    itens: np.ndarray           # ids dos filmes do catálogo (coluna j da matriz = itens[j])
    usuarios: np.ndarray        # ids dos usuários com histórico suficiente
    X: sp.csr_matrix            # matriz usuário × filme binária do treino
    alvo: pd.Series             # usuário -> conjunto de filmes assistidos no teste (só os inéditos para ele)
    linhas: np.ndarray          # linha de X de cada usuário do alvo, na mesma ordem
    corte: pd.Timestamp
    fora_catalogo: float        # fração das avaliações do teste em filmes que o modelo nem conhece


def dividir(av, corte, fim) -> Divisao:
    """Divisão temporal: o modelo vê só o passado e é avaliado no que cada usuário assistiu depois."""
    corte, fim = pd.Timestamp(corte), pd.Timestamp(fim)
    passado = av[av["data"] < corte]
    futuro = av[(av["data"] >= corte) & (av["data"] < fim)]

    cont_f = passado["filme"].value_counts()
    itens = cont_f[cont_f >= MIN_AVALIACOES_FILME].index.to_numpy()
    cont_u = passado["usuario"].value_counts()
    usuarios = cont_u[cont_u >= MIN_AVALIACOES_USUARIO].index.to_numpy()

    treino = passado[passado["filme"].isin(itens) & passado["usuario"].isin(usuarios)]
    col = pd.Series(np.arange(len(itens)), index=itens)
    lin = pd.Series(np.arange(len(usuarios)), index=usuarios)
    X = sp.csr_matrix((np.ones(len(treino), dtype=np.float32),
                       (lin[treino["usuario"]].to_numpy(), col[treino["filme"]].to_numpy())),
                      shape=(len(usuarios), len(itens)))
    X.data[:] = 1  # avaliações repetidas do mesmo filme contam uma vez

    # Alvo: o que usuários conhecidos assistiram no teste e ainda não tinham assistido antes
    futuro = futuro[futuro["usuario"].isin(usuarios)]
    ja_viu = passado[["usuario", "filme"]].drop_duplicates().assign(visto=True)
    futuro = futuro.merge(ja_viu, on=["usuario", "filme"], how="left")
    futuro = futuro[futuro["visto"].isna()]
    alvo = futuro.groupby("usuario")["filme"].apply(set)
    return Divisao(treino, itens, usuarios, X, alvo, lin[alvo.index].to_numpy(), corte,
                   fora_catalogo=float((~futuro["filme"].isin(itens)).mean()))


# ---------------------------------------------------------------- modelos
# Todos devolvem uma matriz de pontuações (usuários do alvo × filmes do catálogo).

def popularidade(d: Divisao, dias=None):
    """Mesma lista para todos: os filmes mais assistidos (no histórico todo ou nos últimos `dias`)."""
    t = d.treino if dias is None else d.treino[d.treino["data"] >= d.corte - pd.Timedelta(days=dias)]
    cont = t["filme"].value_counts().reindex(d.itens, fill_value=0).to_numpy(np.float32)
    return np.tile(cont, (len(d.linhas), 1))


def matriz_com_decaimento(d: Divisao, tau=None):
    """Matriz usuário × filme em que cada interação pesa exp(-idade / tau): o recente pesa mais."""
    if tau is None:
        return d.X
    idade = (d.corte - d.treino["data"]).dt.days.to_numpy()
    lin = pd.Series(np.arange(len(d.usuarios)), index=d.usuarios)[d.treino["usuario"]].to_numpy()
    col = pd.Series(np.arange(len(d.itens)), index=d.itens)[d.treino["filme"]].to_numpy()
    W = sp.coo_matrix((np.exp(-idade / tau).astype(np.float32), (lin, col)), shape=d.X.shape).tocsr()
    W.sum_duplicates()
    return W


def ease(W, lam):
    """EASE (Steck, 2019): modelo linear item × item com solução fechada.

    Resolve min ||W - W B||² + lam ||B||² com diagonal de B zerada (um filme não pode recomendar a si mesmo):
    B = I - P · diag(1 / diag(P)), em que P = (WᵀW + lam·I)⁻¹.
    """
    G = (W.T @ W).toarray().astype(np.float64)
    G[np.diag_indices_from(G)] += lam
    P = np.linalg.inv(G)
    B = -P / np.diag(P)
    np.fill_diagonal(B, 0.0)
    return B.astype(np.float32)


def pontuar_ease(d: Divisao, lam, tau=None):
    """Treina o EASE (com ou sem decaimento temporal) e pontua os usuários do alvo."""
    W = matriz_com_decaimento(d, tau)
    return np.asarray(W[d.linhas] @ ease(W, lam))


def item_knn(d: Divisao):
    """Vizinhança item × item por cosseno: 'quem viu este filme também viu…'."""
    G = (d.X.T @ d.X).toarray()
    n = np.sqrt(np.diag(G)) + 1e-9
    S = G / np.outer(n, n)
    np.fill_diagonal(S, 0.0)
    return np.asarray(d.X[d.linhas] @ S)


def hibrido(score_pessoal, score_pop, alfa):
    """Mistura linear de duas pontuações normalizadas por usuário para a faixa 0–1."""
    def norm(M):
        M = M - M.min(axis=1, keepdims=True)
        return M / (M.max(axis=1, keepdims=True) + 1e-9)
    return alfa * norm(np.log1p(score_pop)) + (1 - alfa) * norm(score_pessoal)


# ---------------------------------------------------------------- avaliação

def top_k(d: Divisao, S, k=K):
    """Índices dos k filmes de maior pontuação por usuário, sem repetir o que ele já assistiu."""
    S = np.array(S, dtype=np.float32, copy=True)
    S[d.X[d.linhas].nonzero()] = -np.inf
    parcial = np.argpartition(-S, k, axis=1)[:, :k]
    ordem = np.take_along_axis(S, parcial, axis=1).argsort(axis=1)[:, ::-1]
    return np.take_along_axis(parcial, ordem, axis=1)


def metricas(d: Divisao, top, k=K):
    """Recall@k, NDCG@k e cobertura de catálogo. Também devolve o recall por usuário."""
    rec, ndcg = [], []
    desconto = 1 / np.log2(np.arange(2, k + 2))
    for idx, usuario in enumerate(d.alvo.index):
        certos = d.alvo[usuario]
        acertos = np.isin(d.itens[top[idx]], list(certos))
        rec.append(acertos.sum() / min(k, len(certos)))
        ideal = desconto[:min(k, len(certos))].sum()
        ndcg.append((acertos * desconto).sum() / ideal)
    rec = np.array(rec)
    return {"recall": rec.mean(), "ndcg": float(np.mean(ndcg)),
            "cobertura": len(np.unique(top)) / len(d.itens)}, rec
