"""Funções de preparação e análise usadas no notebook de segmentação de clientes da Olist."""
from pathlib import Path

import numpy as np
import pandas as pd

PASTA = Path(__file__).resolve().parent
DATA_REF = pd.Timestamp("2018-09-01")   # "hoje" da análise: logo após o último pedido da base
JANELA_RECOMPRA = 180                   # dias para considerar que o cliente voltou

# Variáveis de comportamento usadas na clusterização (todas conhecidas na própria compra)
VARIAVEIS = ["ticket", "parcelas", "frete_pct", "itens", "nota"]

CATEGORIA_PT = {
    "cama_mesa_banho": "Cama, mesa e banho", "beleza_saude": "Beleza e saúde",
    "esporte_lazer": "Esporte e lazer", "moveis_decoracao": "Móveis e decoração",
    "informatica_acessorios": "Informática", "utilidades_domesticas": "Utilidades domésticas",
    "relogios_presentes": "Relógios e presentes", "telefonia": "Telefonia",
    "ferramentas_jardim": "Ferramentas e jardim", "brinquedos": "Brinquedos",
    "automotivo": "Automotivo", "perfumaria": "Perfumaria", "cool_stuff": "Cool stuff",
    "bebes": "Bebês", "eletronicos": "Eletrônicos", "papelaria": "Papelaria",
    "fashion_bolsas_e_acessorios": "Bolsas e acessórios", "pet_shop": "Pet shop",
    "moveis_escritorio": "Móveis de escritório", "consoles_games": "Games",
    "malas_acessorios": "Malas", "construcao_ferramentas_construcao": "Construção",
    "eletrodomesticos": "Eletrodomésticos", "instrumentos_musicais": "Instrumentos musicais",
    "eletroportateis": "Eletroportáteis", "casa_construcao": "Casa e construção",
    "livros_interesse_geral": "Livros", "alimentos": "Alimentos",
}


def carregar_pedidos() -> pd.DataFrame:
    """Lê a tabela de pedidos entregues (gerada por preparar_dados.py)."""
    return pd.read_csv(PASTA / "pedidos_olist.csv.gz", parse_dates=["data"])


def tabela_clientes(pedidos: pd.DataFrame) -> pd.DataFrame:
    """Um registro por cliente com as medidas de RFM e de comportamento de compra."""
    cli = pedidos.groupby("cliente").agg(
        primeira=("data", "min"), ultima=("data", "max"),
        frequencia=("data", lambda s: s.dt.normalize().nunique()),  # dias distintos de compra
        valor=("valor", "sum"), ticket=("valor", "mean"), parcelas=("parcelas", "mean"),
        frete=("frete", "sum"), itens=("itens", "mean"), nota=("nota", "mean"),
    )
    # Pedidos no mesmo dia costumam ser um carrinho dividido, não uma volta do cliente,
    # por isso a frequência conta dias distintos, e não pedidos.
    cli["recencia"] = (DATA_REF - cli["ultima"]).dt.days
    cli["frete_pct"] = (cli["frete"] / cli["valor"]).clip(0, 1)
    cli["nota"] = cli["nota"].fillna(cli["nota"].median())  # 0,7% sem avaliação
    return cli


def primeira_compra(pedidos: pd.DataFrame) -> pd.DataFrame:
    """Primeira compra de cada cliente e se ele voltou a comprar em outro dia dentro da janela.

    Só entram clientes com a janela inteira observada (primeira compra até DATA_REF - JANELA),
    senão os clientes recentes pareceriam menos fiéis apenas por falta de tempo.
    """
    p = pedidos.sort_values("data").copy()
    p["dia"] = p["data"].dt.normalize()
    primeira = p.groupby("cliente").first()
    # Próximo dia de compra diferente do dia da primeira compra
    outras = p.merge(primeira[["dia"]], left_on="cliente", right_index=True, suffixes=("", "_1"))
    outras = outras[outras["dia"] > outras["dia_1"]]
    segunda = outras.groupby("cliente")["dia"].min()
    dias_ate_voltar = (segunda - primeira["dia"].reindex(segunda.index)).dt.days

    primeira["voltou"] = primeira.index.isin(dias_ate_voltar[dias_ate_voltar <= JANELA_RECOMPRA].index)
    primeira["dias_ate_voltar"] = dias_ate_voltar.reindex(primeira.index)
    primeira["frete_pct"] = (primeira["frete"] / primeira["valor"]).clip(0, 1)
    limite = DATA_REF - pd.Timedelta(days=JANELA_RECOMPRA)
    return primeira[primeira["data"] < limite]


def recompra_por_coorte(pedidos: pd.DataFrame, meses: int = 6) -> pd.DataFrame:
    """% acumulado de clientes de cada coorte mensal que voltaram a comprar até o mês k."""
    p = pedidos.copy()
    p["mes"] = p["data"].dt.to_period("M")
    p["dia"] = p["data"].dt.normalize()
    inicio = p.groupby("cliente").agg(coorte=("mes", "min"), dia1=("dia", "min"))
    p = p.join(inicio, on="cliente")
    volta = p[p["dia"] > p["dia1"]].groupby("cliente")["mes"].min()
    inicio["mes_volta"] = volta
    inicio["k"] = (inicio["mes_volta"] - inicio["coorte"]).apply(lambda x: x.n if pd.notna(x) else np.nan)

    linhas = {}
    ultimo_mes = DATA_REF.to_period("M") - 1
    for coorte, g in inicio.groupby("coorte"):
        if coorte < pd.Period("2017-01", "M"):
            continue  # 2016 tem poucas centenas de clientes
        linha = {}
        for k in range(1, meses + 1):
            if coorte + k > ultimo_mes:
                break  # mês ainda não observado por inteiro
            linha[k] = (g["k"] <= k).mean()
        linhas[str(coorte)] = linha
    return pd.DataFrame(linhas).T


def segmentos_rfm(cli: pd.DataFrame) -> pd.Series:
    """Segmentos por regras de RFM.

    Como 97% dos clientes compram uma única vez, a frequência vira um corte binário
    (1 compra × 2+), e o resto da base é dividido por recência e valor (quintis).
    """
    r = pd.qcut(cli["recencia"].rank(method="first"), 5, labels=[5, 4, 3, 2, 1]).astype(int)  # 5 = mais recente
    m = pd.qcut(cli["valor"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    seg = np.select(
        [cli["frequencia"] >= 2, (r >= 4) & (m >= 4), (r <= 2) & (m >= 4), r >= 4],
        ["Recorrentes", "Alto valor, recentes", "Alto valor, adormecidos", "Baixo valor, recentes"],
        default="Baixo valor, adormecidos",
    )
    return pd.Series(seg, index=cli.index, name="segmento_rfm")


def taxa_wilson(sucessos, total, z: float = 1.96):
    """Proporção e intervalo de confiança de Wilson (bom para taxas pequenas como 3%)."""
    sucessos, total = np.asarray(sucessos, float), np.asarray(total, float)
    p = sucessos / total
    centro = (p + z**2 / (2 * total)) / (1 + z**2 / total)
    margem = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / (1 + z**2 / total)
    return p, centro - margem, centro + margem


def nomear_clusters(medias: pd.DataFrame) -> dict:
    """Dá nomes de negócio aos clusters a partir das médias de cada um.

    As regras olham qual cluster é extremo em cada variável, então os nomes não dependem
    da numeração arbitrária que o K-means devolve.
    """
    nomes, livres = {}, list(medias.index)

    def escolher(coluna, maior, nome):
        alvo = medias.loc[livres, coluna].idxmax() if maior else medias.loc[livres, coluna].idxmin()
        nomes[alvo] = nome
        livres.remove(alvo)

    escolher("nota", False, "Insatisfeitos")
    escolher("itens", True, "Cesta com vários itens")
    escolher("parcelas", True, "Parcelado, ticket alto")
    escolher("frete_pct", True, "Ticket baixo, frete pesado")
    for resto in livres:
        nomes[resto] = "Compra padrão"
    return nomes
