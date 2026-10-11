"""Engenharia de variáveis e pipeline do Spaceship Titanic.

Toda a imputação e a codificação ficam dentro do pipeline do scikit-learn. Assim, em cada dobra da validação
cruzada, médias e categorias são aprendidas só com os dados de treino daquela dobra (sem vazamento).
"""
import numpy as np
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

GASTOS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
NUMERICAS = ["Age", "tam_grupo", "num_cabine", "log_gasto", "sem_gasto", "CryoSleep", "VIP"] + [f"log_{g}" for g in GASTOS]
CATEGORICAS = ["HomePlanet", "Destination", "deck", "lado"]


def criar_variaveis(df: pd.DataFrame) -> pd.DataFrame:
    """Extrai sinal do ID, da cabine e dos gastos. Não usa o alvo, então vale igual para treino e teste."""
    d = df.copy()

    # PassengerId = gggg_pp: os 4 primeiros dígitos são o grupo de viagem (famílias, amigos)
    d["grupo"] = d["PassengerId"].str[:4]
    d["tam_grupo"] = d.groupby("grupo")["PassengerId"].transform("size")

    # Cabin = deck/número/lado (P = bombordo, S = estibordo)
    cabine = d["Cabin"].str.split("/", expand=True)
    d["deck"], d["num_cabine"], d["lado"] = cabine[0], pd.to_numeric(cabine[1]), cabine[2]

    # Quem está em sono criogênico não consome: gasto ausente nesses casos é zero, não "desconhecido"
    em_crio = d["CryoSleep"].eq(True)
    d.loc[em_crio, GASTOS] = d.loc[em_crio, GASTOS].fillna(0)

    # Gastos são muito assimétricos (muitos zeros e poucos valores altos): log(1 + x)
    d["gasto_total"] = d[GASTOS].sum(axis=1, min_count=1)
    d["sem_gasto"] = (d["gasto_total"] == 0).astype(float).where(d["gasto_total"].notna())
    d["log_gasto"] = np.log1p(d["gasto_total"])
    for g in GASTOS:
        d[f"log_{g}"] = np.log1p(d[g])

    # Booleanos com ausentes viram 0/1/NaN; o NaN é imputado dentro do pipeline
    for col in ["CryoSleep", "VIP"]:
        d[col] = d[col].map({True: 1.0, False: 0.0})
    # Nome e ID ficam de fora: são identificadores, não características
    return d


def pre_processador():
    """Imputação + padronização (numéricas) e imputação + one-hot (categóricas)."""
    numericas = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    categoricas = make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore"))
    return make_column_transformer((numericas, NUMERICAS), (categoricas, CATEGORICAS))
