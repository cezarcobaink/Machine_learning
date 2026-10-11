"""Gera pedidos_olist.csv.gz (um registro por pedido entregue) a partir dos CSVs originais da Olist.

Uso (com os CSVs do Kaggle numa pasta):
    python preparar_dados.py caminho/para/csvs

O arquivo gerado já está no repositório; este script só documenta e reproduz a preparação.
Dados: Brazilian E-Commerce Public Dataset by Olist (CC BY-NC-SA 4.0).
"""
import sys
from pathlib import Path

import pandas as pd

PASTA = Path(__file__).resolve().parent


def preparar(origem: Path) -> pd.DataFrame:
    """Junta pedidos, clientes, pagamentos, itens, produtos e avaliações num registro por pedido."""
    pedidos = pd.read_csv(origem / "olist_orders_dataset.csv", parse_dates=[
        "order_purchase_timestamp", "order_delivered_customer_date", "order_estimated_delivery_date"])
    clientes = pd.read_csv(origem / "olist_customers_dataset.csv")
    pagamentos = pd.read_csv(origem / "olist_order_payments_dataset.csv")
    itens = pd.read_csv(origem / "olist_order_items_dataset.csv")
    produtos = pd.read_csv(origem / "olist_products_dataset.csv")
    avaliacoes = pd.read_csv(origem / "olist_order_reviews_dataset.csv")

    # Só pedidos entregues: cancelados e indisponíveis não representam uma compra concluída
    pedidos = pedidos[pedidos.order_status == "delivered"].merge(clientes, on="customer_id")

    # Valor pago e número máximo de parcelas por pedido (um pedido pode ter vários pagamentos)
    pag = pagamentos.groupby("order_id").agg(valor=("payment_value", "sum"),
                                            parcelas=("payment_installments", "max"))

    # Itens: quantidade, frete e a categoria do item mais caro (representa o pedido)
    itens = itens.merge(produtos[["product_id", "product_category_name"]], on="product_id", how="left")
    itens = itens.sort_values("price", ascending=False)
    it = itens.groupby("order_id").agg(itens=("order_item_id", "size"), frete=("freight_value", "sum"),
                                       categoria=("product_category_name", "first"))

    nota = avaliacoes.groupby("order_id").review_score.mean()

    df = pd.DataFrame({
        "pedido": pedidos.order_id,
        "cliente": pedidos.customer_unique_id,
        "data": pedidos.order_purchase_timestamp,
        "estado": pedidos.customer_state,
        "atraso_dias": (pedidos.order_delivered_customer_date - pedidos.order_estimated_delivery_date).dt.days,
        "dias_entrega": (pedidos.order_delivered_customer_date - pedidos.order_purchase_timestamp).dt.days,
    }).set_index("pedido").join(pag).join(it)
    df["nota"] = nota.reindex(df.index).values
    df = df.dropna(subset=["valor", "itens"]).reset_index()

    # Identificadores longos viram códigos curtos (o arquivo fica menor e continua anônimo)
    df["cliente"] = pd.factorize(df["cliente"])[0]
    df["pedido"] = pd.factorize(df["pedido"])[0]
    df["categoria"] = df["categoria"].fillna("sem_categoria")
    return df.sort_values("data")


if __name__ == "__main__":
    origem = Path(sys.argv[1]) if len(sys.argv) > 1 else PASTA
    tabela = preparar(origem)
    tabela.to_csv(PASTA / "pedidos_olist.csv.gz", index=False, float_format="%.2f")
    print(f"{len(tabela):,} pedidos de {tabela.cliente.nunique():,} clientes")
