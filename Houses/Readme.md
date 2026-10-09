# House Prices — Advanced Regression Techniques (Kaggle)

**Objetivo:** prever o preço de venda de casas em Ames (Iowa, EUA) a partir de 79 características do imóvel.

## Solução

1. **Valores ausentes:** frente do lote preenchida pela média, área de alvenaria com zero e ano da garagem pela média.
2. **Variáveis categóricas:** one-hot encoding, com alinhamento das colunas entre treino e teste.
3. **Validação:** separação de 20% do treino.
4. **Modelo:** XGBoost Regressor, com dados padronizados.

**Resultado:** RMSLE de **0,142** na validação (a métrica oficial da competição).

## Próximos passos

- Transformar o preço com log antes de treinar e tratar a assimetria das variáveis.
- Validação cruzada e ajuste de hiperparâmetros.

**Stack:** Python · pandas · scikit-learn · XGBoost
