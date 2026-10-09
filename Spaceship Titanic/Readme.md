# Spaceship Titanic (Kaggle)

**Objetivo:** prever quais passageiros da nave foram transportados para outra dimensão, a partir de dados como planeta de origem, cabine, idade e gastos a bordo. São 8.693 passageiros no treino.

## Solução

1. **Valores ausentes:** média para variáveis numéricas e moda para categóricas.
2. **Codificação:** one-hot encoding das variáveis categóricas.
3. **Validação:** separação de 20% do treino.
4. **Modelo:** Random Forest Classifier.

**Resultado:** acurácia de **78%** na validação.

## O que eu mudaria

A codificação aplicou one-hot também ao ID e ao nome do passageiro, o que cria milhares de colunas sem valor preditivo. A próxima versão vai:
- extrair o grupo do ID e o deck e o lado da cabine, que têm sinal real;
- descartar o nome;
- somar os gastos a bordo em uma variável de consumo total.

**Stack:** Python · pandas · scikit-learn
