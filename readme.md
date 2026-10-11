# Machine Learning: um projeto por tipo de problema

Projetos de ponta a ponta com dados públicos. Cada pasta traz um notebook executado, um README com a resposta de negócio no início e gráficos no mesmo padrão visual.

| Projeto | Tipo de problema | Técnicas | Resultado |
|---|---|---|---|
| [Churn em telecom](Churn_Telecom) | Classificação + decisão | Regressão logística, gradient boosting, calibração, Kaplan-Meier, curva de lucro | Abordar só quem tem valor esperado positivo quase dobra o lucro da campanha |
| [NLP em português](NLP_Avaliacoes) | Classificação de texto | TF-IDF, regressão logística, Naive Bayes, divisão temporal, temas por regras | Triagem pega 90% das reclamações revisando só 27% das avaliações; 36% das reclamações são sobre entrega |
| [Segmentação de clientes](Segmentacao_Clientes) | Clusterização + análise de coortes | RFM, K-means com escolha de k por silhueta e estabilidade, coortes, intervalos de Wilson | 98% compram uma vez só; insatisfeitos e frete pesado voltam um terço menos |
| [Sistema de recomendação](Sistema_Recomendacao) | Recomendação top-N | EASE implementado do zero, decaimento temporal, híbrido, bootstrap pareado | Híbrido supera a lista do que está em alta e mostra 16× mais do catálogo |

## Competições do Kaggle

Em [`kaggle/`](kaggle): exercícios de competição, com foco em pré-processamento e validação.

| Competição | Tipo | Resultado |
|---|---|---|
| [House Prices](kaggle/House_Prices) | Regressão | RMSLE de 0,142 na validação |
| [Playground S4E6](kaggle/Playground_S4E6) | Classificação multiclasse | Acurácia de 82,8% na validação |
| [Spaceship Titanic](kaggle/Spaceship_Titanic) | Classificação binária | Acurácia de 81,1% em validação cruzada, contra 78,5% da versão anterior |

## Algoritmos e onde aparecem

Cada algoritmo aparece dentro de um problema real, comparado com uma linha de base e avaliado com validação adequada ao caso.

| Algoritmo | Onde aparece | Para quê |
|---|---|---|
| Regressão linear | [Airbnb em Buenos Aires](https://github.com/cezarcobaink/DataScience/tree/main/Buenos_Aires) · [diagnóstico de resíduos](https://github.com/cezarcobaink/estatistica-visual/blob/main/graficos/41_residuos.py) | Linha de base de preço; checar suposições do modelo |
| Regressão logística | [Churn](Churn_Telecom) · [NLP](NLP_Avaliacoes) · [fraude](https://github.com/cezarcobaink/DataScience/tree/main/Fraudes_Cartao_de_credito) · [forest plot](https://github.com/cezarcobaink/estatistica-visual/blob/main/graficos/37_forest_plot.py) | Probabilidade calibrada; no forest plot, implementada do zero (Newton-Raphson) |
| Regressão de Poisson | [Previsão de demanda Olist](https://github.com/cezarcobaink/previsao-demanda-olist) | Contagem de pedidos por dia |
| Naive Bayes | [NLP](NLP_Avaliacoes) | Linha de base para classificação de texto |
| Árvore de decisão | [Fraude](https://github.com/cezarcobaink/DataScience/tree/main/Fraudes_Cartao_de_credito) | Classificador com classes desbalanceadas (undersampling e SMOTE) |
| Random Forest | [Spaceship Titanic](kaggle/Spaceship_Titanic) · [Playground S4E6](kaggle/Playground_S4E6) | Classificação tabular |
| Gradient boosting | [Churn](Churn_Telecom) · [fraude](https://github.com/cezarcobaink/DataScience/tree/main/Fraudes_Cartao_de_credito) · [Airbnb](https://github.com/cezarcobaink/DataScience/tree/main/Buenos_Aires) · [previsão Olist](https://github.com/cezarcobaink/previsao-demanda-olist) · [Spaceship Titanic](kaggle/Spaceship_Titanic) · [recompra](Segmentacao_Clientes) | Classificação, regressão e contagem (perda de Poisson) |
| XGBoost | [House Prices](kaggle/House_Prices) | Regressão tabular |
| K-means | [Segmentação de clientes](Segmentacao_Clientes) | Segmentos de comportamento, com escolha de k por silhueta e estabilidade |
| Agrupamento hierárquico | [Matriz de correlação agrupada](https://github.com/cezarcobaink/estatistica-visual/blob/main/graficos/24_correlacao_agrupada.py) | Ordenar variáveis parecidas |
| PCA | [Fraude](https://github.com/cezarcobaink/DataScience/tree/main/Fraudes_Cartao_de_credito) | Visualizar as classes em duas dimensões |
| TF-IDF | [NLP](NLP_Avaliacoes) | Representar texto em português |
| Vizinhança item × item e EASE | [Sistema de recomendação](Sistema_Recomendacao) | Recomendação; EASE implementado do zero, com decaimento temporal |
| Kaplan-Meier | [Churn](Churn_Telecom) | Tempo até o cancelamento, implementado do zero |
| Séries temporais | [Previsão Olist](https://github.com/cezarcobaink/previsao-demanda-olist) · [inferência causal](https://github.com/cezarcobaink/inferencia-causal) · [decomposição](https://github.com/cezarcobaink/estatistica-visual/blob/main/graficos/33_decomposicao.py) | Backtest em janelas móveis, série temporal interrompida, sazonalidade |
| Controle sintético e testes A/B | [Inferência causal](https://github.com/cezarcobaink/inferencia-causal) | Efeito causal com placebos, CUPED e correção de múltiplas comparações |

Ainda não cobertos, e onde devem entrar: DBSCAN e Isolation Forest (detecção de anomalias em gastos públicos, próximo projeto), ARIMA (como concorrente no backtest da previsão Olist), SVM e redes neurais.

`comum/estilo.py` concentra o padrão visual compartilhado pelos projetos.

Outros projetos: [Estatística visual](https://github.com/cezarcobaink/estatistica-visual) · [Inferência causal](https://github.com/cezarcobaink/inferencia-causal) · [Previsão de demanda — Olist](https://github.com/cezarcobaink/previsao-demanda-olist) · [DataScience](https://github.com/cezarcobaink/DataScience) · Portfólio: [cezarcobaink.github.io/WebSite](https://cezarcobaink.github.io/WebSite/)
