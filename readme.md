# Machine Learning: um projeto por tipo de problema

Projetos de ponta a ponta com dados públicos. Cada pasta traz um notebook executado, um README com a resposta de negócio no início e gráficos no mesmo padrão visual.

- **[Churn em telecom](Churn_Telecom)** · *Classificação + decisão*<br>Abordar só quem tem valor esperado positivo quase dobra o lucro da campanha.<br>Técnicas: regressão logística, gradient boosting, calibração, Kaplan-Meier, curva de lucro.
- **[NLP em português](NLP_Avaliacoes)** · *Classificação de texto*<br>Triagem pega 90% das reclamações revisando só 27% das avaliações; 36% das reclamações são sobre entrega.<br>Técnicas: TF-IDF, regressão logística, Naive Bayes, divisão temporal, temas por regras.
- **[Segmentação de clientes](Segmentacao_Clientes)** · *Clusterização + análise de coortes*<br>98% compram uma vez só; insatisfeitos e frete pesado voltam um terço menos.<br>Técnicas: RFM, K-means com escolha de k por silhueta e estabilidade, coortes, intervalos de Wilson.
- **[Sistema de recomendação](Sistema_Recomendacao)** · *Recomendação top-N*<br>Híbrido supera a lista do que está em alta e mostra 16× mais do catálogo.<br>Técnicas: EASE implementado do zero, decaimento temporal, híbrido, bootstrap pareado.

## Competições do Kaggle

Em [`kaggle/`](kaggle): exercícios de competição, com foco em pré-processamento e validação.

| Competição | Tipo | Resultado |
|---|---|---|
| [House Prices](kaggle/House_Prices) | Regressão | RMSLE de 0,142 na validação |
| [Playground S4E6](kaggle/Playground_S4E6) | Classificação multiclasse | Acurácia de 82,8% na validação |
| [Spaceship Titanic](kaggle/Spaceship_Titanic) | Classificação binária | Acurácia de 81,1% em validação cruzada, contra 78,5% da versão anterior |

## Algoritmos e onde aparecem

Cada algoritmo aparece dentro de um problema real, comparado com uma linha de base e avaliado com validação adequada ao caso.

- **Regressão linear**: linha de base de preço; checar suposições do modelo. Em: [Airbnb em Buenos Aires](https://github.com/cezarcobaink/DataScience/tree/main/Buenos_Aires) · [diagnóstico de resíduos](https://github.com/cezarcobaink/estatistica-visual/blob/main/graficos/41_residuos.py)
- **Regressão logística**: probabilidade calibrada; no forest plot, implementada do zero (Newton-Raphson). Em: [Churn](Churn_Telecom) · [NLP](NLP_Avaliacoes) · [fraude](https://github.com/cezarcobaink/DataScience/tree/main/Fraudes_Cartao_de_credito) · [forest plot](https://github.com/cezarcobaink/estatistica-visual/blob/main/graficos/37_forest_plot.py)
- **Regressão de Poisson**: contagem de pedidos por dia. Em: [Previsão de demanda Olist](https://github.com/cezarcobaink/previsao-demanda-olist)
- **Naive Bayes**: linha de base para classificação de texto. Em: [NLP](NLP_Avaliacoes)
- **Árvore de decisão**: classificador com classes desbalanceadas (undersampling e SMOTE). Em: [Fraude](https://github.com/cezarcobaink/DataScience/tree/main/Fraudes_Cartao_de_credito)
- **Random Forest**: classificação tabular. Em: [Spaceship Titanic](kaggle/Spaceship_Titanic) · [Playground S4E6](kaggle/Playground_S4E6)
- **Gradient boosting**: classificação, regressão e contagem (perda de Poisson). Em: [Churn](Churn_Telecom) · [fraude](https://github.com/cezarcobaink/DataScience/tree/main/Fraudes_Cartao_de_credito) · [Airbnb](https://github.com/cezarcobaink/DataScience/tree/main/Buenos_Aires) · [previsão Olist](https://github.com/cezarcobaink/previsao-demanda-olist) · [Spaceship Titanic](kaggle/Spaceship_Titanic) · [recompra](Segmentacao_Clientes)
- **XGBoost**: regressão tabular. Em: [House Prices](kaggle/House_Prices)
- **K-means**: segmentos de comportamento, com escolha de k por silhueta e estabilidade. Em: [Segmentação de clientes](Segmentacao_Clientes)
- **Agrupamento hierárquico**: ordenar variáveis parecidas. Em: [Matriz de correlação agrupada](https://github.com/cezarcobaink/estatistica-visual/blob/main/graficos/24_correlacao_agrupada.py)
- **PCA**: visualizar as classes em duas dimensões. Em: [Fraude](https://github.com/cezarcobaink/DataScience/tree/main/Fraudes_Cartao_de_credito)
- **TF-IDF**: representar texto em português. Em: [NLP](NLP_Avaliacoes)
- **Vizinhança item × item e EASE**: recomendação; EASE implementado do zero, com decaimento temporal. Em: [Sistema de recomendação](Sistema_Recomendacao)
- **Kaplan-Meier**: tempo até o cancelamento, implementado do zero. Em: [Churn](Churn_Telecom)
- **Séries temporais**: backtest em janelas móveis, série temporal interrompida, sazonalidade. Em: [Previsão Olist](https://github.com/cezarcobaink/previsao-demanda-olist) · [inferência causal](https://github.com/cezarcobaink/inferencia-causal) · [decomposição](https://github.com/cezarcobaink/estatistica-visual/blob/main/graficos/33_decomposicao.py)
- **Controle sintético e testes A/B**: efeito causal com placebos, CUPED e correção de múltiplas comparações. Em: [Inferência causal](https://github.com/cezarcobaink/inferencia-causal)

Ainda não cobertos, e onde devem entrar: DBSCAN e Isolation Forest (detecção de anomalias em gastos públicos, próximo projeto), ARIMA (como concorrente no backtest da previsão Olist), SVM e redes neurais.

`comum/estilo.py` concentra o padrão visual compartilhado pelos projetos.

Outros projetos: [Estatística visual](https://github.com/cezarcobaink/estatistica-visual) · [Inferência causal](https://github.com/cezarcobaink/inferencia-causal) · [Previsão de demanda — Olist](https://github.com/cezarcobaink/previsao-demanda-olist) · [DataScience](https://github.com/cezarcobaink/DataScience) · Portfólio: [cezarcobaink.github.io/WebSite](https://cezarcobaink.github.io/WebSite/)
