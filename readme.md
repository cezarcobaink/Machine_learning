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

`comum/estilo.py` concentra o padrão visual compartilhado pelos projetos.

Outros projetos: [Estatística visual](https://github.com/cezarcobaink/estatistica-visual) · [Inferência causal](https://github.com/cezarcobaink/inferencia-causal) · [Previsão de demanda — Olist](https://github.com/cezarcobaink/previsao-demanda-olist) · [DataScience](https://github.com/cezarcobaink/DataScience) · Portfólio: [cezarcobaink.github.io/WebSite](https://cezarcobaink.github.io/WebSite/)
