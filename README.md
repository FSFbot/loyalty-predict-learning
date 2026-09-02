# Loyalty Predict Learning

Reconstrução educacional, do zero, do projeto Loyalty Predict apresentado por
Téo Me Why.

O projeto original será utilizado somente como referência conceitual e
arquitetural. Todo o código desta versão será desenvolvido e discutido em
blocos incrementais.

## Problema de negócio

Uma comunidade oferece diferentes formas de participação, como cursos e um
sistema de pontos. Com o passar do tempo, o nível de engajamento de cada
usuário pode aumentar, permanecer estável ou diminuir.

Queremos transformar os registros dessas interações em informação útil para
acompanhar o ciclo de vida dos usuários e identificar mudanças de engajamento.

## Objetivo

Construir uma solução de Data Science capaz de:

1. descrever o comportamento geral da comunidade;
2. acompanhar o ciclo de vida dos usuários;
3. identificar grupos com comportamentos semelhantes;
4. prever perda ou ganho de engajamento;
5. registrar e comparar experimentos de Machine Learning.

## Pergunta principal

Com base no histórico conhecido de um usuário, conseguimos prever se o seu
engajamento irá diminuir ou aumentar em um período futuro?

## Fontes de dados

Inicialmente, trabalharemos com dados públicos relacionados a:

- sistema de pontos;
- plataforma de cursos.

## Etapas previstas

1. aquisição dos dados;
2. entendimento e validação dos dados;
3. definição da unidade de análise e da variável-alvo;
4. construção das variáveis explicativas;
5. criação das feature stores;
6. treinamento e avaliação dos modelos;
7. registro dos experimentos com MLflow;
8. criação de uma interface de inferência.

## Decisões ainda em aberto

Antes da modelagem, precisaremos determinar:

- o que representa uma observação;
- como medir engajamento;
- qual será a janela histórica;
- qual será o horizonte futuro;
- como evitar vazamento de informação;
- quais métricas representarão sucesso técnico e de negócio.

## Referência

- [Projeto Loyalty Predict — Téo Me Why](https://github.com/TeoMeWhy/loyalty-predict)