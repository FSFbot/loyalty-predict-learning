# Loyalty Predict Learning

Reconstrução educacional do projeto Loyalty Predict, apresentado por
Téo Me Why.

O desenvolvimento acontece em etapas, com estudo das decisões,
implementação manual dos arquivos e validação dos resultados.

## Problema de negócio

Uma comunidade possui diferentes formas de participação, como transações
em um sistema de pontos e atividades em uma plataforma de cursos.

O objetivo geral é compreender esse comportamento e acompanhar mudanças
de engajamento.

A primeira implementação de Machine Learning responde à pergunta:

> Com base no histórico disponível até um mês de referência, o cliente
> terá alguma transação no sistema de pontos no mês seguinte?

Essa previsão representa atividade futura no sistema de pontos.
Ela não mede, sozinha, todas as formas de engajamento da comunidade.

## Fontes de dados

São utilizados dois datasets públicos do Kaggle:

- Sistema de pontos: `teocalvo/teomewhy-loyalty-system`.
- Plataforma de cursos: `teocalvo/teomewhy-education-platform`.

Os bancos são armazenados localmente:

```text
data/raw/loyalty/database.db
data/raw/education/database.db
```

Os dados e os registros locais do MLflow não são versionados no Git.

## Unidade de análise e alvo

Cada observação representa um cliente em um mês de referência.

O alvo é `target_active_next_month`:

- `1`: o cliente teve alguma transação no mês seguinte.
- `0`: o cliente não teve transações no mês seguinte.

Linhas de treinamento precisam de alvo conhecido.

Na preparação para inferência, o alvo pode ser desconhecido e não é
incluído no arquivo de características.

## Características

O modelo recebe 30 características numéricas, incluindo:

- quantidades de transações e movimentações de pontos;
- atividade em janelas recentes;
- tempo desde a última atividade;
- histórico acumulado de participação;
- atividades mensais e históricas na plataforma de cursos.

`customer_id` e `reference_month` identificam as observações e não são
utilizados como características do classificador.

## Divisão temporal

A avaliação usa meses de referência em ordem cronológica:

| Conjunto | Período definido |
|---|---|
| Treino | Até dezembro de 2025 |
| Validação | Janeiro a abril de 2026 |
| Teste | Depois de abril de 2026 |

No conjunto de dados avaliado durante o desenvolvimento, o teste contém
maio a julho de 2026.

O treino ajusta o modelo e os transformadores. A validação foi usada para
comparar limiares de decisão. O teste foi usado para avaliar a configuração
escolhida.

Um cliente pode aparecer em conjuntos diferentes, em meses diferentes.
A separação avalia períodos posteriores, não exclusivamente clientes novos.

## Modelo

O baseline utiliza `DummyClassifier` com a estratégia `most_frequent`.

A regressão logística utiliza um Pipeline com:

1. preenchimento de ausências pela mediana, com indicadores de ausência;
2. padronização com `StandardScaler`;
3. classificação com `LogisticRegression`.

Os transformadores são ajustados somente com o conjunto de treino.

O limiar de decisão escolhido foi `0.30`, por apresentar o maior F1
na validação entre os cortes investigados:

```text
0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70
```

A classe positiva representa atividade futura.

## Resultados registrados

Resultados da regressão logística com limiar `0.30`:

| Métrica | Validação | Teste |
|---|---:|---:|
| Precisão | 64,72% | 71,18% |
| Recall | 56,28% | 56,21% |
| F1 | 60,21% | 62,81% |
| ROC AUC | 90,40% | 89,94% |
| Average precision | 62,88% | 63,85% |

Esses resultados correspondem aos dados utilizados no desenvolvimento.
Atualizações dos datasets podem alterar períodos, quantidades e métricas.

O teste possui 16.446 observações cliente-mês. Das 580 observações com
atividade futura, o modelo identificou 326, deixando de identificar 254.
Também produziu 132 falsos positivos.

## Estrutura

```text
src/
├── engineering/   # Aquisição, inspeção e verificações das fontes
├── analytics/     # Construção das tabelas e características
└── modeling/      # Treinamento, avaliação, MLflow e inferência

data/
├── raw/           # Bancos obtidos das fontes
└── inference/     # Características e previsões geradas

mlruns/            # Experimentos e modelos locais do MLflow
```

## Ambiente de desenvolvimento

O projeto foi executado no Linux com Python 3.12.

Com Python 3.12 disponível, crie e ative o ambiente na raiz do projeto:

```bash
python3.12 -m venv .venv-py312
source .venv-py312/bin/activate
python -m pip install -r requirements.txt
```

Se o Python 3.12 foi instalado pelo `uv`, a criação pode ser feita com:

```bash
uv venv --python 3.12 --seed .venv-py312
source .venv-py312/bin/activate
uv pip install -r requirements.txt
```

Use uma das formas de criação. Não é necessário recriar um ambiente
que já esteja funcionando.

Em terminais novos, ative o ambiente antes de executar os módulos:

```bash
source .venv-py312/bin/activate
```

Todos os comandos seguintes devem ser executados na raiz do projeto.

## Aquisição e inspeção dos dados

Configure a autenticação exigida pela API do Kaggle no seu ambiente.

Baixe os datasets:

```bash
python -m src.engineering.get_data
```

Inspecione os períodos das tabelas:

```bash
python -m src.engineering.inspect_time_ranges
```

Verifique o mês de referência comum:

```bash
python -m src.engineering.source_coverage
```

A regra atual exclui o mês do evento mais recente de cada fonte e
seleciona o menor dos limites resultantes.

Na fonte educacional, o limite utiliza o evento mais recente entre
episódios completos, habilidades e recompensas.

Essa regra usa as datas dos eventos como referência. Ela não comprova
a completude da extração.

## Treinamento e avaliação

Execute cada etapa separadamente:

```bash
python -m src.modeling.modeling_dataset
python -m src.modeling.temporal_split
python -m src.modeling.baseline_model
python -m src.modeling.logistic_model
python -m src.modeling.threshold_analysis
python -m src.modeling.final_evaluation
```

Esses módulos reconstruem os dados necessários às suas respectivas
execuções.

A análise de limiares usa a validação. O limiar escolhido não deve ser
reajustado para melhorar os resultados do teste.

## Registro no MLflow

Registre o modelo e os resultados:

```bash
python -m src.modeling.track_experiment
```

A saída informa:

- Run ID da execução de treinamento;
- URI do modelo salvo;
- localização dos registros locais.

O registro inclui o Pipeline, parâmetros, métricas, assinatura do modelo
e o artefato `decision_config.json`.

Esse artefato contém o limiar, a classe positiva e as características
esperadas.

Para consultar a interface local:

```bash
mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
```

Abra http://127.0.0.1:5000 no navegador.

O Run ID identifica uma execução no armazenamento local. Ele não fica
disponível em outro computador apenas por clonar o repositório: é
necessário preservar os registros e artefatos ou registrar outra execução.

## Inferência completa

Substitua `SEU_RUN_ID` pelo identificador informado pelo MLflow:

```bash
python -m src.modeling.run_inference --run-id SEU_RUN_ID
```

O programa:

1. carrega o Pipeline e sua configuração;
2. verifica os limites temporais das fontes;
3. constrói as características históricas;
4. seleciona o mês de referência comum;
5. calcula probabilidades;
6. aplica o limiar salvo;
7. exporta características e previsões.

Cada processamento cria uma pasta própria:

```text
data/inference/runs/<execution_id>/
├── features.csv
└── predictions.csv
```

O identificador da inferência usa data e hora em UTC.
Ele é diferente do Run ID do treinamento.

O arquivo de previsões contém:

- `customer_id`;
- `reference_month`;
- `probability_active_next_month`;
- `predicted_active_next_month`;
- `decision_threshold`;
- `run_id`.

O mês previsto é o mês seguinte ao `reference_month`.

## Inferência por CSV

Também é possível fornecer um CSV já preparado:

```bash
python -m src.modeling.batch_inference \
  --run-id SEU_RUN_ID \
  --input data/inference/sample_features.csv \
  --output data/inference/sample_predictions.csv
```

O arquivo de entrada deve existir e conter os identificadores e todas as
características esperadas, calculadas com as definições do treinamento.

O alvo futuro não é necessário.

Para exportar separadamente as características do mês comum:

```bash
python -m src.modeling.export_inference_features
```

A saída é `data/inference/latest_features.csv`.

## Interface local

Com o ambiente ativo, execute na raiz do projeto:

```bash
python -m streamlit run app.py --server.address 127.0.0.1
```

Abra http://127.0.0.1:8501, informe um Run ID disponível no MLflow
local e clique em **Gerar previsões**.

A interface mostra o mês de referência, o mês previsto, o limiar,
as quantidades e a tabela de previsões. O resultado pode ser baixado
em CSV pelo botão **Baixar previsões em CSV**.

Os resultados ficam na sessão da interface. Para salvar uma execução
automaticamente em `data/inference/runs/`, use o módulo `run_inference`.

## Limitações e próximos desenvolvimentos

- As datas dos eventos não garantem que uma extração esteja completa.
- Os dados disponíveis podem estar defasados em relação ao calendário.
- O modelo prevê atividade no sistema de pontos, não uma medida completa
  de ganho ou perda de engajamento.
- O limiar foi escolhido por F1 na validação; custos de negócio para os
  diferentes erros ainda não foram definidos.
- Atualizações ou correções nas fontes podem alterar as características.
- As métricas históricas não garantem o desempenho em períodos futuros.
- A análise de agrupamento de perfis ainda não foi implementada.
- Monitoramento operacional e implantação não fazem parte da versão atual.

## Referência

Projeto original:
[TeoMeWhy/loyalty-predict](https://github.com/TeoMeWhy/loyalty-predict).
