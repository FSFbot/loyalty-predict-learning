from dataclasses import dataclass

import pandas as pd

from src.modeling.modeling_dataset import (
    IDENTIFIER_COLUMNS,
    TARGET_COLUMN,
    build_modeling_dataset,
    load_feature_rows,
    validate_modeling_dataset,
)


TRAIN_END_MONTH = pd.Timestamp("2025-12-01")
VALIDATION_END_MONTH = pd.Timestamp("2026-04-01")


@dataclass
class TemporalSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def split_modeling_dataset(
    dataset: pd.DataFrame,
) -> TemporalSplit:
    train_mask = (
        dataset["reference_month"]
        <= TRAIN_END_MONTH
    )

    validation_mask = (
        (
            dataset["reference_month"]
            > TRAIN_END_MONTH
        )
        & (
            dataset["reference_month"]
            <= VALIDATION_END_MONTH
        )
    )

    test_mask = (
        dataset["reference_month"]
        > VALIDATION_END_MONTH
    )

    train = (
        dataset.loc[train_mask]
        .copy()
        .reset_index(drop=True)
    )

    validation = (
        dataset.loc[validation_mask]
        .copy()
        .reset_index(drop=True)
    )

    test = (
        dataset.loc[test_mask]
        .copy()
        .reset_index(drop=True)
    )

    return TemporalSplit(
        train=train,
        validation=validation,
        test=test,
    )


def validate_temporal_split(
    dataset: pd.DataFrame,
    split: TemporalSplit,
) -> None:
    split_datasets = {
        "treino": split.train,
        "validação": split.validation,
        "teste": split.test,
    }

    for split_name, split_dataset in (
        split_datasets.items()
    ):
        if split_dataset.empty:
            raise ValueError(
                f"O conjunto de {split_name} está vazio."
            )

        if (
            split_dataset.columns.tolist()
            != dataset.columns.tolist()
        ):
            raise ValueError(
                f"As colunas de {split_name} foram alteradas."
            )

        target_values = set(
            split_dataset[TARGET_COLUMN].unique()
        )

        if target_values != {0, 1}:
            raise ValueError(
                f"O conjunto de {split_name} não possui "
                "as duas classes do alvo."
            )

    if (
        split.train["reference_month"].max()
        >= split.validation["reference_month"].min()
    ):
        raise ValueError(
            "Treino e validação possuem períodos sobrepostos."
        )

    if (
        split.validation["reference_month"].max()
        >= split.test["reference_month"].min()
    ):
        raise ValueError(
            "Validação e teste possuem períodos sobrepostos."
        )

    total_split_rows = (
        len(split.train)
        + len(split.validation)
        + len(split.test)
    )

    if total_split_rows != len(dataset):
        raise ValueError(
            "A divisão temporal perdeu ou duplicou linhas."
        )

    expected_keys = set(
        dataset.loc[
            :,
            list(IDENTIFIER_COLUMNS),
        ].itertuples(
            index=False,
            name=None,
        )
    )

    observed_dataset = pd.concat(
        [
            split.train,
            split.validation,
            split.test,
        ],
        ignore_index=True,
    )

    observed_keys = set(
        observed_dataset.loc[
            :,
            list(IDENTIFIER_COLUMNS),
        ].itertuples(
            index=False,
            name=None,
        )
    )

    if observed_keys != expected_keys:
        raise ValueError(
            "As chaves foram alteradas durante a divisão temporal."
        )


def print_split_summary(
    split_name: str,
    dataset: pd.DataFrame,
) -> None:
    positive_count = int(
        dataset[TARGET_COLUMN].sum()
    )

    negative_count = (
        len(dataset)
        - positive_count
    )

    positive_rate = (
        positive_count
        / len(dataset)
    )

    print(f"\n{split_name}")
    print(
        "  Período: "
        f"{dataset['reference_month'].min():%Y-%m} "
        "até "
        f"{dataset['reference_month'].max():%Y-%m}"
    )
    print(f"  Linhas: {len(dataset):,}")
    print(f"  Negativos: {negative_count:,}")
    print(f"  Positivos: {positive_count:,}")
    print(
        "  Taxa positiva: "
        f"{positive_rate:.2%}"
    )


def main() -> None:
    feature_rows = load_feature_rows()

    dataset = build_modeling_dataset(
        feature_rows
    )

    validate_modeling_dataset(
        dataset,
        expected_row_count=len(feature_rows),
    )

    split = split_modeling_dataset(dataset)

    validate_temporal_split(
        dataset,
        split,
    )

    print("Validação da divisão temporal: OK")

    print_split_summary(
        "Treino",
        split.train,
    )
    print_split_summary(
        "Validação",
        split.validation,
    )
    print_split_summary(
        "Teste",
        split.test,
    )


if __name__ == "__main__":
    main()