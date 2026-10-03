import sqlite3
from contextlib import closing
from dataclasses import asdict, dataclass

from src.analytics.customer_month import (
    load_customer_months,
    validate_customer_months,
)
from src.analytics.customer_month_target import (
    build_customer_month_targets,
    validate_customer_month_targets,
)
from src.analytics.education_month import (
    EducationMonth,
    load_education_months,
    validate_education_months,
)
from src.analytics.historical_features import (
    build_historical_features,
    validate_historical_features,
)
from src.analytics.lifecycle_features import (
    LifecycleFeatureRow,
    build_lifecycle_features,
    validate_lifecycle_features,
)
from src.analytics.modeling_table import (
    build_modeling_table,
    validate_modeling_table,
)
from src.analytics.recency_features import (
    build_recency_features,
    load_active_months,
    validate_recency_features,
)
from src.engineering.config import KAGGLE_DATASETS


@dataclass(frozen=True)
class IntegratedFeatureRow(LifecycleFeatureRow):
    completed_episode_count: int
    distinct_course_count: int
    skill_record_count: int
    distinct_skill_count: int
    reward_count: int
    education_event_count: int
    has_education_activity: int


def build_integrated_features(
    lifecycle_rows: list[LifecycleFeatureRow],
    education_months: list[EducationMonth],
) -> list[IntegratedFeatureRow]:
    education_index: dict[
        tuple[str, str],
        EducationMonth,
    ] = {
        (
            education_month.customer_id,
            education_month.reference_month,
        ): education_month
        for education_month in education_months
    }

    integrated_rows: list[IntegratedFeatureRow] = []

    for lifecycle_row in lifecycle_rows:
        key = (
            lifecycle_row.customer_id,
            lifecycle_row.reference_month,
        )

        education_month = education_index.get(key)

        if education_month is None:
            completed_episode_count = 0
            distinct_course_count = 0
            skill_record_count = 0
            distinct_skill_count = 0
            reward_count = 0
            education_event_count = 0
            has_education_activity = 0
        else:
            completed_episode_count = (
                education_month.completed_episode_count
            )
            distinct_course_count = (
                education_month.distinct_course_count
            )
            skill_record_count = (
                education_month.skill_record_count
            )
            distinct_skill_count = (
                education_month.distinct_skill_count
            )
            reward_count = education_month.reward_count
            education_event_count = (
                education_month.education_event_count
            )
            has_education_activity = 1

        integrated_rows.append(
            IntegratedFeatureRow(
                **asdict(lifecycle_row),
                completed_episode_count=(
                    completed_episode_count
                ),
                distinct_course_count=(
                    distinct_course_count
                ),
                skill_record_count=skill_record_count,
                distinct_skill_count=distinct_skill_count,
                reward_count=reward_count,
                education_event_count=(
                    education_event_count
                ),
                has_education_activity=(
                    has_education_activity
                ),
            )
        )

    return integrated_rows


def validate_integrated_features(
    integrated_rows: list[IntegratedFeatureRow],
    lifecycle_rows: list[LifecycleFeatureRow],
) -> None:
    if not integrated_rows:
        raise ValueError(
            "A tabela integrada está vazia."
        )

    if len(integrated_rows) != len(lifecycle_rows):
        raise ValueError(
            "A quantidade de linhas mudou durante a integração "
            "dos dados educacionais."
        )

    lifecycle_index = {
        (
            row.customer_id,
            row.reference_month,
        ): row
        for row in lifecycle_rows
    }

    observed_keys: set[tuple[str, str]] = set()

    for row in integrated_rows:
        key = (
            row.customer_id,
            row.reference_month,
        )

        if key in observed_keys:
            raise ValueError(
                "Linha integrada duplicada para "
                f"{row.customer_id} em "
                f"{row.reference_month}."
            )

        observed_keys.add(key)

        source_row = lifecycle_index.get(key)

        if source_row is None:
            raise ValueError(
                "Linha integrada sem origem na base de fidelidade para "
                f"{row.customer_id} em "
                f"{row.reference_month}."
            )

        for field_name, expected_value in asdict(
            source_row
        ).items():
            actual_value = getattr(
                row,
                field_name,
            )

            if actual_value != expected_value:
                raise ValueError(
                    "Campo de fidelidade alterado durante a integração: "
                    f"{field_name} para "
                    f"{row.customer_id} em "
                    f"{row.reference_month}."
                )

        education_metrics = (
            row.completed_episode_count,
            row.distinct_course_count,
            row.skill_record_count,
            row.distinct_skill_count,
            row.reward_count,
            row.education_event_count,
        )

        if any(metric < 0 for metric in education_metrics):
            raise ValueError(
                "Métrica educacional negativa para "
                f"{row.customer_id} em "
                f"{row.reference_month}."
            )

        expected_event_count = (
            row.completed_episode_count
            + row.skill_record_count
            + row.reward_count
        )

        if row.education_event_count != expected_event_count:
            raise ValueError(
                "Total educacional inconsistente para "
                f"{row.customer_id} em "
                f"{row.reference_month}."
            )

        expected_activity_flag = int(
            row.education_event_count > 0
        )

        if (
            row.has_education_activity
            != expected_activity_flag
        ):
            raise ValueError(
                "Indicador educacional inconsistente para "
                f"{row.customer_id} em "
                f"{row.reference_month}."
            )

        if (
            row.distinct_course_count
            > row.completed_episode_count
        ):
            raise ValueError(
                "Cursos distintos excedem episódios concluídos para "
                f"{row.customer_id} em "
                f"{row.reference_month}."
            )

        if (
            row.distinct_skill_count
            > row.skill_record_count
        ):
            raise ValueError(
                "Habilidades distintas excedem registros para "
                f"{row.customer_id} em "
                f"{row.reference_month}."
            )

    expected_keys = set(lifecycle_index)

    if observed_keys != expected_keys:
        raise ValueError(
            "As chaves da tabela principal foram alteradas "
            "durante a integração."
        )


def main() -> None:
    datasets = {
        dataset.destination.name: dataset
        for dataset in KAGGLE_DATASETS
    }

    loyalty_database = (
        datasets["loyalty"].destination
        / "database.db"
    )
    education_database = (
        datasets["education"].destination
        / "database.db"
    )

    with closing(
        sqlite3.connect(loyalty_database)
    ) as loyalty_connection:
        customer_months = load_customer_months(
            loyalty_connection
        )
        active_months_by_customer = load_active_months(
            loyalty_connection
        )

    with closing(
        sqlite3.connect(education_database)
    ) as education_connection:
        education_months = load_education_months(
            education_connection
        )

    validate_customer_months(customer_months)
    validate_education_months(education_months)

    targets = build_customer_month_targets(
        customer_months
    )
    validate_customer_month_targets(targets)

    modeling_rows = build_modeling_table(
        customer_months,
        targets,
    )
    validate_modeling_table(
        modeling_rows,
        targets,
    )

    historical_rows = build_historical_features(
        modeling_rows
    )
    validate_historical_features(
        historical_rows,
        modeling_rows,
    )

    recency_rows = build_recency_features(
        historical_rows,
        active_months_by_customer,
    )
    validate_recency_features(
        recency_rows,
        historical_rows,
    )

    lifecycle_rows = build_lifecycle_features(
        recency_rows
    )
    validate_lifecycle_features(
        lifecycle_rows,
        recency_rows,
    )

    integrated_rows = build_integrated_features(
        lifecycle_rows,
        education_months,
    )
    validate_integrated_features(
        integrated_rows,
        lifecycle_rows,
    )

    rows_with_education = sum(
        row.has_education_activity
        for row in integrated_rows
    )

    rows_without_education = (
        len(integrated_rows)
        - rows_with_education
    )

    customers_with_education = {
        row.customer_id
        for row in integrated_rows
        if row.has_education_activity == 1
    }

    matched_education_keys = {
        (
            row.customer_id,
            row.reference_month,
        )
        for row in integrated_rows
        if row.has_education_activity == 1
    }

    unmatched_education_rows = (
        len(education_months)
        - len(matched_education_keys)
    )

    matched_education_events = sum(
        row.education_event_count
        for row in integrated_rows
    )

    print("Validação da tabela integrada: OK")
    print(f"Total de linhas: {len(integrated_rows):,}")
    print(
        "Linhas com atividade educacional: "
        f"{rows_with_education:,}"
    )
    print(
        "Linhas sem atividade educacional: "
        f"{rows_without_education:,}"
    )
    print(
        "Clientes com atividade educacional: "
        f"{len(customers_with_education):,}"
    )
    print(
        "Linhas educacionais fora da grade do modelo: "
        f"{unmatched_education_rows:,}"
    )
    print(
        "Eventos educacionais incorporados: "
        f"{matched_education_events:,}"
    )

    print("\nPrimeiras 5 linhas com atividade educacional:")

    education_examples = [
        row
        for row in integrated_rows
        if row.has_education_activity == 1
    ]

    for row in education_examples[:5]:
        print(row)


if __name__ == "__main__":
    main()