import sqlite3
from collections import defaultdict
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
from src.analytics.integrated_features import (
    IntegratedFeatureRow,
    build_integrated_features,
    validate_integrated_features,
)
from src.analytics.lifecycle_features import (
    build_lifecycle_features,
    validate_lifecycle_features,
)
from src.analytics.modeling_table import (
    build_modeling_table,
    validate_modeling_table,
)
from src.analytics.recency_features import (
    build_recency_features,
    calculate_month_distance,
    load_active_months,
    validate_recency_features,
)
from src.engineering.config import KAGGLE_DATASETS


@dataclass(frozen=True)
class EducationHistoryFeatureRow(IntegratedFeatureRow):
    education_active_months_last_3: int
    education_event_count_last_3_months: int
    months_since_last_education_activity: int | None
    education_active_months_to_date: int
    education_event_count_to_date: int


def get_previous_month(reference_month: str) -> str:
    year, month = map(
        int,
        reference_month.split("-"),
    )

    if month == 1:
        year -= 1
        month = 12
    else:
        month -= 1

    return f"{year:04d}-{month:02d}"


def build_education_history_features(
    integrated_rows: list[IntegratedFeatureRow],
    education_months: list[EducationMonth],
) -> list[EducationHistoryFeatureRow]:
    model_rows_by_customer: defaultdict[
        str,
        list[IntegratedFeatureRow],
    ] = defaultdict(list)

    for row in integrated_rows:
        model_rows_by_customer[row.customer_id].append(
            row
        )

    education_rows_by_customer: defaultdict[
        str,
        list[EducationMonth],
    ] = defaultdict(list)

    for education_month in education_months:
        education_rows_by_customer[
            education_month.customer_id
        ].append(education_month)

    education_index = {
        (
            education_month.customer_id,
            education_month.reference_month,
        ): education_month
        for education_month in education_months
    }

    history_rows: list[EducationHistoryFeatureRow] = []

    for (
        customer_id,
        customer_model_rows,
    ) in model_rows_by_customer.items():
        customer_model_rows.sort(
            key=lambda row: row.reference_month
        )

        customer_education_rows = (
            education_rows_by_customer.get(
                customer_id,
                [],
            )
        )
        customer_education_rows.sort(
            key=lambda row: row.reference_month
        )

        education_position = 0
        education_active_months_to_date = 0
        education_event_count_to_date = 0
        last_education_activity_month: str | None = None

        for model_row in customer_model_rows:
            while (
                education_position
                < len(customer_education_rows)
                and customer_education_rows[
                    education_position
                ].reference_month
                <= model_row.reference_month
            ):
                education_row = customer_education_rows[
                    education_position
                ]

                education_active_months_to_date += 1
                education_event_count_to_date += (
                    education_row.education_event_count
                )
                last_education_activity_month = (
                    education_row.reference_month
                )
                education_position += 1

            if last_education_activity_month is None:
                months_since_last_education_activity = None
            else:
                months_since_last_education_activity = (
                    calculate_month_distance(
                        last_education_activity_month,
                        model_row.reference_month,
                    )
                )

            previous_month = get_previous_month(
                model_row.reference_month
            )
            two_months_ago = get_previous_month(
                previous_month
            )

            window_months = (
                model_row.reference_month,
                previous_month,
                two_months_ago,
            )

            education_window = [
                education_index.get(
                    (
                        customer_id,
                        window_month,
                    )
                )
                for window_month in window_months
            ]

            education_active_months_last_3 = sum(
                education_month is not None
                for education_month in education_window
            )

            education_event_count_last_3_months = sum(
                education_month.education_event_count
                for education_month in education_window
                if education_month is not None
            )

            history_rows.append(
                EducationHistoryFeatureRow(
                    **asdict(model_row),
                    education_active_months_last_3=(
                        education_active_months_last_3
                    ),
                    education_event_count_last_3_months=(
                        education_event_count_last_3_months
                    ),
                    months_since_last_education_activity=(
                        months_since_last_education_activity
                    ),
                    education_active_months_to_date=(
                        education_active_months_to_date
                    ),
                    education_event_count_to_date=(
                        education_event_count_to_date
                    ),
                )
            )

    history_rows.sort(
        key=lambda row: (
            row.reference_month,
            row.customer_id,
        )
    )

    return history_rows


def validate_education_history_features(
    history_rows: list[EducationHistoryFeatureRow],
    integrated_rows: list[IntegratedFeatureRow],
) -> None:
    if not history_rows:
        raise ValueError(
            "A tabela de histórico educacional está vazia."
        )

    if len(history_rows) != len(integrated_rows):
        raise ValueError(
            "A quantidade de linhas mudou durante a construção "
            "do histórico educacional."
        )

    integrated_index = {
        (
            row.customer_id,
            row.reference_month,
        ): row
        for row in integrated_rows
    }

    observed_keys: set[tuple[str, str]] = set()

    rows_by_customer: defaultdict[
        str,
        list[EducationHistoryFeatureRow],
    ] = defaultdict(list)

    for row in history_rows:
        key = (
            row.customer_id,
            row.reference_month,
        )

        if key in observed_keys:
            raise ValueError(
                "Histórico educacional duplicado para "
                f"{row.customer_id} em "
                f"{row.reference_month}."
            )

        observed_keys.add(key)
        rows_by_customer[row.customer_id].append(row)

        source_row = integrated_index.get(key)

        if source_row is None:
            raise ValueError(
                "Histórico educacional sem linha integrada de origem."
            )

        for field_name, expected_value in asdict(
            source_row
        ).items():
            if getattr(row, field_name) != expected_value:
                raise ValueError(
                    "Campo anterior alterado durante a construção "
                    f"do histórico educacional: {field_name}."
                )

        if not (
            0
            <= row.education_active_months_last_3
            <= 3
        ):
            raise ValueError(
                "Quantidade inválida de meses educacionais ativos."
            )

        if (
            row.education_event_count_last_3_months
            < row.education_event_count
        ):
            raise ValueError(
                "A janela educacional possui menos eventos "
                "que o mês atual."
            )

        if (
            row.education_active_months_to_date
            < row.education_active_months_last_3
        ):
            raise ValueError(
                "O acumulado de meses educacionais é menor "
                "que a janela recente."
            )

        if (
            row.education_event_count_to_date
            < row.education_event_count_last_3_months
        ):
            raise ValueError(
                "O acumulado de eventos educacionais é menor "
                "que a janela recente."
            )

        if (
            row.has_education_activity == 1
            and row.months_since_last_education_activity != 0
        ):
            raise ValueError(
                "Mês educacionalmente ativo com recência "
                "diferente de zero."
            )

        if (
            row.has_education_activity == 0
            and row.months_since_last_education_activity == 0
        ):
            raise ValueError(
                "Mês sem atividade educacional com recência zero."
            )

        if row.months_since_last_education_activity is None:
            if (
                row.education_active_months_to_date != 0
                or row.education_event_count_to_date != 0
            ):
                raise ValueError(
                    "Cliente sem atividade conhecida possui "
                    "acumulados educacionais."
                )

    if observed_keys != set(integrated_index):
        raise ValueError(
            "As chaves foram alteradas durante a construção "
            "do histórico educacional."
        )

    for customer_rows in rows_by_customer.values():
        customer_rows.sort(
            key=lambda row: row.reference_month
        )

        previous_row: EducationHistoryFeatureRow | None = None

        for row in customer_rows:
            if previous_row is not None:
                expected_active_months = (
                    previous_row.education_active_months_to_date
                    + row.has_education_activity
                )
                expected_event_count = (
                    previous_row.education_event_count_to_date
                    + row.education_event_count
                )

                if (
                    row.education_active_months_to_date
                    != expected_active_months
                ):
                    raise ValueError(
                        "Acúmulo de meses educacionais inconsistente."
                    )

                if (
                    row.education_event_count_to_date
                    != expected_event_count
                ):
                    raise ValueError(
                        "Acúmulo de eventos educacionais inconsistente."
                    )

                if row.has_education_activity == 0:
                    previous_recency = (
                        previous_row
                        .months_since_last_education_activity
                    )

                    if previous_recency is None:
                        expected_recency = None
                    else:
                        expected_recency = previous_recency + 1

                    if (
                        row.months_since_last_education_activity
                        != expected_recency
                    ):
                        raise ValueError(
                            "Evolução da recência educacional "
                            "inconsistente."
                        )

            previous_row = row


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

    education_history_rows = (
        build_education_history_features(
            integrated_rows,
            education_months,
        )
    )
    validate_education_history_features(
        education_history_rows,
        integrated_rows,
    )

    no_known_education = sum(
        row.months_since_last_education_activity is None
        for row in education_history_rows
    )

    active_current_month = sum(
        row.has_education_activity
        for row in education_history_rows
    )

    inactive_with_education_history = (
        len(education_history_rows)
        - no_known_education
        - active_current_month
    )

    maximum_education_recency = max(
        (
            row.months_since_last_education_activity
            for row in education_history_rows
            if row.months_since_last_education_activity
            is not None
        ),
        default=0,
    )

    print("Validação do histórico educacional: OK")
    print(
        f"Total de linhas: "
        f"{len(education_history_rows):,}"
    )
    print(
        "Sem atividade educacional conhecida: "
        f"{no_known_education:,}"
    )
    print(
        "Ativos educacionalmente no mês atual: "
        f"{active_current_month:,}"
    )
    print(
        "Inativos com histórico educacional: "
        f"{inactive_with_education_history:,}"
    )
    print(
        "Maior recência educacional: "
        f"{maximum_education_recency} meses"
    )

    print("\nPrimeiros 5 inativos com histórico educacional:")

    examples = [
        row
        for row in education_history_rows
        if row.has_education_activity == 0
        and row.months_since_last_education_activity
        is not None
    ]

    for row in examples[:5]:
        print(row)


if __name__ == "__main__":
    main()