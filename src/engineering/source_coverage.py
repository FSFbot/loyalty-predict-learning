import sqlite3
from contextlib import closing
from datetime import date, timedelta

from src.engineering.config import KAGGLE_DATASETS
from src.engineering.inspect_time_ranges import (
    CHECKS,
    inspect_period,
)


def get_source_limits() -> dict[str, date]:
    limits: dict[str, date] = {}

    datasets = {
        dataset.destination.name: dataset
        for dataset in KAGGLE_DATASETS
    }

    for source_name, checks in CHECKS.items():
        database = (
            datasets[source_name].destination
            / "database.db"
        )

        if not database.is_file():
            raise FileNotFoundError(database)

        last_event_dates: list[date] = []

        print(f"\nFonte: {source_name}")

        with closing(
            sqlite3.connect(
                database.resolve().as_uri() + "?mode=ro",
                uri=True,
            )
        ) as connection:
            for check in checks:
                (
                    total_rows,
                    null_dates,
                    invalid_dates,
                    _,
                    maximum_date,
                ) = inspect_period(connection, check)

                if null_dates or invalid_dates:
                    raise ValueError(
                        f"Datas ausentes ou inválidas em "
                        f"{source_name}.{check.table}."
                    )

                if total_rows == 0:
                    print(f"  {check.table}: sem eventos")
                    continue

                if maximum_date is None:
                    raise ValueError(
                        f"Data final ausente em {check.table}."
                    )

                last_event = date.fromisoformat(
                    maximum_date[:10]
                )
                last_event_dates.append(last_event)

                print(
                    f"  {check.table}: "
                    f"último evento em {last_event:%Y-%m-%d}"
                )

        if not last_event_dates:
            raise ValueError(
                f"A fonte {source_name} não possui eventos."
            )

        latest_event = max(last_event_dates)
        latest_month_start = latest_event.replace(day=1)

        previous_month = (
            latest_month_start - timedelta(days=1)
        ).replace(day=1)

        limits[source_name] = previous_month

        print(
            "  Último mês aceito pela regra: "
            f"{previous_month:%Y-%m}"
        )

    return limits


def get_common_reference_month(
    limits: dict[str, date],
) -> date:
    if not limits:
        raise ValueError("Nenhum limite temporal disponível.")

    return min(limits.values())


def main() -> None:
    limits = get_source_limits()

    reference_month = get_common_reference_month(limits)

    print(
        "\nMês de referência comum: "
        f"{reference_month:%Y-%m}"
    )
    print(
        "Regra baseada nas datas dos eventos; "
        "não comprova completude da extração."
    )


if __name__ == "__main__":
    main()