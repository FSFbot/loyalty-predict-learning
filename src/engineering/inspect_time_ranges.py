import sqlite3
from dataclasses import dataclass
from pathlib import Path

from .config import KAGGLE_DATASETS


@dataclass(frozen=True)
class TemporalCheck:
    table: str
    date_column: str


CHECKS = {
    "loyalty": (
        TemporalCheck("transacoes", "DtCriacao"),
    ),
    "education": (
        TemporalCheck(
            "cursos_episodios_completos",
            "dtCriacao",
        ),
        TemporalCheck(
            "habilidades_usuarios",
            "dtCriacao",
        ),
        TemporalCheck(
            "recompensas_usuarios",
            "dtRecompensa",
        ),
    ),
}


def quote_identifier(identifier: str) -> str:
    escaped_identifier = identifier.replace('"', '""')
    return f'"{escaped_identifier}"'


def inspect_period(
    connection: sqlite3.Connection,
    check: TemporalCheck,
) -> tuple[int, int, int, str | None, str | None]:
    table = quote_identifier(check.table)
    date_column = quote_identifier(check.date_column)

    query = f"""
        SELECT
            COUNT(*) AS total_rows,
            COALESCE(SUM({date_column} IS NULL), 0) AS null_dates,
            COALESCE(
                SUM(
                    {date_column} IS NOT NULL
                    AND datetime({date_column}) IS NULL
                ),
                0
            ) AS invalid_dates,
            MIN({date_column}) AS minimum_date,
            MAX({date_column}) AS maximum_date
        FROM {table}
    """

    row = connection.execute(query).fetchone()

    return (
        int(row[0]),
        int(row[1]),
        int(row[2]),
        row[3],
        row[4],
    )


def main() -> None:
    databases = {
        dataset.destination.name: dataset.destination / "database.db"
        for dataset in KAGGLE_DATASETS
    }

    for source_name, checks in CHECKS.items():
        print(f"\nFonte: {source_name}")

        with sqlite3.connect(databases[source_name]) as connection:
            for check in checks:
                (
                    total_rows,
                    null_dates,
                    invalid_dates,
                    minimum_date,
                    maximum_date,
                ) = inspect_period(connection, check)

                print(f"\nTabela: {check.table}")
                print(f"  Coluna temporal: {check.date_column}")
                print(f"  Total de eventos: {total_rows:,}")
                print(f"  Datas nulas: {null_dates:,}")
                print(f"  Datas inválidas: {invalid_dates:,}")
                print(f"  Data inicial: {minimum_date}")
                print(f"  Data final: {maximum_date}")


if __name__ == "__main__":
    main()