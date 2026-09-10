"""Lista as tabelas disponíveis nos bancos SQLite brutos."""

import sqlite3
from pathlib import Path

from .config import KAGGLE_DATASETS


def list_tables(database_path: Path) -> list[str]:
    """Retorna os nomes das tabelas de um banco SQLite."""

    query = """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
          AND name NOT LIKE 'sqlite_%'
        ORDER BY name
    """

    with sqlite3.connect(database_path) as connection:
        rows = connection.execute(query).fetchall()

    return [row[0] for row in rows]


def main() -> None:
    """Lista as tabelas de cada fonte configurada."""

    for dataset in KAGGLE_DATASETS:
        database_path = dataset.destination / "database.db"

        if not database_path.exists():
            raise FileNotFoundError(
                f"Banco não encontrado: {database_path}. "
                "Execute primeiro o pipeline de aquisição."
            )

        print(f"\nDataset: {dataset.slug}")
        print(f"Banco: {database_path}")

        for table_name in list_tables(database_path):
            print(f"- {table_name}")


if __name__ == "__main__":
    main()