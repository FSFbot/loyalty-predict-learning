"""Inspeciona as tabelas dos bancos SQLite brutos."""

import sqlite3
from pathlib import Path

from .config import KAGGLE_DATASETS


def list_tables(connection: sqlite3.Connection) -> list[str]:
    """Retorna os nomes das tabelas de uma conexão SQLite."""

    query = """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
          AND name NOT LIKE 'sqlite_%'
        ORDER BY name
    """
    rows = connection.execute(query).fetchall()

    return [row[0] for row in rows]


def quote_identifier(identifier: str) -> str:
    """Protege um identificador para uso em uma consulta SQL."""

    escaped_identifier = identifier.replace('"', '""')
    return f'"{escaped_identifier}"'


def count_rows(
    connection: sqlite3.Connection,
    table_name: str,
) -> int:
    """Conta as linhas de uma tabela."""

    quoted_table = quote_identifier(table_name)
    query = f"SELECT COUNT(*) FROM {quoted_table}"
    result = connection.execute(query).fetchone()

    return result[0]


def list_columns(
    connection: sqlite3.Connection,
    table_name: str,
) -> list[tuple[str, str]]:
    """Retorna o nome e o tipo declarado das colunas de uma tabela."""

    query = """
        SELECT name, type
        FROM pragma_table_info(?)
        ORDER BY cid
    """
    rows = connection.execute(query, (table_name,)).fetchall()

    return [(row[0], row[1]) for row in rows]


def inspect_database(database_path: Path) -> None:
    """Exibe tabelas, quantidades de linhas e colunas de um banco."""

    with sqlite3.connect(database_path) as connection:
        for table_name in list_tables(connection):
            row_count = count_rows(connection, table_name)
            columns = list_columns(connection, table_name)

            print(f"\n- {table_name}: {row_count:,} linhas")
            print("  Colunas:")

            for column_name, data_type in columns:
                print(f"    - {column_name} ({data_type})")


def main() -> None:
    """Inspeciona os bancos de todas as fontes configuradas."""

    for dataset in KAGGLE_DATASETS:
        database_path = dataset.destination / "database.db"

        if not database_path.exists():
            raise FileNotFoundError(
                f"Banco não encontrado: {database_path}. "
                "Execute primeiro o pipeline de aquisição."
            )

        print(f"\nDataset: {dataset.slug}")
        print(f"Banco: {database_path}")

        inspect_database(database_path)


if __name__ == "__main__":
    main()