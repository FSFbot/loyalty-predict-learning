"""Valida a chave que conecta os sistemas de fidelidade e educação."""

import sqlite3
from pathlib import Path

from .config import KAGGLE_DATASETS


def get_database_paths() -> tuple[Path, Path]:
    """Localiza os bancos de fidelidade e educação."""

    databases = {
        dataset.destination.name: dataset.destination / "database.db"
        for dataset in KAGGLE_DATASETS
    }

    return databases["loyalty"], databases["education"]


def calculate_bridge_metrics(
    loyalty_database: Path,
    education_database: Path,
) -> dict[str, int]:
    """Calcula métricas de qualidade e cobertura da chave entre sistemas."""

    query = """
        WITH loyalty_ids AS (
            SELECT DISTINCT idCliente AS customer_id
            FROM clientes
            WHERE idCliente IS NOT NULL
        ),
        education_ids AS (
            SELECT DISTINCT idTMWCliente AS customer_id
            FROM education.usuarios_tmw
            WHERE idTMWCliente IS NOT NULL
        )
        SELECT
            (SELECT COUNT(*) FROM clientes) AS loyalty_rows,
            (
                SELECT COUNT(DISTINCT idCliente)
                FROM clientes
            ) AS loyalty_unique_ids,
            (
                SELECT COUNT(*)
                FROM clientes
                WHERE idCliente IS NULL
            ) AS loyalty_null_ids,
            (
                SELECT COUNT(*)
                FROM education.usuarios_tmw
            ) AS education_rows,
            (
                SELECT COUNT(DISTINCT idTMWCliente)
                FROM education.usuarios_tmw
            ) AS education_unique_ids,
            (
                SELECT COUNT(*)
                FROM education.usuarios_tmw
                WHERE idTMWCliente IS NULL
            ) AS education_null_ids,
            (
                SELECT COUNT(*)
                FROM loyalty_ids
                INNER JOIN education_ids USING (customer_id)
            ) AS matched_ids,
            (
                SELECT COUNT(*)
                FROM loyalty_ids
                LEFT JOIN education_ids USING (customer_id)
                WHERE education_ids.customer_id IS NULL
            ) AS loyalty_only_ids,
            (
                SELECT COUNT(*)
                FROM education_ids
                LEFT JOIN loyalty_ids USING (customer_id)
                WHERE loyalty_ids.customer_id IS NULL
            ) AS education_only_ids
    """

    with sqlite3.connect(loyalty_database) as connection:
        connection.execute(
            "ATTACH DATABASE ? AS education",
            (str(education_database),),
        )
        cursor = connection.execute(query)
        row = cursor.fetchone()
        column_names = [column[0] for column in cursor.description]

    return dict(zip(column_names, row, strict=True))


def main() -> None:
    """Exibe as métricas da ponte entre os dois sistemas."""

    loyalty_database, education_database = get_database_paths()
    metrics = calculate_bridge_metrics(
        loyalty_database,
        education_database,
    )

    labels = {
        "loyalty_rows": "Clientes no sistema de fidelidade",
        "loyalty_unique_ids": "IDs únicos em fidelidade",
        "loyalty_null_ids": "IDs nulos em fidelidade",
        "education_rows": "Usuários vinculados na educação",
        "education_unique_ids": "IDs TMW únicos na educação",
        "education_null_ids": "IDs TMW nulos na educação",
        "matched_ids": "IDs presentes nos dois sistemas",
        "loyalty_only_ids": "IDs somente em fidelidade",
        "education_only_ids": "IDs somente na educação",
    }

    for metric_name, value in metrics.items():
        print(f"{labels[metric_name]}: {value:,}")


if __name__ == "__main__":
    main()