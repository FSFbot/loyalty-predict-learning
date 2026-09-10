"""Valida as relações internas dos bancos SQLite brutos."""

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from .config import KAGGLE_DATASETS


@dataclass(frozen=True)
class ReferenceCheck:
    """Descreve uma relação entre uma tabela filha e uma tabela pai."""

    child_table: str
    child_column: str
    parent_table: str
    parent_column: str


LOYALTY_CHECKS = (
    ReferenceCheck(
        child_table="transacoes",
        child_column="IdCliente",
        parent_table="clientes",
        parent_column="idCliente",
    ),
    ReferenceCheck(
        child_table="transacao_produto",
        child_column="IdTransacao",
        parent_table="transacoes",
        parent_column="IdTransacao",
    ),
    ReferenceCheck(
        child_table="transacao_produto",
        child_column="IdProduto",
        parent_table="produtos",
        parent_column="IdProduto",
    ),
)

EDUCATION_CHECKS = (
    ReferenceCheck(
        child_table="cursos_episodios",
        child_column="descSlugCurso",
        parent_table="cursos",
        parent_column="descSlugCurso",
    ),
    ReferenceCheck(
        child_table="cursos_episodios_completos",
        child_column="idUsuario",
        parent_table="usuarios_tmw",
        parent_column="idUsuario",
    ),
    ReferenceCheck(
        child_table="cursos_episodios_completos",
        child_column="descSlugCurso",
        parent_table="cursos",
        parent_column="descSlugCurso",
    ),
    ReferenceCheck(
        child_table="habilidades_usuarios",
        child_column="idUsuario",
        parent_table="usuarios_tmw",
        parent_column="idUsuario",
    ),
    ReferenceCheck(
        child_table="habilidades_usuarios",
        child_column="descNomeHabilidade",
        parent_table="habilidades",
        parent_column="descNomeHabilidade",
    ),
    ReferenceCheck(
        child_table="habilidades_cargos",
        child_column="descNomeHabilidade",
        parent_table="habilidades",
        parent_column="descNomeHabilidade",
    ),
    ReferenceCheck(
        child_table="recompensas_usuarios",
        child_column="idUsuario",
        parent_table="usuarios_tmw",
        parent_column="idUsuario",
    ),
)


def quote_identifier(identifier: str) -> str:
    """Protege um identificador para uso em SQL."""

    escaped_identifier = identifier.replace('"', '""')
    return f'"{escaped_identifier}"'


def count_reference_problems(
    connection: sqlite3.Connection,
    check: ReferenceCheck,
) -> dict[str, int]:
    """Mede volume e diversidade das chaves sem correspondência."""

    child_table = quote_identifier(check.child_table)
    child_column = quote_identifier(check.child_column)
    parent_table = quote_identifier(check.parent_table)
    parent_column = quote_identifier(check.parent_column)

    query = f"""
        SELECT
            COUNT(*) AS total_rows,
            COALESCE(
                SUM(child.{child_column} IS NULL),
                0
            ) AS null_keys,
            COALESCE(
                SUM(
                    child.{child_column} IS NOT NULL
                    AND parent.{parent_column} IS NULL
                ),
                0
            ) AS unmatched_rows,
            COUNT(
                DISTINCT CASE
                    WHEN child.{child_column} IS NOT NULL
                     AND parent.{parent_column} IS NULL
                    THEN child.{child_column}
                END
            ) AS unmatched_distinct_keys
        FROM {child_table} AS child
        LEFT JOIN {parent_table} AS parent
            ON child.{child_column} = parent.{parent_column}
    """

    cursor = connection.execute(query)
    row = cursor.fetchone()
    column_names = [column[0] for column in cursor.description]

    return dict(zip(column_names, row, strict=True))


def validate_database(
    database_path: Path,
    checks: tuple[ReferenceCheck, ...],
) -> None:
    """Executa e exibe as validações de um banco."""

    with sqlite3.connect(database_path) as connection:
        for check in checks:
            metrics = count_reference_problems(
                connection,
                check,
            )

            valid_population = (
                metrics["total_rows"] - metrics["null_keys"]
            )
            matched_rows = (
                valid_population - metrics["unmatched_rows"]
            )
            coverage = (
                matched_rows / valid_population * 100
                if valid_population
                else 0.0
            )

            relationship = (
                f"{check.child_table}.{check.child_column} -> "
                f"{check.parent_table}.{check.parent_column}"
            )

            print(relationship)
            print(f"  Total de linhas: {metrics['total_rows']:,}")
            print(f"  Chaves nulas: {metrics['null_keys']:,}")
            print(
                "  Linhas sem correspondência: "
                f"{metrics['unmatched_rows']:,}"
            )
            print(
                "  Chaves distintas sem correspondência: "
                f"{metrics['unmatched_distinct_keys']:,}"
            )
            print(f"  Cobertura: {coverage:.2f}%")

def main() -> None:
    """Valida as relações conhecidas nas duas fontes."""

    databases = {
        dataset.destination.name: dataset.destination / "database.db"
        for dataset in KAGGLE_DATASETS
    }

    print("\nSistema de fidelidade")
    validate_database(
        databases["loyalty"],
        LOYALTY_CHECKS,
    )

    print("\nPlataforma educacional")
    validate_database(
        databases["education"],
        EDUCATION_CHECKS,
    )


if __name__ == "__main__":
    main()