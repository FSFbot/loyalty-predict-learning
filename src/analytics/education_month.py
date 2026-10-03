import sqlite3
from contextlib import closing
from dataclasses import dataclass

from src.engineering.config import KAGGLE_DATASETS


@dataclass(frozen=True)
class EducationMonth:
    customer_id: str
    reference_month: str
    completed_episode_count: int
    distinct_course_count: int
    skill_record_count: int
    distinct_skill_count: int
    reward_count: int
    education_event_count: int


def load_education_months(
    connection: sqlite3.Connection,
) -> list[EducationMonth]:
    connection.row_factory = sqlite3.Row

    query = """
        WITH

        education_events AS (
            SELECT
                dtCriacao AS event_at
            FROM cursos_episodios_completos

            UNION ALL

            SELECT
                dtCriacao AS event_at
            FROM habilidades_usuarios

            UNION ALL

            SELECT
                dtRecompensa AS event_at
            FROM recompensas_usuarios
        ),

        boundaries AS (
            SELECT
                date(
                    MAX(event_at),
                    'start of month'
                ) AS last_month_start
            FROM education_events
        ),

        completion_activity AS (
            SELECT
                user_map.idTMWCliente AS customer_id,
                date(
                    completion.dtCriacao,
                    'start of month'
                ) AS reference_month,
                COUNT(*) AS completed_episode_count,
                COUNT(
                    DISTINCT completion.descSlugCurso
                ) AS distinct_course_count
            FROM cursos_episodios_completos AS completion
            INNER JOIN usuarios_tmw AS user_map
                ON user_map.idUsuario
                   = completion.idUsuario
            CROSS JOIN boundaries
            WHERE user_map.idTMWCliente IS NOT NULL
              AND datetime(completion.dtCriacao)
                  < boundaries.last_month_start
            GROUP BY
                user_map.idTMWCliente,
                reference_month
        ),

        skill_activity AS (
            SELECT
                user_map.idTMWCliente AS customer_id,
                date(
                    skill.dtCriacao,
                    'start of month'
                ) AS reference_month,
                COUNT(*) AS skill_record_count,
                COUNT(
                    DISTINCT skill.descNomeHabilidade
                ) AS distinct_skill_count
            FROM habilidades_usuarios AS skill
            INNER JOIN usuarios_tmw AS user_map
                ON user_map.idUsuario
                   = skill.idUsuario
            CROSS JOIN boundaries
            WHERE user_map.idTMWCliente IS NOT NULL
              AND datetime(skill.dtCriacao)
                  < boundaries.last_month_start
            GROUP BY
                user_map.idTMWCliente,
                reference_month
        ),

        reward_activity AS (
            SELECT
                user_map.idTMWCliente AS customer_id,
                date(
                    reward.dtRecompensa,
                    'start of month'
                ) AS reference_month,
                COUNT(*) AS reward_count
            FROM recompensas_usuarios AS reward
            INNER JOIN usuarios_tmw AS user_map
                ON user_map.idUsuario
                   = reward.idUsuario
            CROSS JOIN boundaries
            WHERE user_map.idTMWCliente IS NOT NULL
              AND datetime(reward.dtRecompensa)
                  < boundaries.last_month_start
            GROUP BY
                user_map.idTMWCliente,
                reference_month
        ),

        education_month_keys AS (
            SELECT
                customer_id,
                reference_month
            FROM completion_activity

            UNION

            SELECT
                customer_id,
                reference_month
            FROM skill_activity

            UNION

            SELECT
                customer_id,
                reference_month
            FROM reward_activity
        )

        SELECT
            month_key.customer_id,
            strftime(
                '%Y-%m',
                month_key.reference_month
            ) AS reference_month,
            COALESCE(
                completion.completed_episode_count,
                0
            ) AS completed_episode_count,
            COALESCE(
                completion.distinct_course_count,
                0
            ) AS distinct_course_count,
            COALESCE(
                skill.skill_record_count,
                0
            ) AS skill_record_count,
            COALESCE(
                skill.distinct_skill_count,
                0
            ) AS distinct_skill_count,
            COALESCE(
                reward.reward_count,
                0
            ) AS reward_count,
            COALESCE(
                completion.completed_episode_count,
                0
            )
            + COALESCE(
                skill.skill_record_count,
                0
            )
            + COALESCE(
                reward.reward_count,
                0
            ) AS education_event_count
        FROM education_month_keys AS month_key
        LEFT JOIN completion_activity AS completion
            ON completion.customer_id
               = month_key.customer_id
           AND completion.reference_month
               = month_key.reference_month
        LEFT JOIN skill_activity AS skill
            ON skill.customer_id
               = month_key.customer_id
           AND skill.reference_month
               = month_key.reference_month
        LEFT JOIN reward_activity AS reward
            ON reward.customer_id
               = month_key.customer_id
           AND reward.reference_month
               = month_key.reference_month
        ORDER BY
            reference_month,
            month_key.customer_id
    """

    cursor = connection.execute(query)
    rows = cursor.fetchall()

    return [
        EducationMonth(
            customer_id=str(row["customer_id"]),
            reference_month=str(row["reference_month"]),
            completed_episode_count=int(
                row["completed_episode_count"]
            ),
            distinct_course_count=int(
                row["distinct_course_count"]
            ),
            skill_record_count=int(
                row["skill_record_count"]
            ),
            distinct_skill_count=int(
                row["distinct_skill_count"]
            ),
            reward_count=int(row["reward_count"]),
            education_event_count=int(
                row["education_event_count"]
            ),
        )
        for row in rows
    ]


def validate_education_months(
    education_months: list[EducationMonth],
) -> None:
    if not education_months:
        raise ValueError(
            "A base mensal educacional está vazia."
        )

    observed_keys: set[tuple[str, str]] = set()

    for row in education_months:
        key = (
            row.customer_id,
            row.reference_month,
        )

        if key in observed_keys:
            raise ValueError(
                "Atividade educacional duplicada para "
                f"{row.customer_id} em "
                f"{row.reference_month}."
            )

        observed_keys.add(key)

        metrics = (
            row.completed_episode_count,
            row.distinct_course_count,
            row.skill_record_count,
            row.distinct_skill_count,
            row.reward_count,
            row.education_event_count,
        )

        if any(metric < 0 for metric in metrics):
            raise ValueError(
                "Métrica educacional negativa para "
                f"{row.customer_id} em "
                f"{row.reference_month}."
            )

        if (
            row.distinct_course_count
            > row.completed_episode_count
        ):
            raise ValueError(
                "Quantidade de cursos maior que a quantidade "
                "de episódios concluídos para "
                f"{row.customer_id} em "
                f"{row.reference_month}."
            )

        if (
            row.distinct_skill_count
            > row.skill_record_count
        ):
            raise ValueError(
                "Quantidade de habilidades distintas maior que "
                "a quantidade de registros para "
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
                "Total de eventos educacionais inconsistente para "
                f"{row.customer_id} em "
                f"{row.reference_month}."
            )

        if row.education_event_count == 0:
            raise ValueError(
                "Linha educacional sem eventos para "
                f"{row.customer_id} em "
                f"{row.reference_month}."
            )


def main() -> None:
    datasets = {
        dataset.destination.name: dataset
        for dataset in KAGGLE_DATASETS
    }

    education_database = (
        datasets["education"].destination
        / "database.db"
    )

    with closing(
        sqlite3.connect(education_database)
    ) as connection:
        education_months = load_education_months(
            connection
        )

    validate_education_months(education_months)

    unique_customers = {
        row.customer_id
        for row in education_months
    }

    total_completed_episodes = sum(
        row.completed_episode_count
        for row in education_months
    )

    total_skill_records = sum(
        row.skill_record_count
        for row in education_months
    )

    total_rewards = sum(
        row.reward_count
        for row in education_months
    )

    print("Validação da base mensal educacional: OK")
    print(f"Total de linhas: {len(education_months):,}")
    print(
        "Clientes com atividade educacional: "
        f"{len(unique_customers):,}"
    )
    print(
        "Período: "
        f"{min(row.reference_month for row in education_months)} "
        "até "
        f"{max(row.reference_month for row in education_months)}"
    )
    print(
        "Episódios concluídos vinculados: "
        f"{total_completed_episodes:,}"
    )
    print(
        "Registros de habilidades vinculados: "
        f"{total_skill_records:,}"
    )
    print(
        "Recompensas vinculadas: "
        f"{total_rewards:,}"
    )

    print("\nPrimeiras 5 linhas:")

    for row in education_months[:5]:
        print(row)


if __name__ == "__main__":
    main()