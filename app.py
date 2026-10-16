import pandas as pd
import streamlit as st

from src.modeling.batch_inference import load_saved_model, prepare_features
from src.modeling.export_inference_features import build_inference_features
from src.modeling.modeling_dataset import IDENTIFIER_COLUMNS


def generate_predictions(run_id: str) -> pd.DataFrame:
    model, config = load_saved_model(run_id)
    threshold = float(config["decision_threshold"])

    if not 0 <= threshold <= 1:
        raise ValueError("O limiar deve estar entre 0 e 1.")

    dataset = build_inference_features()
    features = prepare_features(dataset, config["feature_columns"])
    positive_index = list(model.classes_).index(config["positive_class"])
    probabilities = model.predict_proba(features)[:, positive_index]

    result = dataset.loc[:, list(IDENTIFIER_COLUMNS)].copy()
    result["probability_active_next_month"] = probabilities
    result["predicted_active_next_month"] = (
        probabilities >= threshold
    ).astype(int)
    result["decision_threshold"] = threshold
    result["run_id"] = run_id
    return result


def main() -> None:
    st.set_page_config(page_title="Loyalty Predict Learning", layout="wide")
    st.title("Previsão de atividade")
    st.write(
        "Estime quais clientes terão transações no mês "
        "seguinte ao mês de referência disponível."
    )

    with st.form("prediction_form"):
        run_id = st.text_input(
            "Run ID do modelo no MLflow",
            placeholder="Informe uma execução registrada localmente",
        )
        submitted = st.form_submit_button("Gerar previsões")

    if submitted:
        st.session_state.pop("predictions", None)
        if not run_id.strip():
            st.error("Informe o Run ID.")
        else:
            try:
                with st.spinner("Verificando fontes e gerando previsões..."):
                    result = generate_predictions(run_id.strip())
                st.session_state["predictions"] = result
            except Exception:
                st.error(
                    "Não foi possível gerar as previsões. "
                    "Consulte os detalhes no terminal."
                )
                raise

    result = st.session_state.get("predictions")
    if result is None:
        return

    reference_month = pd.Timestamp(result["reference_month"].iloc[0])
    prediction_month = reference_month + pd.DateOffset(months=1)
    threshold = result["decision_threshold"].iloc[0]
    positive_count = int(result["predicted_active_next_month"].sum())

    st.success("Previsões geradas.")
    st.write(
        f"Referência: **{reference_month:%Y-%m}** · "
        f"Mês previsto: **{prediction_month:%Y-%m}**"
    )
    st.caption(f"Modelo utilizado — Run ID: {result['run_id'].iloc[0]}")

    total_column, active_column, threshold_column = st.columns(3)
    total_column.metric("Clientes avaliados", len(result))
    active_column.metric("Classificados como ativos", positive_count)
    threshold_column.metric("Limiar de decisão", f"{threshold:.2f}")

    st.dataframe(
        result.sort_values("probability_active_next_month", ascending=False),
        hide_index=True,
    )
    st.download_button(
        label="Baixar previsões em CSV",
        data=result.to_csv(index=False).encode("utf-8"),
        file_name=f"predictions_{reference_month:%Y-%m}.csv",
        mime="text/csv",
    )
    st.caption(
        "As previsões são estimativas de atividade. "
        "O mês de referência depende dos dados disponíveis."
    )


if __name__ == "__main__":
    main()
