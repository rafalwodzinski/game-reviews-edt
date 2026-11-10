"""Interfejs graficzny Streamlit integrujący scraper oraz klasyfikator sentymentu recenzji Steam."""

import pandas as pd
import streamlit as st

from src.predictor import SentimentPredictor
from src.scraper import fetch_reviews


@st.cache_resource
def get_predictor() -> SentimentPredictor:
    """Ładuje i buforuje w pamięci RAM instancję SentimentPredictor.

    Dzięki dekoratorowi @st.cache_resource model HuggingFace i tokenizator są ładowane
    tylko raz przy pierwszym uruchomieniu, a nie po każdym kliknięciu w interfejsie.
    """
    return SentimentPredictor()


def main():
    st.set_page_config(
        page_title="Steam Reviews Sentiment Analyzer",
        page_icon="🎮",
        layout="wide",
    )

    st.title("🎮 Steam Reviews Sentiment Analyzer")
    st.markdown(
        "Aplikacja łącząca **Scraper Steam Web API** oraz **pre-trenowany Transformer (DistilBERT)** "
        "do automatycznej analizy sentymentu opinii graczy."
    )

    # Pole tekstowe na Steam App ID (domyślnie 730 dla CS:GO / Counter-Strike 2)
    col_input, col_num = st.columns([3, 1])
    with col_input:
        app_id = st.text_input(
            label="Steam App ID:",
            value="730",
            help="Identyfikator gry ze sklepu Steam (np. 730 dla CS2, 1091500 dla Cyberpunk 2077, 1086940 dla Baldur's Gate 3)",
        )
    with col_num:
        num_reviews = st.number_input(
            label="Liczba recenzji:",
            min_value=5,
            max_value=100,
            value=20,
            step=5,
            help="Liczba najnowszych recenzji do pobrania i przeanalizowania (maks. 100 na zapytanie).",
        )

    button_clicked = st.button("🚀 Pobierz i analizuj sentyment", type="primary")

    if button_clicked:
        if not app_id.strip():
            st.error("Proszę podać prawidłowy Steam App ID!")
            return

        with st.spinner("1/2: Pobieranie recenzji ze Steam API..."):
            try:
                df_reviews = fetch_reviews(app_id=app_id.strip(), num_reviews=int(num_reviews))
            except Exception as e:
                st.error(f"Wystąpił błąd podczas komunikacji ze Steam API: {e}")
                return

        if df_reviews.empty:
            st.warning("Nie znaleziono recenzji w języku angielskim dla podanej gry.")
            return

        with st.spinner("2/2: Inferencja modelu Transformera..."):
            try:
                predictor = get_predictor()
                df_predicted = predictor.predict(df_reviews, text_col="raw_text")
            except Exception as e:
                st.error(f"Błąd podczas predykcji sentymentu: {e}")
                return

        st.success(f"Pomyślnie przetworzono {len(df_predicted)} recenzji!")

        # Podsumowanie metryk
        total = len(df_predicted)
        pos_count = (df_predicted["sentiment_label"] == "POSITIVE").sum()
        neg_count = (df_predicted["sentiment_label"] == "NEGATIVE").sum()

        m1, m2, m3 = st.columns(3)
        m1.metric("Liczba przeanalizowanych recenzji", total)
        m2.metric("Pozytywne (Model)", f"{pos_count} ({(pos_count / total):.1%})")
        m3.metric("Negatywne (Model)", f"{neg_count} ({(neg_count / total):.1%})")

        # Tabela wynikowa z kolumną sentiment_label
        st.subheader("📊 Zestawienie wyników")
        st.dataframe(
            df_predicted[["review_id", "raw_text", "voted_up", "sentiment_label"]],
            use_container_width=True,
            column_config={
                "review_id": st.column_config.TextColumn("ID Recenzji"),
                "raw_text": st.column_config.TextColumn("Treść opinii", width="large"),
                "voted_up": st.column_config.CheckboxColumn("Rekomendacja Steam"),
                "sentiment_label": st.column_config.TextColumn("Predykcja Transformera"),
            },
        )


if __name__ == "__main__":
    main()

