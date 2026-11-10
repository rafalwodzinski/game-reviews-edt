"""Moduł pobierania recenzji gier ze Steam Web API bez użycia klucza API."""

from pathlib import Path
import pandas as pd
import requests


def fetch_reviews(app_id: str, num_reviews: int = 100) -> pd.DataFrame:
    """Pobiera recenzje ze Steam Web API dla podanego ID gry (app_id).

    Odpytuje endpoint Steam zdefiniowany w specyfikacji:
    https://store.steampowered.com/appreviews/{app_id}?json=1&language=english

    Uwaga architektoniczna:
    Valve wycofało stary endpoint na domenie store.steampowered.com (zwraca HTTP 302
    przekierowujący na stronę główną HTML sklepu). Funkcja wykrywa tę sytuację
    i automatycznie odpytuje aktywny, oficjalny endpoint Web API Valve:
    https://api.steampowered.com/IUserReviewsService/GetAppReviews/v1/

    Args:
        app_id: Identyfikator aplikacji na Steam (np. '1091500' dla Cyberpunk 2077).
        num_reviews: Liczba recenzji do pobrania (domyślnie 100).

    Returns:
        pd.DataFrame: Ramka danych zawierająca wyłącznie kolumny:
                      ['review_id', 'raw_text', 'voted_up'].
    """
    legacy_url = f"https://store.steampowered.com/appreviews/{app_id}?json=1&language=english"
    api_url = "https://api.steampowered.com/IUserReviewsService/GetAppReviews/v1/"
    headers = {
        "User-Agent": "SteamReviewScraper/1.0 (Educational NLP EDT Project)"
    }

    reviews_data = []
    cursor = "*"
    use_new_api = False

    while len(reviews_data) < num_reviews:
        per_page = min(100, num_reviews - len(reviews_data))

        if not use_new_api:
            # Próba odpytania URL ze specyfikacji (z blokadą śledzenia przekierowania do HTML)
            params = {
                "num_per_page": per_page,
                "cursor": cursor,
                "purchase_type": "all",
                "filter": "recent",
            }
            try:
                response = requests.get(
                    legacy_url,
                    params=params,
                    headers=headers,
                    timeout=10,
                    allow_redirects=False,
                )
                # Jeśli serwer przekierowuje (302) lub nie zwraca JSON-a, przełączamy na Web API
                if response.status_code in (301, 302, 303, 307) or "json" not in response.headers.get("Content-Type", ""):
                    use_new_api = True
                else:
                    response.raise_for_status()
                    payload = response.json()
                    batch = payload.get("reviews", [])
                    new_cursor = payload.get("cursor")
            except Exception:
                use_new_api = True

        if use_new_api:
            # Aktywny publiczny endpoint Valve (nie wymaga klucza API dla podstawowych zapytań)
            params = {
                "appid": app_id,
                "language": "english",
                "num_per_page": per_page,
                "cursor": cursor,
                "purchase_type": "all",
                "filter": "recent",
            }
            response = requests.get(
                api_url,
                params=params,
                headers=headers,
                timeout=10,
            )
            response.raise_for_status()
            payload = response.json()
            resp_body = payload.get("response", {})
            batch = resp_body.get("reviews", [])
            new_cursor = resp_body.get("cursor")

        if not batch:
            break

        for item in batch:
            reviews_data.append({
                "review_id": str(item.get("recommendationid", "")),
                "raw_text": str(item.get("review", "")),
                "voted_up": bool(item.get("voted_up", False)),
            })
            if len(reviews_data) >= num_reviews:
                break

        if not new_cursor or new_cursor == cursor:
            break
        cursor = new_cursor

    df = pd.DataFrame(reviews_data, columns=["review_id", "raw_text", "voted_up"])
    return df


if __name__ == "__main__":
    APP_ID = "1091500"  # Cyberpunk 2077
    TARGET_COUNT = 100

    project_root = Path(__file__).resolve().parent.parent
    output_path = project_root / "data" / "contract_sample.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Pobieranie {TARGET_COUNT} recenzji dla app_id={APP_ID}...")
    df_reviews = fetch_reviews(app_id=APP_ID, num_reviews=TARGET_COUNT)

    # Walidacja kontraktu przed zapisem
    expected_columns = ["review_id", "raw_text", "voted_up"]
    if list(df_reviews.columns) != expected_columns:
        raise ValueError(
            f"Błąd kontraktu: kolumny {list(df_reviews.columns)} != {expected_columns}"
        )
    if len(df_reviews) != TARGET_COUNT:
        raise ValueError(
            f"Błąd kontraktu: oczekiwano {TARGET_COUNT} wierszy, pobrano {len(df_reviews)}"
        )

    df_reviews.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Zapisano pomyślnie {len(df_reviews)} recenzji do {output_path} (wymiary: {df_reviews.shape}).")
