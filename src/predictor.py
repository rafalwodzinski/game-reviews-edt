"""Moduł klasyfikacji sentymentu recenzji za pomocą pre-trenowanego Transformera (HuggingFace)."""

from pathlib import Path
from typing import List
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def validate_input_tensor(input_ids: torch.Tensor, max_len: int = 512) -> None:
    """Weryfikuje poprawność wymiarów tensora wejściowego zgodnie z żelaznymi zasadami architektonicznymi.

    Wymagania architektoniczne:
    1. input_ids musi być tensorem 2D o kształcie [batch_size, seq_len].
    2. seq_len nie może przekraczać max_length=512 (gwarantowane przez truncation=True).

    Args:
        input_ids: Tensor identyfikatorów tokenów z tokenizatora.
        max_len: Maksymalna dozwolona długość sekwencji (domyślnie 512).
    """
    if input_ids.ndim != 2:
        raise AssertionError(
            f"Błąd architektoniczny: input_ids musi być tensorem 2D [batch_size, seq_len], "
            f"otrzymano tensor o wymiarze {input_ids.ndim}D: {input_ids.shape}"
        )
    if input_ids.shape[1] > max_len:
        raise AssertionError(
            f"Błąd architektoniczny: przekroczono dopuszczalny limit długości {max_len}: "
            f"rzeczywisty seq_len={input_ids.shape[1]}. Sprawdź nałożenie flagi truncation=True!"
        )
    print(f"[Transformer Architecture Check] Tensor 2D input_ids shape: {input_ids.shape} ([batch_size, seq_len])")


class SentimentPredictor:
    """Klasyfikator sentymentu oparty na architekturze Transformer."""

    DEFAULT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = None):
        """Inicjalizuje model i tokenizator HuggingFace.

        Zgodnie z zasadami architektonicznymi:
        - NIE usuwamy stop-words,
        - NIE lematyzujemy,
        - Pilnujemy max_length=512, truncation=True oraz dynamicznego paddingu.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Ładowanie modelu '{model_name}' na urządzenie: {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, df: pd.DataFrame, text_col: str = "raw_text", batch_size: int = 100) -> pd.DataFrame:
        """Generuje etykietę sentymentu ('POSITIVE' lub 'NEGATIVE') dla każdej recenzji.

        Args:
            df: Wejściowa ramka danych zawierająca kolumnę z surowym tekstem.
            text_col: Nazwa kolumny z tekstem (domyślnie 'raw_text').
            batch_size: Rozmiar partii danych (batch) przekazywanej do modelu.

        Returns:
            pd.DataFrame: Kopia ramki wejściowej z dołączoną kolumną 'sentiment_label'.
        """
        if text_col not in df.columns:
            raise KeyError(f"Brak kolumny '{text_col}' w ramce danych. Dostępne kolumny: {list(df.columns)}")

        result_df = df.copy()
        texts: List[str] = [str(text) if pd.notna(text) else "" for text in df[text_col]]
        all_labels: List[str] = []

        # Przetwarzanie wsadowe (batch processing)
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            # Tokenizacja z żelaznymi zasadami: padding dynamiczny, truncation=True, max_length=512
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )

            # Weryfikacja wymiaru tensora wejściowego [batch_size, seq_len]
            validate_input_tensor(encoded["input_ids"], max_len=512)

            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)

            # Inferencja z wyłączeniem obliczania gradientów
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probs = F.softmax(logits, dim=1)
                predicted_class_ids = torch.argmax(probs, dim=1).cpu().tolist()

            batch_labels = [self.model.config.id2label[class_id] for class_id in predicted_class_ids]
            all_labels.extend(batch_labels)

        result_df["sentiment_label"] = all_labels
        return result_df


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    sample_file = project_root / "data" / "contract_sample.csv"

    if not sample_file.exists():
        raise FileNotFoundError(f"Nie odnaleziono pliku kontraktu: {sample_file}")

    print(f"Wczytywanie próby danych kontraktowych: {sample_file}")
    df_sample = pd.read_csv(sample_file)

    predictor = SentimentPredictor()
    df_result = predictor.predict(df_sample, text_col="raw_text", batch_size=100)

    # Asercje jakościowe DoD
    assert "sentiment_label" in df_result.columns, "Brak kolumny sentiment_label!"
    assert df_result["sentiment_label"].notna().all(), "Kolumna sentiment_label zawiera wartości puste/NaN!"
    assert len(df_result) == len(df_sample), "Niezgodność liczby wierszy po predykcji!"
    assert set(df_result["sentiment_label"].unique()).issubset({"POSITIVE", "NEGATIVE"}), "Nierozpoznane etykiety klas!"

    print("\nPrzykładowe wyniki predykcji (5 pierwszych rekordów):")
    for _, row in df_result[["review_id", "raw_text", "voted_up", "sentiment_label"]].head(5).iterrows():
        preview_text = (row['raw_text'][:60] + "...") if len(str(row['raw_text'])) > 60 else row['raw_text']
        print(f"- ID: {row['review_id']} | Steam: {row['voted_up']} | Model: {row['sentiment_label']} | Text: {preview_text}")

    print(f"\n[Definition of Done SUCCESS] Wygenerowano DataFrame z kolumną 'sentiment_label'.")

