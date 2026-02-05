# Klasyfikacja danych tabelarycznych z wykorzystaniem dużych modeli językowych

Repozytorium zawiera implementację narzędzi programistycznych powstałych w ramach pracy inżynierskiej, której celem było zbadanie możliwości wykorzystania dużych modeli językowych w klasyfikacji danych tabelarycznych.

## Spis treści

- [Szybki start](#szybki-start)
- [Opis projektu](#opis-projektu)
- [Struktura repozytorium](#struktura-repozytorium)
- [Wymagania systemowe](#wymagania-systemowe)
- [Instalacja - krok po kroku](#instalacja---krok-po-kroku)
- [Biblioteka TabuLLaM](#biblioteka-tabullam)
- [Aplikacja eksperymentalna](#aplikacja-eksperymentalna)
- [Uruchamianie testów](#uruchamianie-testów)

## Szybki start

Projekt używa [uv](https://docs.astral.sh/uv/getting-started/installation) do zarządzania zależnościami. 

Podstawowe kroki do uruchomienia projektu:

```bash
# 1. Sklonowanie repozytorium
git clone https://github.com/m-baj/TabuLLaM.git
cd TabuLLaM

# 2. Instalacja zależności
uv sync

# 3. (Opcjonalnie) Konfiguracja kluczy API w pliku .env
# Dla modeli Ollama lokalnie krok ten nie jest wymagany

# 4. Uruchomienie aplikacji eksperymentalnej w trybie interaktywnym
make run

# Lub uruchomienie testów w celu weryfikacji instalacji
make test
```

Szczegółowe instrukcje znajdują się poniżej.

## Opis projektu

W repozytorium znadują się:

1. **Biblioteka `tabullam`** - moduł Python implementujący klasyfikator zgodny z interfejsem scikit-learn, który wykorzystuje modele LLM do predykcji klas na podstawie cech tabelarycznych.

2. **Aplikacja eksperymentalna `experiment_app`** - narzędzie do przeprowadzania eksperymentów porównawczych na różnych zbiorach danych, modelach, trybach i konfiguracjach.

### Główne funkcjonalności

- Klasyfikacja danych tabelarycznych z wykorzystaniem LLM.
- Trzy tryby klasyfikacji: _zero-shot_, losowy _few-shot_, semantyczny _few-shot_.
- Obsługa wielu dostawców LLM: LangChain, Ollama (modele lokalne).
- Pełna zgodność z interfejsem scikit-learn (`fit`, `predict`, `predict_proba`).
- Możliwość estymacji prawdopodobieństw klas.
- Modułowa architektura umożliwiająca rozszerzanie o nowe implementacje.

## Struktura repozytorium

```
TabuLLaM/
├── tabullam/                    # Główna biblioteka
│   ├── __init__.py
│   ├── classifier.py            # TabularLLMClassifier - główny klasyfikator
│   ├── exceptions.py            # Definicje wyjątków
│   ├── base/                    # Klasy bazowe (abstrakcyjne)
│   │   ├── llm_backend.py
│   │   ├── prompt_builder.py
│   │   ├── response_parser.py
│   │   └── vector_store.py
│   ├── llm_backends/            # Implementacje dostawców modeli LLM
│   │   ├── openai.py
│   │   ├── ollama.py
│   │   ├── google.py
│   │   └── langchain.py
│   ├── embeddings/              # Moduły do generowania zanurzeń
│   │   ├── openai.py
│   │   ├── ollama.py
│   │   └── sentence_transformers.py
│   ├── prompt_builders/         # Konstruktory promptów
│   ├── response_parsers/        # Parsery odpowiedzi LLM
│   ├── vector_stores/           # Magazyny wektorów (dla RAG)
│   ├── utils/                   # Funkcje pomocnicze
│   └── tests/                   # Testy jednostkowe
├── experiment_app/              # Aplikacja eksperymentalna
│   ├── __main__.py              # Punkt wejścia aplikacji
│   ├── config.py                # Konfiguracja eksperymentów
│   ├── datasets.py              # Ładowanie zbiorów danych
│   ├── executor.py              # Wykonywanie eksperymentów
│   └── runner.py                # Interaktywny interfejs
├── data/                        # Zbiory danych (4 dołączone z repozytorium)
├── experiment-results/          # Wyniki eksperymentów (generowane)
├── pyproject.toml               # Konfiguracja projektu i zależności
├── uv.lock                      # Zablokowane wersje zależności
├── Makefile                     # Skrypty automatyzacji (make)
├── .env.example                 # Szablon zmiennych środowiskowych
└── README.md
```

## Wymagania systemowe

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/) - menedżer pakietów i środowisk Python
- Dostęp do co najmniej jednego z modeli:
  - Klucz API OpenAI (zmienna środowiskowa `OPENAI_API_KEY`)
  - Lokalnie uruchomiony serwer Ollama
  - Klucz API Google (zmienna środowiskowa `GOOGLE_API_KEY`)

## Instalacja - krok po kroku

### 1. Instalacja uv (jeśli nie jest zainstalowane)

```bash
# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Lub za pomocą pip
pip install uv
```

### 2. Klonowanie repozytorium

```bash
git clone <url-repozytorium>
cd TabuLLaM
```

### 3. Instalacja zależności

```bash
# Utworzenie środowiska wirtualnego i instalacja wszystkich zależności
uv sync
```

Komenda `uv sync` automatycznie:
- Tworzy środowisko wirtualne w katalogu `.venv`
- Instaluje wszystkie zależności z pliku `pyproject.toml` i `uv.lock`
- Instaluje bibliotekę `tabullam` w trybie edytowalnym

### 4. Konfiguracja kluczy API

Należy utworzyć plik `.env` w katalogu głównym projektu z kluczami API:

```bash
# Przykład pliku .env (opcjonalnie - w zależności od używanego modelu)
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...
```

Plik `.env.example` zawiera szablon wymaganych zmiennych środowiskowych.

**Uwaga:** W przypadku używania lokalnych modeli Ollama klucze API nie są wymagane. Wymagane jest jedynie uruchomienie serwera Ollama:

```bash
# Uruchomienie serwera Ollama (dla modeli lokalnych)
ollama serve
```

### 5. Weryfikacja instalacji

Aby zweryfikować poprawność instalacji, można uruchomić testy:

```bash
# Uruchomienie testów jednostkowych
make test
```

Wszystkie testy powinny zakończyć się sukcesem.

## Biblioteka TabuLLaM

Biblioteka `tabullam` implementuje klasyfikator `TabularLLMClassifier`, który jest w pełni zgodny z interfejsem scikit-learn.

### Podstawowe użycie

```python
from tabullam import TabularLLMClassifier
import pandas as pd

# Przygotowanie danych
X_train = pd.DataFrame({
    'age': [25, 45, 35],
    'income': [30000, 80000, 55000]
})
y_train = ['low', 'high', 'medium']

# Utworzenie i trenowanie klasyfikatora
clf = TabularLLMClassifier(
    llm='openai:gpt-5-mini',
    mode='zero_shot',
    task_description='Predict customer spending category'
)
clf.fit(X_train, y_train)

# Predykcja
predictions = clf.predict(X_test)
probabilities = clf.predict_proba(X_test)
```

### Tryby klasyfikacji

Biblioteka obsługuje trzy tryby klasyfikacji:

| Tryb | Opis |
|------|------|
| `zero_shot` | Klasyfikacja bez przykładów - model otrzymuje tylko opis zadania i cechy do sklasyfikowania |
| `random_few_shot` | Losowe przykłady z zestawu treningowego dołączane do promptu |
| `semantic_few_shot` | Przykłady wybierane na podstawie podobieństwa semantycznego (RAG) - najbardziej podobne do klasyfikowanej instancji |

Przykład użycia trybu semantic few-shot z lokalnymi embeddingami:

```python
from tabullam import TabularLLMClassifier
from tabullam.embeddings import SentenceTransformerEmbedder

embedder = SentenceTransformerEmbedder(model='all-MiniLM-L6-v2')

clf = TabularLLMClassifier(
    llm='ollama:llama3.1:8b',
    mode='semantic_few_shot',
    k_shots=5,
    embedder=embedder
)
```

### Obsługiwane backendy LLM

| Backend | Format identyfikatora | Przykład |
|---------|----------------------|----------|
| OpenAI | `openai:<model>` | `openai:gpt-5-mini` |
| Ollama | `ollama:<model>` | `ollama:llama3.1:8b` |
| Google Gemini | `google:<model>` | `google:gemini-1.5-flash` |
| Langchain | `langchain:<provider>:<model>` | `langchain:anthropic:claude-3-sonnet` |

### Integracja ze scikit-learn

Klasyfikator jest w pełni zgodny z interfejsem scikit-learn:

```python
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

# Walidacja krzyżowa
scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')

# Użycie w pipeline
pipeline = Pipeline([
    ('classifier', TabularLLMClassifier())
])
```

## Aplikacja eksperymentalna

Aplikacja `experiment_app` umożliwia systematyczne przeprowadzanie eksperymentów klasyfikacyjnych z zapisem wyników i obliczaniem metryk.

### Pierwsze uruchomienie

Przed pierwszym uruchomieniem aplikacji należy upewnić się, że:
1. Skonfigurowane są klucze API lub uruchomiony jest serwer Ollama (patrz: [Konfiguracja kluczy API](#4-konfiguracja-kluczy-api))
2. W katalogu `data/` znajdują się zbiory danych (dostarczone z repozytorium)

**Uwaga o zbiorach danych:** Repozytorium zawiera 4 gotowe zbiory danych w katalogu `data/`:
- `breast_cancer`
- `caesarian`
- `car_evaluation`
- `wine_quality`

Są to 4 z 6 zbiorów danych używanych w eksperymentach. Pozostałe dwa zbiory (`bank_marketing` i `adult`) nie zostały dołączone do repozytorium ze względu na ich duży rozmiar, mogą jednak zostać pobrane i dodane ręcznie.

### Uruchomienie aplikacji

```bash
# Tryb interaktywny
make run

# Z pliku konfiguracyjnego
uv run python -m experiment_app -c experiment_config.yaml
```

### Tryb interaktywny - przewodnik

Po uruchomieniu w trybie interaktywnym aplikacja przeprowadza przez proces konfiguracji eksperymentu:

1. **Wybór zbiorów danych** - możliwy wybór jednego lub więcej zbiorów z dostępnych
2. **Wybór modeli LLM** - np. `openai:gpt-5-mini` lub `ollama:llama3.1:8b`
3. **Wybór trybów klasyfikacji** - `zero_shot`, `random_few_shot`, `semantic_few_shot`
4. **Parametry eksperymentu**:
   - Liczba przykładów few-shot (`k_shots`)
   - Maksymalna liczba próbek do przetestowania
   - Rozmiar zestawu testowego
   - Ziarna losowości (seeds) dla powtarzalności
5. **Tryb predykcji** - `predict` (tylko etykiety) lub `predict_proba` (prawdopodobieństwa)

Po zakończeniu konfiguracji aplikacja automatycznie przeprowadza eksperymenty i zapisuje wyniki.

### Konfiguracja eksperymentów

Eksperymenty można konfigurować poprzez plik YAML:

```yaml
experiments:
  - name: "Porównanie modeli"
    datasets:
      - adults_income
      - breast_cancer
    models:
      - openai:gpt-5-mini
      - ollama:llama3.1:8b
    modes:
      - zero_shot
      - semantic_few_shot
    seeds: [42, 7, 123]
    k_shots: 5
    max_samples: 500
    test_size: 0.2
    prediction_mode: predict_proba
```

### Dostępne zbiory danych

#### Zbiory dołączone do repozytorium (katalog `data/`)

| Nazwa | Opis | Typ klasyfikacji | Liczba próbek |
|-------|------|------------------|---------------|
| `breast_cancer` | Klasyfikacja nawrotu nowotworu | Binarna | 286 |
| `caesarian` | Predykcja konieczności cesarskiego cięcia | Binarna | 80 |
| `car_evaluation` | Ocena akceptowalności samochodu | Wieloklasowa | 1728 |
| `wine_quality` | Predykcja oceny wina (0-10) | Wieloklasowa | 6497 |

### Wyniki eksperymentów

Wyniki są zapisywane w następującej strukturze katalogów:

```
experiment-results/
└── <dataset>/
    └── <model>/
        └── <mode>_<k_shots>/
            └── <timestamp>_<prediction_mode>/
                ├── config.json
                ├── run_1_seed_42_results.json
                ├── run_2_seed_7_results.json
                └── metrics_summary.json
```

Dla każdego eksperymentu obliczane są następujące metryki:
- w przypadku bezpośredniej klasyfikacji:
  - Accuracy
  - F1-score (macro)
  - Matthews Correlation Coefficient (MCC)
- w przypadku wyrażania pewności:
  - ROC AUC
  - PR AUC

## Uruchamianie testów

Projekt zawiera kompleksowy zestaw testów jednostkowych (194 testy).

```bash
# Wszystkie testy
make test

# Testy z pokryciem kodu
make test-cov

# Konkretny plik testowy
make test-file FILE=tabullam/tests/test_classifier.py

# Testy pasujące do wzorca
make test-match MATCH=test_classifier

# Lub bezpośrednio z uv:
uv run pytest tabullam/tests -v
uv run pytest tabullam/tests --cov=tabullam --cov-report=term-missing
```

Wszystkie komendy `make` są zdefiniowane w pliku `Makefile` w katalogu głównym projektu. Lista wszystkich dostępnych komend:

```bash
make help
```

## Rozwiązywanie problemów

### Problem: `uv: command not found`

**Rozwiązanie:** Należy zainstalować `uv` zgodnie z instrukcjami w sekcji [Instalacja - krok po kroku](#instalacja---krok-po-kroku).

### Problem: Błąd API "Authentication failed" lub "Invalid API key"

**Rozwiązanie:**
1. Należy upewnić się, że plik `.env` zawiera poprawne klucze API
2. Weryfikacja poprawności zmiennych środowiskowych:
   ```bash
   echo $OPENAI_API_KEY
   ```
3. W przypadku używania modeli lokalnych Ollama należy upewnić się, że serwer Ollama jest uruchomiony:
   ```bash
   ollama serve
   ```

### Problem: Brak zbiorów danych

**Rozwiązanie:** Należy upewnić się, że katalog `data/` zawiera pliki parquet ze zbiorami danych. Zbiory powinny być automatycznie dołączone do sklonowanego repozytorium.

### Problem: Testy nie przechodzą

**Rozwiązanie:**
1. Weryfikacja instalacji zależności:
   ```bash
   uv sync
   ```
2. Weryfikacja wersji Python (wymagane >= 3.12):
   ```bash
   python --version
   ```

### Problem: Import error - brak modułu `tabullam`

**Rozwiązanie:** Należy uruchamiać komendy z prefiksem `uv run`, który aktywuje środowisko wirtualne:
```bash
uv run python -m experiment_app
uv run pytest tabullam/tests
```

Alternatywnie można ręcznie aktywować środowisko wirtualne:
```bash
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows
```

## Licencja

Projekt powstał w ramach pracy inżynierskiej.
