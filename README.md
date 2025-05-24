# Projekt Dyplomowy

Projekt w ramach przedmiotu **Projekt Dyplomowy** na semestrze 6 Informatyki Stosowanej na Politechnice Warszawskiej.

## Spis treści

* [Opis](#opis)
* [Wymagania wstępne](#wymagania-wstępne)
* [Instalacja](#instalacja)
* [Struktura projektu](#struktura-projektu)
* [Uruchomienie](#uruchomienie)
* [Użycie](#użycie)

## Opis

Aplikacja służy do

* zbierania danych dłoni i landmarków,
* trenowania modelu sieci neuronowej,
* detekcji liter w czasie rzeczywistym,
* rozpoznawania tekstu migowego według dostarczonego pliku tekstowego.

## Wymagania wstępne

* Python **3.10.11**
* Virtualenv
* System operacyjny: Windows, Linux lub macOS

## Instalacja

1. Sklonuj repozytorium:

   ```bash
   git clone https://github.com/MTT555/Projekt_dyplomowy.git
   cd Projekt_dyplomowy
   ```
2. Utwórz i aktywuj wirtualne środowisko:

   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux / macOS
   source venv/bin/activate
   ```
3. Zainstaluj zależności:

   ```bash
   pip install -r requirements.txt
   ```

## Struktura projektu

```text
Projekt_dyplomowy/
├── main/                   # Główny kod aplikacji
│   ├── app_main.py         # Logika GUI i główna klasa aplikacji
│   ├── gui_collect.py      # Zakładka zbierania danych
│   ├── gui_train.py        # Zakładka treningu modelu
│   ├── gui_detection.py    # Zakładka detekcji znaków
│   ├── gui_text_detection.py # Zakładka wykrywania tekstu migowego
│   ├── gui_instructions.py # Zakładka instrukcji w aplikacji
│   ├── detection.py        # Funkcje detekcji dłoni i predykcji
│   ├── training.py         # Skrypt trenowania modelu
│   └── utils.py            # Pomocnicze funkcje
├── images/                 # Zapisane obrazy dłoni
├── data/                   # Plik CSV z danymi landmarków
├── models/                 # Wytrenowane modele (.h5) i skalery (.pkl)
├── other/                  # Logi aplikacji i inne pliki
├── text_files/             # Przykładowe pliki tekstowe do detekcji
├── requirements.txt        # Lista zależności Pythona
└── README.md               # Ten plik
```

## Uruchomienie

1. Przejdź do katalogu z kodem aplikacji:

   ```bash
   cd main
   ```
2. Uruchom aplikację:

   ```bash
   python main.py
   ```

## Użycie

Po uruchomieniu aplikacji dostępne są cztery główne zakładki:

1. **Zbieranie danych** – zapisuje obraz dłoni oraz współrzędne landmarków do pliku CSV i folderu `images/`.
2. **Trening modelu** – ustawia parametry (test size, random state, epochs, batch size, patience) i trenuje model, zapisując wagę modelu i skaler.
3. **Detekcja znaków** – rozpoznaje litery w czasie rzeczywistym i wyświetla top10 prawdopodobieństw.
4. **Miganie tekstu** – porównuje rozpoznane litery z dostarczonym plikiem tekstowym i podświetla poprawne znaki.

Szczegółowa instrukcja jest dostępna w aplikacji, w zakładce **Instrukcja**.

## Szybki start – od kolekcji danych do detekcji

Poniższe kroki pomogą Ci błyskawicznie uruchomić cały pipeline: zebranie danych, trening modelu i detekcję w czasie rzeczywistym.

### 1. Zbieranie danych

1. **Uruchom aplikację** i przejdź do zakładki **„Zbieranie danych”**.  
2. **Wybierz kamerę** i wpisz aktualnie nagrywaną literę/cyfrę.  
3. **Ustaw dłoń w wyraźnym świetle**, unikaj cieni i prześwietleń.  
4. **Kliknij „Zapisz dane”** (lub naciśnij Enter), aby zapisać współrzędne 21 landmarków dłoni do CSV oraz zdjęcie do folderu `images/` 
5. Powtórz dla każdej klasy, zbierając co najmniej **100–200 próbek** na klasę, poruszając dłonią w różnych kątach.

### 2. Struktura pliku CSV

- Plik `data/data.csv` zawiera wiersze o długości **44 kolumn**:  
  - `x0, y0, x1, y1, …, x20, y20` (współrzędne)  
  - `label` (litera/cyfra)  
  - `index` (numer próbki)

Sprawdź nagłówek, żeby upewnić się, że wszystkie kolumny są obecne.

### 3. Trening modelu

1. Przejdź do zakładki **„Trening modelu”**.  
2. **Wskaż ścieżkę** do CSV, pliku do zapisu modelu (`.h5`) i skalera (`.pkl`).  
3. Ustaw parametry:
   - **Test size**: 0.1–0.2  
   - **Batch size**: 16–32  
   - **Epochs**: 20–50  
   - **Patience**: 5
  Lub ustaw własne
4. Kliknij **„Rozpocznij trening”**. Proces odbywa się w tle, a postęp widać na pasku.  
5. Po zakończeniu zobaczysz zapisany model i skalera oraz wynik accuracy na zbiorze testowym

### 4. Detekcja w czasie rzeczywistym

1. Przejdź do zakładki **„Detekcja znaków”**.  
2. Upewnij się, że masz załadowany właściwy model i skaler.  
3. **Ustaw interwał** (np. 1000 ms) i **próg pewności** (np. 0.7).  
4. Kliknij **„Start Detekcji”** – rozpoznane litery będą pojawiać się w polu tekstowym.  
5. (Opcjonalnie) Włącz tryb **„wstawiaj znak tylko po Enterze”**, by potwierdzać wyniki ręcznie

---

*Teraz wystarczy uruchomić appkę i od razu zacząć zbierać, trenować i testować rozpoznawanie znaków!*  
