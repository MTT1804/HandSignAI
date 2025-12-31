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
   git clone https://github.com/MTT1804/HandSignAI.git
   cd HandSignAI
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
HandSignAI/
├── main/                    # Kod aplikacji + dane runtime
│   ├── main.py              # Punkt wejścia uruchamiający GUI
│   ├── ctk_app/             # Aplikacja CustomTkinter (modułowa)
│   │   ├── app.py           # Kontroler aplikacji (widoki, kamera, trening)
│   │   ├── training_worker.py # Trening modelu (TensorFlow/Keras)
│   │   └── views/           # Widoki GUI
│   ├── locales/             # Tłumaczenia (PL/EN)
│   ├── images/              # Zapisane obrazy dłoni (snapshoty)
│   ├── data/                # data.csv: cechy + label + index
│   ├── models/              # Wytrenowany model (.h5)
│   ├── other/               # settings.json, scaler (.pkl), logi, motywy
│   └── text_files/          # Pliki tekstowe do zakładki Tekst
├── requirements.txt         # Lista zależności Pythona
├── README.md
└── README.pl.md
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

Po uruchomieniu aplikacji dostępnych jest sześć głównych zakładek:

1. **Zbieranie danych** – zapisuje snapshot dłoni oraz landmarki do pliku CSV i folderu `main/images/`.
2. **Detekcja znaków** – rozpoznaje litery w czasie rzeczywistym i wyświetla top-10 prawdopodobieństw.
3. **Trening modelu** – ustawia parametry i trenuje model, zapisując model (`.h5`) oraz skaler (`.pkl`).
4. **Tekst** – porównuje rozpoznane litery z dostarczonym plikiem tekstowym.
5. **Instrukcja** – wbudowana instrukcja użycia aplikacji.
6. **Ustawienia** – konfiguracja ścieżek, motywu/języka oraz parametrów detekcji i MediaPipe.

Szczegółowa instrukcja jest dostępna w aplikacji, w zakładce **Instrukcja**.

## Szybki start – od kolekcji danych do detekcji

Poniższe kroki pomogą Ci błyskawicznie uruchomić cały pipeline: zebranie danych, trening modelu i detekcję w czasie rzeczywistym.

### 1. Zbieranie danych

1. **Uruchom aplikację** i przejdź do zakładki **„Zbieranie danych”**.  
2. **Wybierz kamerę** i wpisz aktualnie nagrywaną literę/cyfrę.  
3. **Ustaw dłoń w wyraźnym świetle**, unikaj cieni i prześwietleń.  
4. **Kliknij „Zapisz dane”** (lub naciśnij Enter), aby zapisać współrzędne 21 landmarków dłoni do CSV oraz zdjęcie do folderu `main/images/` 
5. Powtórz dla każdej klasy, zbierając co najmniej **100–200 próbek** na klasę, poruszając dłonią w różnych kątach.

### 2. Struktura pliku CSV

- Plik `main/data/data.csv` zawiera jeden wiersz na jedną próbkę. Domyślnie obsługuje do **4 dłoni** (brakujące dłonie są wypełniane zerami):
   - `h1_x0, h1_y0, …, h1_x20, h1_y20` (landmarki dłoni #1)
   - …
   - `h4_x0, h4_y0, …, h4_x20, h4_y20` (landmarki dłoni #4)
   - `label` (klasa)
   - `index` (numer próbki)

Razem daje to **170 kolumn** w domyślnej konfiguracji (4 dłonie × 21 landmarków × 2 współrzędne + `label` + `index`).

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
