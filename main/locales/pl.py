STRINGS: dict[str, str] = {
    # ---------- zakładki ----------
    "tab_collect":          "Zbieranie danych",
    "tab_train":            "Trening modelu",
    "tab_detection":        "Detekcja znaków",
    "tab_text":             "Miganie tekstu",
    "tab_instr":            "Instrukcja",

    # ---------- wspólne przyciski / etykiety ----------
    "btn_start":            "Start",
    "btn_stop":             "Stop",
    "btn_clear_screen":     "Wyczyść ekran",
    "lbl_interval":         "Interwał (ms):",
    "lbl_threshold":        "Próg:",

    # ---------- zakładka „Zbieranie danych” ----------
    "lbl_choose_camera":    "Wybierz kamerę:",
    "lbl_enter_label":      "Podaj literę/cyfrę do zbierania:",
    "btn_set_label":        "Ustaw literę",
    "btn_save_data":        "Zapisz dane [Enter]",
    "btn_flip":             "Flip w pionie (tab)",
    "section_data_mgmt":    "--- Zarządzanie danymi ---",
    "btn_clear_images":     "Wyczyść folder images",
    "btn_reset_csv":        "Wyzeruj plik CSV",
    "section_reset":        "--- Reset ustawień ---",
    "btn_reset_defaults":   "Przywróć domyślne",
    "btn_quit":             "Wyjdź (q)",

    "section_img_settings": "--- Ustawienia obrazu ---",
    "lbl_brightness":       "Jasność (beta):",
    "lbl_contrast":         "Kontrast (alpha%):",
    "lbl_gamma":            "Gamma (1.0 = brak):",
    "lbl_color_shift":      "Przesunięcie koloru (R, G, B):",
    "lbl_R":                "R:",
    "lbl_G":                "G:",
    "lbl_B":                "B:",

    # ---------- ramka MediaPipe ----------
    "section_mediapipe":    "--- Ustawienia MediaPipe ---",
    "chk_static_img_mode":  "static_image_mode (True = zdjęcia statyczne)",
    "lbl_max_num_hands":    "max_num_hands:",
    "lbl_model_complexity": "model_complexity (0-2):",
    "lbl_min_det_conf":     "min_detection_confidence (%):",
    "lbl_min_track_conf":   "min_tracking_confidence (%):",
    "btn_apply_mp":         "Zastosuj zmiany w MediaPipe",

    "chk_show_overlays":    "Pokaż dodatkowe elementy (tekst, kropki)",

    # ---------- ramka nazw plików ----------
    "frame_file_labels":    "Nazwa plików (domyślne wartości)",
    "lbl_csv_file":         "CSV (wej./wyj.):",
    "lbl_model_file":       "Model (wyj.):",
    "lbl_scaler_file":      "Scaler (wyj.):",

    # ---------- zakładka „Trening” ----------
    "frame_train_config":   "Konfiguracja treningu",
    "lbl_test_size":        "Test size (np. 0.2):",
    "lbl_random_state":     "Random state:",
    "lbl_epochs":           "Epoki:",
    "lbl_batch_size":       "Batch size:",
    "lbl_patience":         "Cierpliwość (EarlyStopping):",
    "btn_start_training":   "Rozpocznij trening",

    # ---------- zakładka „Detekcja znaków” ----------
    "chk_enter_mode":       "Wstawiaj znak tylko po Enterze",
    "btn_start_detection":  "Start detekcji",
    "btn_stop_detection":   "Stop detekcji",

    # ---------- okno z prawdopodobieństwami ----------
    "win_top_probs":        "Najwyższe prawdopodobieństwa",

    # ---------- zakładka „Miganie tekstu” ----------
    "lbl_cam_preview_text": "Podgląd kamery (tekst)",
    "lbl_select_text_file": "Wybierz plik z tekstem:",
    "btn_load_text":        "Wczytaj tekst",

    # ---------- statystyki (place-holdery) ----------
    "stat_correct":         "Poprawnie rozpoznane: {ok} / {total}",
    "stat_failed":          "Błędy (nieudane próby): {fail}",
    "stat_remaining":       "Pozostało znaków: {remain}",

    # ---------- okna dialogowe ----------
    "dlg_confirm":          "Potwierdzenie",
    "dlg_sure_clear_images":
        "Na pewno usunąć cały katalog 'images' wraz z podkatalogami?",
    "dlg_sure_reset_csv":
        "Na pewno wyzerować zawartość pliku {file}?",

    # ---------- okno TOP-10 ----------
    "win_top10":            "Najwyższe prawdopodobieństwa",

    # ---------- logi / komunikaty ----------
    "log_no_csv_path":      "Brak ścieżki do pliku CSV – nie można wykryć klas.",
    "log_no_label_column":  "Brak kolumny 'label' w pliku CSV – nie można wykryć klas.",
    "log_classes_found":    "Wykryto klasy: {classes}",
    "log_camera_switch":    "Zmieniam kamerę z {old} na {new}...",
    "log_saved_sample":     "Zapisano {label} o indeksie {idx} w pliku CSV: {path}.",
    "log_saved_image":      "Zapisano obraz w {path}.",
        "instructions_text": """\
INSTRUKCJA INTERFEJSU
=====================

Program składa się z pięciu zakładek widocznych u góry okna:

1. ZBIERANIE DANYCH
   • Wybierz kamerę z rozwijanej listy na samej górze.
   • Wpisz znak w polu „Litera/Liczba”, kliknij „Ustaw literę”.
   • Trzymając dłoń w kadrze naciśnij „Zapisz dane” lub klawisz Enter –
     każde wciśnięcie zapisuje klatkę jako plik .jpg i wiersz w CSV.
   • Przycisk „Flip w pionie (tab)” lustrzanie odwraca obraz.
   • Suwaki: Jasność, Kontrast, Gamma, R‑G‑B – szybka korekcja obrazu.
   • „Wyczyść folder images” i „Wyzeruj plik CSV” usuwają zebrane dane.

2. TRENING MODELU
   • Domyślne ścieżki do CSV, modelu i skalera są już wypełnione.
   • Zostaw parametry (Test size 0.2, Epochs 30) lub je zmień.
   • Kliknij „Rozpocznij trening” i poczekaj aż pasek dojdzie do 100 % –
     model zapisze się w models/model.h5, a skaler w other/scaler.pkl.

3. DETEKCJA ZNAKÓW
   • Kliknij „Start Detekcji”, aby rozpocząć rozpoznawanie gestów.
   • Rozpoznane litery lądują w dużym polu tekstowym; okno Top‑10 pokazuje
     prawdopodobieństwa sieci.
   • Suwak „Próg” (domyślnie 0.7) filtruje wyniki – podnieś go, jeśli widzisz błędy.

4. MIGANIE TEKSTU
   • Wybierz plik .txt, kliknij „Wczytaj tekst”, potem „Start”.
   • Program podświetla kolejne litery; poprawnie rozpoznane stają się zielone.
   • Panel z prawej zlicza trafienia, błędy i pozostałe znaki.

5. INSTRUKCJA
   • Czytasz ją właśnie tutaj. :-)


SCENARIUSZE STARTOWE
====================

SCENARIUSZ 1 – „Naucz komputer literę A”
  1) Zakładka ZBIERANIE DANYCH → wpisz A → „Ustaw literę”.
  2) Naciśnij Spację ~50 razy, zmień nieco kąt dłoni, zrób kolejne 50 zdjęć.
  3) Gotowe – litera A zebrana.

SCENARIUSZ 2 – „Stwórz model z własnych liter”
  1) Zbierz zdjęcia dla wszystkich liter, które Cię interesują.
  2) Zakładka TRENING MODELU → „Rozpocznij trening” → czekaj do 100 %.
  3) Komunikat „model zapisany” = sukces.

SCENARIUSZ 3 – „Sprawdź, czy rozpoznawanie działa”
  1) Zakładka DETEKCJA ZNAKÓW → „Start Detekcji”.
  2) Pokaż znak A – litera powinna pojawić się w oknie.
  3) Losowe litery? Podnieś „Próg” z 0.7 na 0.8 i kliknij „Stop” → „Start”.

SCENARIUSZ 4 – „Dorzucam nową literę Ł”
  1) Jak w scenariuszu 1, ale z literą Ł (~100 zdjęć).
  2) TRENING MODELU → „Rozpocznij trening” – model aktualizuje się.
  3) DETEKCJA ZNAKÓW → „Start”, sprawdź gest Ł.

SCENARIUSZ 5 – „Ćwicz całe słowa”
  1) MIGANIE TEKSTU → wybierz plik, „Wczytaj tekst” → „Start”.
  2) Gestykuluj po kolei – poprawne litery świecą na zielono, licznik rośnie.


SZYBKA POMOC
============
• Brak obrazu?
  – Zamknij inne aplikacje korzystające z kamery (Zoom, Teams).
  – Spróbuj innego numeru kamery w liście „Kamera”.
• Obraz zamazany lub za ciemny?
  – Podnieś suwak Jasność (beta) i Kontrast (alpha %).
  – Skorzystaj z suwaków Gamma i R‑G‑B.
• Nie wykrywa dłoni?
  – Upewnij się, że dłoń jest w centrum kadru.
  – Sprawdź oświetlenie i odległość od kamery.
• Pusty CSV lub brak kolumny „label”?
  – W zakładce Zbieranie danych dodaj przynajmniej jedną klatkę.
• Błąd podczas treningu („stratify” itp.)?
  – Upewnij się, że masz dane dla każdej klasy (litery).
  – Zbierz więcej przykładów dla brakujących znaków.
• Trening trwa zbyt długo?
  – Zmniejsz liczbę Epochs (np. z 30 na 10).
  – Obniż Batch size.
• Błąd ładowania modelu (.h5)?
  – Sprawdź poprawność ścieżki i zgodność wersji TensorFlow.
• Skalowanie danych nie działa (.pkl)?
  – Upewnij się, że wskazujesz właściwy plik scaler.pkl.
• Detekcja działa wolno?
  – Zwiększ „Interwał (ms)” w zakładce Detekcja znaków.
  – W panelu MediaPipe obniż Model complexity i kliknij „Zastosuj”.
• Litery skaczą lub są losowe?
  – Podnieś wartość Próg (np. do 0.8–0.9).
  – Przeprowadź dodatkowy trening z lepszymi danymi.
• Brak folderu text_files?
  – Utwórz ręcznie folder text_files w katalogu projektu.
  – Wstaw do niego pliki .txt.
• „Load text” nic nie robi?
  – Sprawdź, czy plik .txt zawiera zwykły tekst (bez BOM).
• Aplikacja się zawiesza lub zamyka?
  – Sprawdź logi w other/logs.log.
  – Upewnij się, że masz wystarczająco pamięci RAM.
• Nie reaguje na klawisze?
  – Upewnij się, że okno aplikacji jest aktywne.
  – Skrót Tab = Flip, Spacja/Enter = Zapisz dane, q = Wyjdź.

------------------------------------------------
""",
    "language_label": "Language/Język:",
    "dlg_error": "Błąd",
"dlg_warning": "Ostrzeżenie",
"dlg_confirm": "Potwierdzenie",
"dlg_quit_app": "Czy chcesz zakończyć działanie aplikacji?",

"err_label_empty": "Pole etykiety jest puste. Podaj nazwę.",
"err_no_filename": "Nie podano nazwy pliku. Wybierz lub wpisz ją ręcznie.",
"err_file_not_exists": "Plik nie istnieje: {file}",
"err_text_file_not_loaded": "Plik tekstowy nie został wczytany.",
"err_incomplete_input": "Niekompletne dane wejściowe. Uzupełnij wymagane pola.",
"err_invalid_numbers": "Podano nieprawidłowe wartości liczbowe.",
"err_csv_path_missing": "Nie podano ścieżki do pliku CSV.",
"err_test_size_range": "Test size musi być liczbą z zakresu (0;1).",
"err_epochs_positive": "Liczba epok musi być większa od zera.",
"err_batch_positive": "Batch size musi być wartością dodatnią.",
"err_patience_nonnegative": "Patience nie może być ujemne.",

"warn_no_label": "Nie wprowadzono etykiety.",
"warn_no_camera": "Nie wybrano kamery.",
"warn_invalid_range": "Nieprawidłowy zakres wartości.",

"log_missing_dir": "Katalog nie istnieje lub brak uprawnień: {dir}",
"log_file_loaded": "Wczytano plik: {file}",
"log_csv_read_error": "Błąd w trakcie odczytu pliku CSV.",
"log_csv_missing_train": "Brak ścieżki lub nieprawidłowa ścieżka do pliku CSV: {path}",
"log_label_missing_csv": "W pliku CSV brakuje kolumny 'label'.",
"log_split_error": "Błąd podczas dzielenia danych: {err}",
"log_scaler_saved": "Zapisano scaler w {path}",
"log_model_summary": "Podsumowanie modelu:\n{summary}",
"log_model_saved": "Zapisano model w {path}",
"log_test_accuracy": "Dokładność testu: {acc}",
"log_confusion_matrix": "Macierz pomyłek:\n{cm}",
"log_training_finished": "Proces treningu zakończył się pomyślnie.",
"app_title": "Aplikacja do zbierania danych dłoni",
"err_no_camera": "Nie znaleziono kamery lub jest używana przez inną aplikację.",
"log_closing": "Zamykanie aplikacji...",
"log_first_set_label": "Ustaw etykietę przed zapisaniem danych.",
"log_no_camera_data": "Brak danych z kamery. Upewnij się, że kamera działa poprawnie.",
"log_no_hand": "Nie wykryto dłoni w kadrze.",
"log_images_empty": "Katalog images nie istnieje lub jest już pusty.",
"log_images_cleared": "Folder images został wyczyszczony.",
"log_action_cancelled": "Anulowano działanie przez użytkownika.",
"log_csv_missing": "Plik CSV nie istnieje: {path}",
"log_csv_reset": "Wyzerowano plik CSV: {path}",
"log_reset_defaults": "Przywrócono ustawienia domyślne.",
"log_mp_updated": "Zaktualizowano ustawienia MediaPipe.",
"lbl_current_label": "Aktualna etykieta: {val}",
"lbl_current_index": "Aktualny indeks: {val}",
"dlg_sure_clear_images": "Czy na pewno wyczyścić wszystkie obrazy?",
"dlg_sure_reset_csv": "Czy na pewno zresetować plik CSV: {file}?",
"log_saved_sample": "Zapisano próbkę dla etykiety: {label}, indeks: {idx} -> dodano do CSV: {path}",
"log_csv_total": "W pliku CSV jest teraz {total} wierszy danych (bez nagłówka).",
"log_saved_image": "Zapisano plik obrazu: {path}",
"log_folder_count": "Folder dla etykiety '{label}' zawiera teraz {count} obrazów.",
"log_camera_switch": "Przełączanie kamery z {old} na {new}",
"log_label_selected": "Wybrana etykieta: {val}",
"log_no_label": "Nie wprowadzono etykiety.",
"status_on": "WŁĄCZONY",
"status_off": "WYŁĄCZONY",
"log_flip_status": "Odbicie pionowe: {val}.",
"log_epoch_progress": "Epoka {curr}/{total} - Strata: {loss:.4f}, Dokł.: {acc:.4f}, Strata wal.: {vloss:.4f}, Dokł. wal.: {vacc:.4f}",
"frame_train_plots":    "Wykresy treningu",
"lbl_validation_split": "Podział na walidację:",
"lbl_monitor":          "Monitor EarlyStopping:",
"err_validation_split_range": "Podział na walidację musi być między 0 i 1.",
"msg_wait_camera": "Proszę czekać, inicjalizacja kamery…",
"main_window_title": "Rozpoznawanie gestów dłoni",
"wait_window_title": "Inicjalizacja kamery…",
"btn_flip_horizontal": "Przerzuć poziomo",
"btn_flip_vertical": "Przerzuć pionowo",
"log_flip_horizontal_on": "Przerzucanie poziomo: WŁĄCZONE",
"log_flip_horizontal_off": "Przerzucanie poziomo: WYŁĄCZONE",
"log_flip_vertical_on": "Przerzucanie pionowo: WŁĄCZONE",
"log_flip_vertical_off": "Przerzucanie pionowo: WYŁĄCZONE",
"btn_restart_camera": "Restartuj kamerę\njeżeli nie działa",
"log_camera_restarted": "Kamera została zrestartowana"
}