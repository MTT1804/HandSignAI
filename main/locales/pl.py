STRINGS: dict[str, str] = {
    # ---------- zakładki ----------
    "tab_collect":          "Zbieranie danych",
    "tab_train":            "Trening modelu",
  "tab_detection":        "Detekcja znaków",
  "tab_text":             "Miganie tekstu",
  "tab_instr":            "Instrukcja",
  "tab_settings":         "Ustawienia",

    "instructions_text": """\
  # HandSignAI — instrukcja

  HandSignAI to aplikacja do:
  - zbierania danych dłoni (obrazy + CSV z landmarkami),
  - trenowania modelu,
  - detekcji znaków,
  - ćwiczenia tekstu ("Miganie tekstu").

  ## Spis treści
  1. Szybki start (4 kroki)
  2. Struktura plików
  3. Zbieranie danych
  4. Trening modelu
  5. Detekcja znaków
  6. Miganie tekstu
  7. Ustawienia
  8. Skróty klawiszowe
  9. Szybka diagnoza

  > Uwaga: przy starcie aplikacja najpierw ładuje TensorFlow i MediaPipe (ekran ładowania).
  > Dopiero potem pojawia się cały interfejs.

  ---

  ## 1) Szybki start (polecany scenariusz)
  1) **Zbieranie danych**: zbierz próbki dla wszystkich znaków, które chcesz rozpoznawać.
  2) **Trening modelu**: uruchom trening na zebranym CSV.
  3) **Detekcja znaków**: włącz detekcję i sprawdź jakość.
  4) (Opcjonalnie) **Miganie tekstu**: ćwicz rozpoznawanie na tekście.

  ---

  ## 2) Struktura plików (domyślne)
  ```
  main/data/data.csv            # cechy + kolumna label
  main/images/<etykieta>/...    # zapisane obrazy
  main/models/model.h5          # wytrenowany model
  main/other/scaler.pkl         # skaler cech
  main/other/settings.json      # ustawienia aplikacji
  text_files/*.txt              # pliki do "Migania tekstu"
  ```

  ---

  ## 3) Zbieranie danych
  **Cel:** zebrać przykłady dla każdej klasy (np. litery/cyfry oraz "#").

  ### Krok po kroku
  1. Wpisz etykietę w polu (np. `A`, `B`, `1`, `#`).
  2. Ustaw etykietę (przycisk albo `Ctrl+Enter`).
  3. Zapisuj próbki (`Enter` / `Spacja` / `Ctrl+S`).

  ### Co się zapisuje?
  - wiersz w `data.csv` (landmarki + `label`),
  - obraz do `images/<etykieta>/...`.

  ### Kontrola jakości (ważne)
  - zapis jest blokowany, gdy **nie ma dłoni** albo **pewność jest zbyt niska**,
  - `Cofnij ostatni zapis` usuwa **ostatni wiersz z CSV** i odpowiadający obraz.

  ### Samowyzwalacz
  - ustaw liczbę sekund, włącz `Samowyzwalacz`, kliknij `Start`,
  - `Pętla` powtarza odliczanie i zapis automatycznie.

  ### Przetwarzanie wsadowe (folder zdjęć)
  - wybierz folder ze zdjęciami (również podfoldery), ustaw etykietę,
  - `Przetwórz folder` zapisze cechy do CSV i kopie obrazów do `images/<etykieta>/`,
  - opcja zapisu overlay narysuje landmarki/etykietę na obrazach wyjściowych.

  ### Overlay + Flip
  - overlay pokazuje landmarki i opis (etykieta + indeks),
  - `Flip H` / `Flip V` odwraca obraz z kamery.

  ---

  ## 4) Trening modelu
  - `Rozpocznij trening` uruchamia trening w tle.
  - `Pokaż zaawansowane` odsłania dodatkowe parametry (np. epochs, batch, split).
  - po treningu pojawia się wykres `Loss/Accuracy`.
  - `Kopiuj wykres do schowka` kopiuje wykres (Windows).

  **Wskazówka:** jeśli masz mało danych, zacznij od mniejszej liczby epok (np. 10).

  ---

  ## 5) Detekcja znaków
  - `Start/Stop` steruje rozpoznawaniem.
  - `Interwał (ms)` ogranicza częstotliwość predykcji (większy = mniej obciążenia).
  - `Próg` filtruje wyniki (większy = mniej błędów, ale częściej pojawi się „—”).
  - tryb `Wstawiaj znak tylko po Enterze` pozwala zatwierdzać znak ręcznie.
  - panel `Top 10` pokazuje najbardziej prawdopodobne klasy.

  ---

  ## 6) Miganie tekstu
  1. Wybierz plik `.txt` z folderu `text_files`.
  2. Kliknij `Wczytaj tekst`, potem `Start`.
  3. Aplikacja przechodzi po znakach i prowadzi statystyki trafień/błędów.

  ---

  ## 7) Ustawienia
  - motyw `Jasny/Ciemny` + osobny akcent dla Light/Dark,
  - szybkie otwieranie: `images` / `models` / `CSV`,
  - ścieżki do CSV/modelu/scalera (jeśli używasz innych plików),
  - MediaPipe: m.in. `max_num_hands`, `model_complexity`, progi confidence.

  ---

  ## 8) Skróty klawiszowe
  > Skróty akcji działają zależnie od aktywnego widoku.
  > Jeśli kursor jest w polu tekstowym (np. wpisujesz etykietę), aplikacja nie „kradnie” spacji/entera.

  ### Nawigacja
  | Skrót | Akcja |
  |---|---|
  | F1 | Instrukcja |
  | F2 | Zbieranie danych |
  | F3 | Detekcja znaków |
  | F4 | Trening modelu |
  | F5 | Miganie tekstu |
  | F6 | Ustawienia |
  | Ctrl+1..Ctrl+6 | Skróty jak w menu |
  | Ctrl+Tab | Następny widok |
  | Ctrl+Shift+Tab | Poprzedni widok |

  ### Globalnie
  | Skrót | Akcja |
  |---|---|
  | Ctrl+L | Pokaż/ukryj konsolę |
  | Ctrl+Shift+T | Przełącz Jasny/Ciemny |
  | Ctrl+Q | Wyjście |

  ### Zbieranie danych
  | Skrót | Akcja |
  |---|---|
  | Enter / Spacja / Ctrl+S | Zapis próbki |
  | Ctrl+Z | Cofnij ostatni zapis |
  | Ctrl+Enter | Ustaw etykietę z pola |
  | Alt+T | Start/Stop samowyzwalacza |
  | Alt+Y | Włącz/wyłącz samowyzwalacz |
  | Alt+L | Pętla samowyzwalacza |
  | Alt+O | Overlay on/off |
  | Ctrl+B | Wybierz folder wsadu |
  | Alt+B | Start wsadu |
  | Alt+R | Restart kamery |
  | Alt+H / Alt+V | Flip H / Flip V |

  ### Detekcja znaków
  | Skrót | Akcja |
  |---|---|
  | Alt+S | Start |
  | Alt+X | Stop |
  | Alt+C | Wyczyść bufor |
  | Alt+R | Restart kamery |
  | Alt+H / Alt+V | Flip H / Flip V |
  | Alt+E | Tryb „tylko po Enterze” |

  ### Trening
  | Skrót | Akcja |
  |---|---|
  | Alt+R | Start treningu |
  | Alt+A | Pokaż/ukryj zaawansowane |
  | Alt+C | Kopiuj wykres do schowka |

  ### Miganie tekstu
  | Skrót | Akcja |
  |---|---|
  | Alt+L | Wczytaj tekst |
  | Alt+S | Start |
  | Alt+X | Stop |
  | Alt+R | Restart kamery |
  | Alt+H / Alt+V | Flip H / Flip V |
  | Alt+O | Overlay on/off |

  ---

  ## 9) Szybka diagnoza
  - **Brak obrazu:** zamknij inne aplikacje korzystające z kamery (Teams/Zoom), użyj `Restart`.
  - **Losowe litery:** podnieś `Próg` (np. 0.8–0.9), dozbieraj dane, wytrenuj ponownie.
  - **Działa wolno:** zwiększ `Interwał (ms)`, obniż `model_complexity`, zmniejsz liczbę rąk.
  - **Brak detekcji:** najpierw zbierz dane → wytrenuj model → dopiero potem detekcja.
  """,
    "startup_loading_tf": "Ładowanie TensorFlow…",
    "startup_loading_mp": "Ładowanie MediaPipe…",
    "startup_loading_training": "Ładowanie modułu treningu…",
    "startup_loading_failed": "Nie udało się załadować zależności: {err}",
    "log_language_changed": "Zmieniono język na: {lang}",
    "log_paths_applied":    "Zastosowano ścieżki.",
    "log_detection_applied": "Zastosowano ustawienia detekcji.",
    "log_detection_started": "Detekcja uruchomiona.",
    "log_detection_stopped": "Detekcja zatrzymana.",
    "log_mediapipe_applied": "Zastosowano ustawienia MediaPipe.",
    "log_overlays":         "Nakładki: {state}",
    "log_delete_all_done":  "Usunięto wszystko (CSV/Model/Scaler/Images).",
    "log_delete_all_cancelled": "Anulowano usuwanie wszystkiego.",
    "log_advanced_shown":   "Opcje zaawansowane treningu: pokazane",
    "log_advanced_hidden":  "Opcje zaawansowane treningu: ukryte",
    "log_training_started": "Rozpoczęto trening (epochs={epochs}, batch={batch}, split={split}).",
    "log_text_practice_started": "Miganie tekstu: start.",
    "log_text_practice_stopped": "Miganie tekstu: stop.",
    "log_theme":            "Motyw: {theme}",

    # ---------- destrukcyjne akcje ----------
    "warn_delete_all_data":
      "UWAGA: To bezpowrotnie usunie poniższe elementy:\n\n"
      "- CSV: {csv}\n"
      "- Model: {model}\n"
      "- Scaler: {scaler}\n"
      "- Katalog images: {images}\n\n"
      "Nie da się tego cofnąć.",
    "confirm_delete_all_data":
      "Czy na pewno usunąć WSZYSTKO z ostrzeżenia?",
    "btn_clear_images":     "Wyczyść folder images",
    "btn_reset_csv":        "Wyzeruj plik CSV",
    "btn_reset_defaults":   "Przywróć domyślne",
    "btn_quit":             "Wyjdź (q)",
    "lbl_G":                "G:",
    "frame_file_labels":    "Nazwa plików (domyślne wartości)",
    "lbl_scaler_file":      "Scaler (wyj.):",
    "lbl_test_size":        "Podział testowy (np. 0.2):",
    "lbl_batch_size":       "Rozmiar batcha:",
    # ---------- zakładka „Detekcja znaków” ----------

    # ---------- okno z prawdopodobieństwami ----------
    # ---------- zakładka „Miganie tekstu” ----------
    "lbl_select_text_file": "Wybierz plik z tekstem:",
    "stat_failed":          "Błędy (nieudane próby): {fail}",

    "language_label": "Język:",
  "cam_placeholder_title": "Kamera nieaktywna",
"camera_preset_medium": "Średnia",
"camera_preset_large": "Duża",
"overlay_label": "Etykieta: {label}",
"overlay_label_idx": "Etykieta: {label} | Indeks: {idx}",
"log_view_changed": "Zmieniono widok: {view}",
"log_view_on_show_error": "Błąd on_show widoku: {err}",
"log_camera_open_failed": "Nie można otworzyć kamery o indeksie {idx}.",
"log_camera_opened": "Otwarto kamerę (index={idx}).",
"log_camera_consumer_error": "Błąd przetwarzania klatki: {err}",
"log_detection_error": "Błąd detekcji: {err}",
"log_plot_error": "Błąd wykresu: {err}",
"log_history_keys": "Klucze historii: {keys}",
"log_camera_size_preset": "Rozmiar podglądu kamery: {preset}",
"log_training_exception": "Trening nie powiódł się: {err}",
"log_settings_loaded": "Wczytano ustawienia z: {path}",
"log_settings_load_failed": "Nie udało się wczytać ustawień: {err}",
"log_settings_saved": "Zapisano ustawienia do: {path}",
"log_settings_save_failed": "Nie udało się zapisać ustawień: {err}",
"lbl_self_timer": "Samowyzwalacz",
"lbl_self_timer_seconds": "Sekundy:",
"chk_self_timer_loop": "Pętla",
"lbl_self_timer_countdown": "Zapis za {seconds}s…",
"btn_self_timer_start": "Start",
"btn_self_timer_cancel": "Anuluj",
"err_timer_seconds": "Sekundy muszą być dodatnią liczbą całkowitą.",
"log_self_timer_started": "Samowyzwalacz uruchomiony ({seconds}s).",
"log_self_timer_fired": "Samowyzwalacz: zapis próbki.",
"btn_restart_camera": "Restartuj/Włącz kamerę",
"log_camera_restarted": "Kamera została zrestartowana"
  ,"log_camera_restarting": "Restartowanie kamery…"

  ,"plot_loss": "Strata"
  ,"plot_accuracy": "Dokładność"
  ,"plot_epoch": "Epoka"
  ,"plot_train": "Trening"
  ,"plot_val": "Walidacja",

  # ---------- CTk: komplet tłumaczeń (parytet z en.py) ----------
  "app_title_ctk": "HandSignAI",
  "theme_switch": "Jasny/Ciemny",
  "console_title": "Logi / Konsola",
  "btn_hide_console": "Ukryj konsolę",
  "btn_show_console": "Pokaż konsolę",
  "section_stats": "Statystyki",
  "stat_samples": "Zapisane próbki:",
  "stat_classes": "Liczba klas:",
  "section_author": "O autorze",
  "author_name": "Mateusz Tyl",
  "author_info": "Stworzone w ramach pracy inżynierskiej na Politechnice Warszawskiej. Aplikacja została rozwinięta w latach 2025–2026.",
  "stat_images": "Zapisane obrazy:",
  "stat_csv_size": "Rozmiar CSV:",
  "stat_csv_mtime": "Ostatnia modyfikacja CSV:",
  "stat_model_size": "Rozmiar pliku modelu:",
  "cam_preview_placeholder": "Podgląd kamery\n(zastępczy)",
  "ctk_pred_title": "Rozpoznany znak / słowo",
  "lbl_top10": "Top 10",
  "lbl_hands_detected": "Wykryte dłonie: {n}",
  "lbl_last_classification_time": "Czas klasyfikacji ostatniego znaku: {ms} ms",
  "lbl_last_classification_times": "Czas klasyfikacji ostatniego znaku: MP {mp_ms} ms | Model {model_ms} ms | Razem {total_ms} ms",

  # ---------- wymagania detekcji ----------
  "det_missing_files_title": "Detekcja niedostępna",
  "det_missing_files_body":
    "Brakujące/nieprawidłowe pliki wymagane do detekcji:\n"
    "{files}\n\n"
    "Najpierw przejdź do \"{step1}\" aby zebrać dane, a potem do \"{step2}\" aby wytrenować model.",
  "det_missing_csv": "CSV: {path}",
  "det_invalid_csv": "CSV nieprawidłowy (brak nagłówka/label): {path}",
  "det_missing_model": "Model: {path}",
  "det_missing_scaler": "Scaler: {path}",

  # ---------- wspólne przyciski ----------
  "btn_clear": "Wyczyść",
  "btn_restart": "Restart",
  "btn_flip_h": "Flip H",
  "btn_flip_v": "Flip V",
  "btn_save_sample": "Zapisz próbkę (Enter/Spacja)",
  "btn_undo_last": "Cofnij ostatni zapis",

  # ---------- wsad ----------
  "section_batch": "Przetwarzanie wsadowe",
  "placeholder_batch_folder": "Folder ze zdjęciami…",
  "btn_browse": "Przeglądaj…",
  "btn_batch_process": "Przetwórz folder",
  "chk_batch_save_overlay": "Rysuj landmarki i etykiety na zapisanych obrazach",
  "chk_collect_augmentation": "Augmentacja (zapisz kilka wariantów)",
  "chk_collect_mirror": "Odbicie lustrzane (zapisz odbite próbki)",
  "err_batch_folder": "Wybierz poprawny folder ze zdjęciami.",
  "err_deps_not_ready": "Aplikacja nadal się ładuje. Poczekaj.",
  "err_mediapipe_not_ready": "MediaPipe nie jest jeszcze gotowy. Poczekaj.",
  "log_batch_started": "Wsadowo: start. Folder={folder}, etykieta={label}",
  "log_batch_no_images": "Wsadowo: brak obrazów w folderze: {folder}",
  "log_batch_progress": "Wsadowo: {done}/{total} (zapisane={saved}, pominięte={skipped})",
  "log_batch_done": "Wsadowo: zakończono. Razem={total}, zapisane={saved}, pominięte={skipped}",
  "log_batch_failed": "Wsadowo: błąd: {err}",

  "placeholder_label": "Litera / cyfra",
  "btn_apply_paths": "Zastosuj ścieżki",
  "btn_apply_detection": "Zastosuj detekcję",
  "btn_delete_all_data": "Usuń WSZYSTKIE dane (CSV/Model/Scaler/Images)",
  "chk_overlay": "Nakładka landmarków",
  "chk_show_advanced": "Pokaż zaawansowane",

  # ---------- sekcje ustawień ----------
  "section_general": "Ogólne",
  "section_paths": "Ścieżki",
  "section_quick_open": "Szybkie otwieranie",
  "btn_open_images_folder": "Otwórz folder images",
  "btn_open_models_folder": "Otwórz folder models",
  "btn_open_csv_file": "Otwórz plik CSV",
  "err_open_path": "Nie można otworzyć ścieżki: {err}",
  "section_camera": "Kamera",
  "lbl_camera": "Kamera:",
  "section_detection": "Detekcja",
  "section_mediapipe": "MediaPipe",
  "section_misc": "Różne",
  "section_theme": "Motyw",
  "section_danger_zone": "Strefa ryzyka",
  "danger_zone_desc": "Usunięcie danych kasuje CSV, model, scaler i obrazy. Tej operacji nie można cofnąć.",
  "btn_save_settings": "Zapisz/Zastosuj zmiany",
  "log_settings_applied": "Zastosowano ustawienia.",
  "lbl_theme_light": "Motyw (Jasny):",
  "lbl_theme_dark": "Motyw (Ciemny):",
  "theme_blue": "Niebieski",
  "theme_green": "Zielony",
  "theme_dark_blue": "Ciemnoniebieski",
  "theme_purple": "Fioletowy",
  "theme_red": "Czerwony",
  "theme_orange": "Pomarańczowy",
  "theme_teal": "Turkusowy",
  "theme_pink": "Różowy",
  "theme_yellow": "Żółty",
  "theme_lime": "Limonkowy",
  "theme_cyan": "Cyjan",
  "theme_indigo": "Indygo",
  "theme_gray": "Szary",
  "theme_amber": "Bursztynowy",
  "btn_apply_theme": "Zastosuj motyw",
  "lbl_camera_size": "Preset rozmiaru kamery:",

  # ---------- trening / wykres ----------
  "lbl_val_split": "Podział walidacyjny:",
  "lbl_monitor": "Monitor:",
  "training_chart_placeholder": "Wykres treningu (Loss/Accuracy)",
  "btn_copy_chart": "Kopiuj wykres do schowka",
  "log_chart_copied": "Skopiowano wykres do schowka.",
  "err_chart_copy": "Nie można skopiować wykresu: {err}",

  "training_cm_title": "Macierz pomyłek",
  "training_cm_placeholder": "Macierz pomyłek pojawi się po zakończeniu treningu.",
  "training_cm_true": "Prawdziwa etykieta",
  "training_cm_pred": "Przewidziana etykieta",
  "btn_copy_cm": "Kopiuj macierz pomyłek",
  "log_cm_copied": "Skopiowano macierz pomyłek do schowka.",
  "err_cm_copy": "Nie można skopiować macierzy pomyłek: {err}",

  # ---------- logi / start ----------
  "log_gui_started": "GUI uruchomione.",
  "startup_loading": "Ładowanie TensorFlow i MediaPipe…",
  "err_delete_all_failed": "Usuwanie nie powiodło się:\n{err}",

  # ---------- przyciski / etykiety widoków ----------
  "btn_start": "Start",
  "btn_stop": "Stop",
  "btn_clear_screen": "Wyczyść ekran",
  "lbl_interval": "Interwał (ms):",
  "lbl_threshold": "Próg:",
  "err_bad_interval_threshold": "Nieprawidłowy interwał lub próg.",
  "warn_interval_threshold_range": "Poprawny zakres: interwał > 0, próg w (0, 1].",
  "lbl_choose_camera": "Wybierz kamerę:",
  "lbl_enter_label": "Wpisz literę/liczbę do zbierania:",
  "btn_set_label": "Ustaw etykietę",
  "btn_save_data": "Zapisz dane [Enter]",
  "btn_flip": "Odbicie pionowe (Tab)",
  "section_data_mgmt": "--- Zarządzanie danymi ---",
  "section_reset": "--- Reset ustawień ---",
  "section_img_settings": "--- Ustawienia obrazu ---",
  "lbl_brightness": "Jasność (beta):",
  "lbl_contrast": "Kontrast (alpha%):",
  "lbl_gamma": "Gamma (1.0 = brak):",
  "lbl_color_shift": "Przesunięcie kolorów (R, G, B):",
  "lbl_R": "R:",
  "lbl_B": "B:",
  "chk_static_img_mode": "static_image_mode (True = zdjęcia statyczne)",
  "lbl_max_num_hands": "max_num_hands:",
  "lbl_model_complexity": "model_complexity (0-2):",
  "lbl_min_det_conf": "min_detection_confidence (%):",
  "lbl_min_track_conf": "min_tracking_confidence (%):",
  "btn_apply_mp": "Zastosuj ustawienia MediaPipe",
  "chk_show_overlays": "Pokaż dodatkowe elementy (tekst, kropki)",
  "lbl_csv_file": "CSV (wej./wyj.):",
  "lbl_model_file": "Model (wyj.):",
  "frame_train_config": "Konfiguracja treningu",
  "lbl_random_state": "Random state:",
  "lbl_epochs": "Epoki:",
  "lbl_patience": "Patience (EarlyStopping):",
  "btn_start_training": "Rozpocznij trening",
  "chk_enter_mode": "Wstawiaj znak tylko po Enterze",
  "btn_start_detection": "Start detekcji",
  "btn_stop_detection": "Stop detekcji",
  "win_top_probs": "Najwyższe prawdopodobieństwa",
  "lbl_cam_preview_text": "Podgląd kamery (tekst)",
  "btn_load_text": "Wczytaj tekst",
  "stat_correct": "Poprawnie rozpoznane: {ok} / {total}",
  "stat_remaining": "Pozostało znaków: {remain}",

  # ---------- dialogi ----------
  "dlg_confirm": "Potwierdzenie",
  "dlg_sure_clear_images": "Czy na pewno wyczyścić wszystkie obrazy?",
  "dlg_sure_reset_csv": "Czy na pewno zresetować plik CSV: {file}?",
  "win_top10": "Top 10 prawdopodobieństw",

  # ---------- logi (zgodność z CTk) ----------
  "log_no_csv_path": "Brak ścieżki do CSV — nie można wykryć klas.",
  "log_no_label_column": "W CSV nie znaleziono kolumny 'label' — nie można wykryć klas.",
  "log_classes_found": "Wykryte klasy: {classes}",
  "log_camera_switch": "Przełączanie kamery z {old} na {new}",
  "log_saved_sample": "Zapisano dane dla etykiety: {label}, indeks: {idx} -> dopisano do CSV: {path}",
  "log_saved_samples_batch": "Zapisano {label}: {count} próbek do CSV: {path}.",
  "log_saved_image": "Zapisano plik obrazu: {path}",
  "log_save_image_failed": "Nie udało się zapisać obrazu: {path}.",
  "log_hand_conf_too_low": "Zbyt niska pewność dłoni ({conf}) — pomijam zapis.",
  "log_undo_none": "Brak czego cofnąć.",
  "log_undo_done": "Cofnięto ostatni zapis (CSV + obraz).",
  "log_undo_failed": "Nie udało się cofnąć ostatniego zapisu: {err}",

  "theme_label": "Motyw:",
  "cam_placeholder_subtitle": "Wybierz kamerę i kliknij Restart",
  "log_header": "Dziennik aktywności",
  "btn_hide_log": "Ukryj log",
  "btn_show_log": "Pokaż log",
  "dlg_error": "Błąd",
  "dlg_warning": "Ostrzeżenie",
  "dlg_quit_app": "Czy chcesz zakończyć działanie aplikacji?",
  "err_label_empty": "Pole etykiety jest puste. Podaj wartość.",
  "err_no_filename": "Nie podano nazwy pliku. Wybierz lub wpisz ją ręcznie.",
  "err_file_not_exists": "Plik nie istnieje: {file}",
  "err_text_file_not_loaded": "Plik tekstowy nie został wczytany.",
  "err_incomplete_input": "Niekompletne dane. Uzupełnij wymagane pola.",
  "err_invalid_numbers": "Podano nieprawidłowe wartości liczbowe.",
  "err_csv_path_missing": "Brak ścieżki do CSV. Podaj poprawną ścieżkę.",
  "err_test_size_range": "Test size musi być liczbą z zakresu (0; 1).",
  "err_epochs_positive": "Liczba epok musi być dodatnią liczbą całkowitą.",
  "err_batch_positive": "Batch size musi być wartością dodatnią.",
  "err_patience_nonnegative": "Patience musi być równe 0 lub dodatnie.",
  "warn_no_label": "Nie wprowadzono etykiety.",
  "warn_no_camera": "Nie wybrano kamery.",
  "warn_invalid_range": "Nieprawidłowa lub poza zakresem wartość.",
  "log_missing_dir": "Katalog nie istnieje lub brak uprawnień: {dir}",
  "log_file_loaded": "Wczytano plik: {file}",
  "log_csv_read_error": "Błąd odczytu pliku CSV.",
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
  "log_csv_total": "W pliku CSV jest teraz {total} wierszy danych (bez nagłówka).",
  "log_folder_count": "Folder dla etykiety '{label}' zawiera teraz {count} obrazów.",
  "log_label_selected": "Wybrana etykieta: {val}",
  "log_no_label": "Nie wprowadzono etykiety.",
  "status_on": "WŁĄCZONE",
  "status_off": "WYŁĄCZONE",
  "log_flip_status": "Odbicie pionowe: {val}.",
  "log_epoch_progress": "Epoka {curr}/{total} - Strata: {loss:.4f}, Dokł.: {acc:.4f}, Strata wal.: {vloss:.4f}, Dokł. wal.: {vacc:.4f}",
  "frame_train_plots": "Wykresy treningu",
  "lbl_validation_split": "Podział na walidację:",
  "lbl_monitor:": "Monitor EarlyStopping:",
  "err_validation_split_range": "Podział na walidację musi być między 0 i 1.",
  "msg_wait_camera": "Proszę czekać, inicjalizacja kamery…",
  "main_window_title": "Rozpoznawanie gestów dłoni na żywo",
  "wait_window_title": "Inicjalizacja kamery…",
  "btn_flip_horizontal": "Przerzuć poziomo",
  "btn_flip_vertical": "Przerzuć pionowo",
  "log_flip_horizontal_on": "Przerzucanie poziomo: WŁĄCZONE",
  "log_flip_horizontal_off": "Przerzucanie poziomo: WYŁĄCZONE",
  "log_flip_vertical_on": "Przerzucanie pionowo: WŁĄCZONE",
  "log_flip_vertical_off": "Przerzucanie pionowo: WYŁĄCZONE",
  "camera_preset_small": "Mała"
}
