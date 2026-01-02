STRINGS: dict[str, str] = {
    # ---------- notebook tabs ----------
    "tab_collect":          "Data collection",
    "tab_train":            "Model training",
    "tab_detection":        "Sign detection",
    "tab_text":             "Text signing",
    "tab_instr":            "Instructions",
    "tab_settings":         "Settings",

    # ---------- CTk app ----------
    "app_title_ctk":        "HandSignAI",
    "theme_switch":         "Light/Dark",
    "console_title":        "Logs / Console",
    "btn_hide_console":     "Hide console",
    "btn_show_console":     "Show console",
    "section_stats":        "Stats",
    "stat_samples":         "Saved samples:",
    "stat_classes":         "Number of classes:",
      "section_author":       "About the author",
      "author_name":         "Mateusz Tyl",
      "author_info":         "Created as part of an engineering diploma thesis at Warsaw University of Technology. The application was developed in 2025–2026.",
    "stat_images":          "Saved images:",
    "stat_csv_size":        "CSV size:",
    "stat_csv_mtime":       "CSV last modified:",
    "stat_model_size":      "Model file size:",
    "cam_preview_placeholder": "Camera preview\n(placeholder)",
    "ctk_pred_title":       "Recognised sign / word",
    "lbl_top10":            "Top 10",
    "lbl_hands_detected":   "Hands detected: {n}",
    "lbl_last_classification_time": "Last classification time: {ms} ms",
    "lbl_last_classification_times": "Last classification time: MP {mp_ms} ms | Model {model_ms} ms | Total {total_ms} ms",

    # ---------- detection prerequisites ----------
    "det_missing_files_title": "Detection unavailable",
    "det_missing_files_body":
      "Missing/invalid files required for detection:\n"
      "{files}\n\n"
      "First go to \"{step1}\" to collect data, then \"{step2}\" to train the model.",
    "det_missing_csv":     "CSV: {path}",
    "det_invalid_csv":     "CSV invalid (missing header/label): {path}",
    "det_missing_model":   "Model: {path}",
    "det_missing_scaler":  "Scaler: {path}",

    # ---------- common CTk buttons ----------
    "btn_clear":            "Clear",
    "btn_restart":          "Restart",
    "btn_restart_camera":   "Restart/Enable camera",
    "btn_flip_h":           "Flip H",
    "btn_flip_v":           "Flip V",
    "btn_save_sample":      "Save sample (Enter/Space)",
    "btn_undo_last":        "Undo last save",
    "section_batch":        "Batch processing",
    "placeholder_batch_folder": "Folder with images…",
    "btn_browse":           "Browse…",
    "btn_batch_process":    "Process folder",
    "chk_batch_save_overlay": "Draw landmarks and labels on output images",
    "chk_collect_augmentation": "Data augmentation (save multiple variants)",
    "chk_collect_mirror": "Mirror (save flipped samples)",
    "err_batch_folder":     "Choose a valid folder with images.",
    "err_deps_not_ready":   "The app is still loading. Please wait.",
    "err_mediapipe_not_ready": "MediaPipe is not ready yet. Please wait.",
    "log_batch_started":    "Batch: started. Folder={folder}, label={label}",
    "log_batch_no_images":  "Batch: no images in folder: {folder}",
    "log_batch_progress":   "Batch: {done}/{total} (saved={saved}, skipped={skipped})",
    "log_batch_done":       "Batch: done. Total={total}, saved={saved}, skipped={skipped}",
    "log_batch_failed":     "Batch: error: {err}",
    "placeholder_label":    "Letter / digit",
    "btn_apply_paths":      "Apply paths",
    "btn_apply_detection":  "Apply detection",
    "btn_delete_all_data":  "Delete ALL data (CSV/Model/Scaler/Images)",
    "chk_overlay":          "Landmarks overlay",
    "chk_show_advanced":    "Show advanced",

    # ---------- settings sections ----------
    "section_general":      "General",
    "section_paths":        "Paths",
    "section_quick_open":   "Quick open",
    "btn_open_images_folder": "Open images folder",
    "btn_open_models_folder": "Open models folder",
    "btn_open_csv_file":    "Open CSV file",
    "err_open_path":        "Cannot open path: {err}",
    "section_camera":       "Camera",
    "lbl_camera":           "Camera:",
    "section_detection":    "Detection",
    "section_mediapipe":    "MediaPipe",
    "section_misc":         "Misc",
    "section_theme":        "Theme",
    "section_danger_zone":  "Danger zone",
    "danger_zone_desc":     "Deleting data removes the CSV, model, scaler and images. This cannot be undone.",
    "btn_save_settings":    "Save changes",
    "log_settings_applied": "Settings applied.",
    "lbl_theme_light":      "Theme (Light):",
    "lbl_theme_dark":       "Theme (Dark):",
    "theme_blue":           "Blue",
    "theme_green":          "Green",
    "theme_dark_blue":      "Dark blue",
    "theme_purple":         "Purple",
    "theme_red":            "Red",
    "theme_orange":         "Orange",
    "theme_teal":           "Teal",
    "theme_pink":           "Pink",
    "theme_yellow":         "Yellow",
    "theme_lime":           "Lime",
    "theme_cyan":           "Cyan",
    "theme_indigo":         "Indigo",
    "theme_gray":           "Gray",
    "theme_amber":          "Amber",
    "btn_apply_theme":      "Apply theme",

    "lbl_camera_size":      "Camera size preset:",

    # ---------- training advanced labels ----------
    "lbl_val_split":        "Validation split:",
    "lbl_monitor":          "Monitor:",
    "training_chart_placeholder": "Training chart (Loss/Accuracy)",
    "btn_copy_chart":       "Copy chart to clipboard",
    "log_chart_copied":     "Chart copied to clipboard.",
    "err_chart_copy":       "Cannot copy chart: {err}",

    "training_cm_title":    "Confusion matrix",
    "training_cm_placeholder": "Confusion matrix will appear after training.",
    "training_cm_true":     "True label",
    "training_cm_pred":     "Predicted label",
    "btn_copy_cm":          "Copy confusion matrix",
    "log_cm_copied":        "Confusion matrix copied to clipboard.",
    "err_cm_copy":          "Cannot copy confusion matrix: {err}",

    # ---------- logs ----------
    "log_gui_started":      "GUI started.",
    "startup_loading": "Loading TensorFlow and MediaPipe…",
    "startup_loading_tf": "Loading TensorFlow…",
    "startup_loading_mp": "Loading MediaPipe…",
    "startup_loading_training": "Loading training module…",
    "startup_loading_failed": "Failed to load dependencies: {err}",
    "log_language_changed": "Language changed to: {lang}",
    "log_paths_applied":    "Paths applied.",
    "log_detection_applied": "Detection settings applied.",
    "log_detection_started": "Detection started.",
    "log_detection_stopped": "Detection stopped.",
    "log_mediapipe_applied": "MediaPipe settings applied.",
    "log_overlays":         "Overlays: {state}",
    "log_delete_all_done":  "All data deleted (CSV/Model/Scaler/Images).",
    "log_delete_all_cancelled": "Delete-all cancelled.",
    "log_advanced_shown":   "Advanced training options: shown",
    "log_advanced_hidden":  "Advanced training options: hidden",
    "log_training_started": "Training started (epochs={epochs}, batch={batch}, split={split}).",
    "log_text_practice_started": "Text practice started.",
    "log_text_practice_stopped": "Text practice stopped.",
    "log_theme":            "Theme: {theme}",

    # ---------- destructive actions ----------
    "warn_delete_all_data":
      "WARNING: This will permanently delete the following:\n\n"
      "- CSV: {csv}\n"
      "- Model: {model}\n"
      "- Scaler: {scaler}\n"
      "- Images directory: {images}\n\n"
      "This cannot be undone.",
    "confirm_delete_all_data":
      "Do you really want to delete EVERYTHING listed in the warning?",
    "err_delete_all_failed": "Delete failed:\n{err}",

    # ---------- common buttons / labels ----------
    "btn_start":            "Start",
    "btn_stop":             "Stop",
    "btn_clear_screen":     "Clear screen",
    "lbl_interval":         "Interval (ms):",
    "lbl_threshold":        "Threshold:",
    "err_bad_interval_threshold": "Invalid interval or threshold.",
    "warn_interval_threshold_range": "Valid range: interval > 0, threshold in (0, 1].",

    # ---------- collect tab ----------
    "lbl_choose_camera":    "Select camera:",
    "lbl_enter_label":      "Enter letter/number to collect:",
    "btn_set_label":        "Set label",
    "btn_save_data":        "Save data [Enter]",
    "btn_flip":             "Vertical flip (Tab)",
    "section_data_mgmt":    "--- Data management ---",
    "btn_clear_images":     "Clear images folder",
    "btn_reset_csv":        "Reset CSV file",
    "section_reset":        "--- Reset settings ---",
    "btn_reset_defaults":   "Restore defaults",
    "btn_quit":             "Quit (q)",

    "section_img_settings": "--- Image settings ---",
    "lbl_brightness":       "Brightness (beta):",
    "lbl_contrast":         "Contrast (alpha%):",
    "lbl_gamma":            "Gamma (1.0 = none):",
    "lbl_color_shift":      "Color shift (R, G, B):",
    "lbl_R":                "R:",
    "lbl_G":                "G:",
    "lbl_B":                "B:",

    # ---------- MediaPipe sub-frame ----------
    "section_mediapipe":    "--- MediaPipe settings ---",
    "chk_static_img_mode":  "static_image_mode (True = still images)",
    "lbl_max_num_hands":    "max_num_hands:",
    "lbl_model_complexity": "model_complexity (0-2):",
    "lbl_min_det_conf":     "min_detection_confidence (%):",
    "lbl_min_track_conf":   "min_tracking_confidence (%):",
    "btn_apply_mp":         "Apply MediaPipe changes",

    "chk_show_overlays":    "Show additional elements (text, dots)",

    # ---------- file names sub-frame ----------
    "frame_file_labels":    "Default file names",
    "lbl_csv_file":         "CSV (in/out):",
    "lbl_model_file":       "Model (out):",
    "lbl_scaler_file":      "Scaler (out):",

    # ---------- train tab ----------
    "frame_train_config":   "Training configuration",
    "lbl_test_size":        "Test size (e.g. 0.2):",
    "lbl_random_state":     "Random state:",
    "lbl_epochs":           "Epochs:",
    "lbl_batch_size":       "Batch size:",
    "lbl_patience":         "Patience (EarlyStopping):",
    "btn_start_training":   "Start training",

    # ---------- detection tab ----------
    "chk_enter_mode":       "Insert character only after Enter",
    "btn_start_detection":  "Start detection",
    "btn_stop_detection":   "Stop detection",

    # window title
    "win_top_probs":        "Top probabilities",

    # ---------- text detection tab ----------
    "lbl_cam_preview_text": "Camera preview (text)",
    "lbl_select_text_file": "Choose text file:",
    "btn_load_text":        "Load text",

    # ---------- stats (format placeholders) ----------
    "stat_correct":         "Correctly recognised: {ok} / {total}",
    "stat_failed":          "Errors (failed attempts): {fail}",
    "stat_remaining":       "Characters remaining: {remain}",

    # ---------- dialogs / confirmations ----------
    "dlg_confirm":          "Confirmation",
    "dlg_sure_clear_images":
        "Are you sure you want to delete the whole 'images' directory with sub-folders?",
    "dlg_sure_reset_csv":
        "Are you sure you want to reset file {file}?",

    # ---------- top-10 prediction window ----------
    "win_top10":            "Top 10 probabilities",

    # ---------- logging / misc (only most frequent) ----------
    "log_no_csv_path":      "CSV path missing – cannot detect classes.",
    "log_no_label_column":  "Column 'label' not found in CSV – cannot detect classes.",
    "log_classes_found":    "Detected classes: {classes}",
    "log_camera_switch":    "Switching camera from {old} to {new}...",
    "log_saved_sample":     "Saved {label} with index {idx} to CSV: {path}.",
    "log_saved_samples_batch": "Saved {label}: {count} samples to CSV: {path}.",
    "log_saved_image":      "Saved image to {path}.",
    "log_save_image_failed": "Failed to save image: {path}.",
    "log_hand_conf_too_low": "Hand confidence too low ({conf}) — skipping save.",
    "log_undo_none":        "Nothing to undo.",
    "log_undo_done":        "Undid the last save (CSV + image).",
    "log_undo_failed":      "Failed to undo last save: {err}",
        "instructions_text": """\
      # HandSignAI — Instructions

      HandSignAI helps you:
      - collect hand-sign data (images + CSV with landmarks),
      - train a model,
      - recognise signs,
      - practise with text files.

      ## Table of contents
      1. Quick start (4 steps)
      2. Default files / folders
      3. Data collection
      4. Model training
      5. Sign detection
      6. Text practice
      7. Settings
      8. Keyboard shortcuts
      9. Troubleshooting (quick)

      > Note: on startup the app first loads TensorFlow and MediaPipe (loading screen).
      > The full UI appears afterwards.

      ---

      ## 1) Quick start (recommended flow)
      1) **Data collection**: collect samples for every sign/class you want.
      2) **Model training**: train on your CSV.
      3) **Sign detection**: start detection and check quality.
      4) (Optional) **Text practice**: practise recognition on text.

      ---

      ## 2) Default files / folders
      ```
      main/data/data.csv            # dataset (features + label column)
      main/images/<label>/...       # saved images
      main/models/model.h5          # trained model
      main/other/scaler.pkl         # feature scaler
      main/other/settings.json      # saved settings
      text_files/*.txt              # text practice files
      ```

      ---

      ## 3) Data collection
      **Goal:** collect examples for each class (e.g. letters/digits and "#").

      ### Step by step
      1. Type a label (e.g. `A`, `B`, `1`, `#`).
      2. Set label (button or `Ctrl+Enter`).
      3. Save samples (`Enter` / `Space` / `Ctrl+S`).

      ### What is saved?
      - one row appended to `data.csv` (landmarks + `label`),
      - one image saved into `images/<label>/...`.

      ### Quality guard
      - saving is blocked when **no hand** is detected or **confidence is too low**,
      - `Undo last save` removes the last CSV row and the corresponding image.

      ### Self-timer
      - set seconds, enable self-timer, press `Start`,
      - `Loop` repeats countdown and saves continuously.

      ### Batch processing (folder of images)
      - choose a folder (including subfolders), set the label,
      - `Process folder` extracts landmarks and appends to CSV, and saves images into `images/<label>/...`,
      - optional overlay draws landmarks/label on output images.

      ### Overlay + Flip
      - overlay shows landmarks and a label (label + index),
      - `Flip H` / `Flip V` flips the camera view.

      ---

      ## 4) Model training
      - `Start training` runs training in the background.
      - `Show advanced` reveals extra parameters.
      - after training you get Loss/Accuracy charts.
      - `Copy chart to clipboard` copies the chart (Windows).

      Tip: if you have a small dataset, start with fewer epochs (e.g. 10).

      ---

      ## 5) Sign detection
      - `Start/Stop` toggles recognition.
      - `Interval (ms)` controls prediction frequency (higher = less CPU).
      - `Threshold` filters predictions (higher = fewer mistakes, more “—”).
      - `Insert only after Enter` requires manual confirmation.
      - `Top 10` shows the most probable classes.

      ---

      ## 6) Text practice
      1. Pick a `.txt` file from `text_files`.
      2. Click `Load text`, then `Start`.
      3. The app checks characters in order and updates stats.

      ---

      ## 7) Settings
      - Light/Dark mode + separate accent color for each,
      - Quick-open: images / models / CSV,
      - custom paths for CSV/model/scaler,
      - MediaPipe parameters (max hands, model_complexity, confidence thresholds).

      ---

      ## 8) Keyboard shortcuts
      > Action shortcuts depend on the active view.
      > If you are typing inside an input field, the app won’t steal Space/Enter.

      ### Navigation
      | Shortcut | Action |
      |---|---|
      | F1 | Instructions |
      | F2 | Data collection |
      | F3 | Sign detection |
      | F4 | Model training |
      | F5 | Text practice |
      | F6 | Settings |
      | Ctrl+1..Ctrl+6 | Same as the sidebar order |
      | Ctrl+Tab | Next view |
      | Ctrl+Shift+Tab | Previous view |

      ### Global
      | Shortcut | Action |
      |---|---|
      | Ctrl+L | Toggle console |
      | Ctrl+Shift+T | Toggle Light/Dark |
      | Ctrl+Q | Quit |

      ### Data collection
      | Shortcut | Action |
      |---|---|
      | Enter / Space / Ctrl+S | Save sample |
      | Ctrl+Z | Undo last save |
      | Ctrl+Enter | Set label from input |
      | Alt+T | Start/Stop self-timer |
      | Alt+Y | Enable/disable self-timer |
      | Alt+L | Self-timer loop |
      | Alt+O | Toggle overlay |
      | Ctrl+B | Browse batch folder |
      | Alt+B | Run batch processing |
      | Alt+R | Restart camera |
      | Alt+H / Alt+V | Flip H / Flip V |

      ### Sign detection
      | Shortcut | Action |
      |---|---|
      | Alt+S | Start |
      | Alt+X | Stop |
      | Alt+C | Clear buffer |
      | Alt+R | Restart camera |
      | Alt+H / Alt+V | Flip H / Flip V |
      | Alt+E | Toggle “only after Enter” |

      ### Training
      | Shortcut | Action |
      |---|---|
      | Alt+R | Start training |
      | Alt+A | Toggle advanced |
      | Alt+C | Copy chart to clipboard |

      ### Text practice
      | Shortcut | Action |
      |---|---|
      | Alt+L | Load text |
      | Alt+S | Start |
      | Alt+X | Stop |
      | Alt+R | Restart camera |
      | Alt+H / Alt+V | Flip H / Flip V |
      | Alt+O | Toggle overlay |

      ---

      ## 9) Troubleshooting (quick)
      - **No video:** close other webcam apps, press `Restart`, change “Cam …”.
      - **Random letters:** increase `Threshold` (e.g. 0.8–0.9), collect more data, retrain.
      - **Slow:** increase `Interval (ms)`, reduce `model_complexity` / max hands.
      - **Detection unavailable:** collect data → train model → then detection.
      """,
    "language_label": "Language:",
  "theme_label": "Theme:",

  # ---------- camera placeholder ----------
  "cam_placeholder_title": "Camera inactive",
  "cam_placeholder_subtitle": "Select a camera and click Restart",

  # ---------- activity log ----------
  "log_header": "Activity log",
  "btn_hide_log": "Hide log",
  "btn_show_log": "Show log",
    "dlg_error": "Error",
"dlg_warning": "Warning",
"dlg_confirm": "Confirmation",
"dlg_quit_app": "Do you want to quit the application?",

"err_label_empty": "Label field is empty. Please provide a value.",
"err_no_filename": "No filename provided. Please select or enter one.",
"err_file_not_exists": "File not found: {file}",
"err_text_file_not_loaded": "A text file has not been loaded yet.",
"err_incomplete_input": "Incomplete input. Please fill in required fields.",
"err_invalid_numbers": "Invalid numeric values provided.",
"err_csv_path_missing": "CSV path is missing. Provide a valid CSV file path.",
"err_test_size_range": "Test size must be a float between 0 and 1.",
"err_epochs_positive": "Number of epochs must be a positive integer.",
"err_batch_positive": "Batch size must be a positive integer.",
"err_patience_nonnegative": "Patience must be zero or a positive integer.",

"warn_no_label": "No label was entered.",
"warn_no_camera": "No camera selected.",
"warn_invalid_range": "Invalid or out-of-range value(s).",

"log_missing_dir": "Missing or non-existent directory: {dir}",
"log_file_loaded": "File loaded successfully: {file}",
"log_csv_read_error": "Error reading CSV file.",
"log_csv_missing_train": "CSV path for training is missing or invalid: {path}",
"log_label_missing_csv": "The 'label' column is missing from the CSV file.",
"log_split_error": "Error while splitting data: {err}",
"log_scaler_saved": "Scaler saved to {path}",
"log_model_summary": "Model summary:\n{summary}",
"log_model_saved": "Model saved to {path}",
"log_test_accuracy": "Test accuracy: {acc}",
"log_confusion_matrix": "Confusion matrix:\n{cm}",
"log_training_finished": "Training has finished successfully.",
"app_title": "Hand Data Collector App",
"err_no_camera": "No camera found or it is currently in use by another application.",
"log_closing": "Closing the application...",
"log_first_set_label": "Please set a label before saving data.",
"log_no_camera_data": "No camera data available. Be sure your camera is working properly.",
"log_no_hand": "No hand detected in the frame.",
"log_images_empty": "Images directory does not exist or is already empty.",
"log_images_cleared": "Images folder has been cleared.",
"log_action_cancelled": "Action cancelled by user.",
"log_csv_missing": "CSV file does not exist: {path}",
"log_csv_reset": "CSV file has been reset: {path}",
"log_reset_defaults": "Default settings have been restored.",
"log_mp_updated": "MediaPipe settings updated.",
"lbl_current_label": "Current label: {val}",
"lbl_current_index": "Current index: {val}",
"dlg_sure_clear_images": "Are you sure you want to clear all images?",
"dlg_sure_reset_csv": "Are you sure you want to reset CSV file: {file}?",
"log_saved_sample": "Saved data for label: {label}, index: {idx} -> appended to CSV: {path}",
"log_csv_total": "CSV now has {total} data rows (excluding header).",
"log_saved_image": "Saved image file: {path}",
"log_folder_count": "Folder for label '{label}' now has {count} images.",
"log_camera_switch": "Switching camera from {old} to {new}",
"log_label_selected": "Label selected: {val}",
"log_no_label": "No label entered or label is empty.",
"status_on": "ON",
"status_off": "OFF",
"log_flip_status": "Flip vertical is now {val}.",
"log_epoch_progress": "Epoch {curr}/{total} - Loss: {loss:.4f}, Acc: {acc:.4f}, Val loss: {vloss:.4f}, Val acc: {vacc:.4f}",
"frame_train_plots":    "Training plots",
"lbl_validation_split": "Validation split:",
"lbl_monitor:":          "EarlyStopping monitor:",
"err_validation_split_range": "Validation split must be between 0 and 1.",
"msg_wait_camera": "Please wait, initializing camera…",
"main_window_title": "Real-Time Sign Language Capture",
"wait_window_title": "Initializing camera…",
"btn_flip_horizontal": "Flip horizontally",
"btn_flip_vertical": "Flip vertically",
"log_flip_horizontal_on": "Horizontal flip: ON",
"log_flip_horizontal_off": "Horizontal flip: OFF",
"log_flip_vertical_on": "Vertical flip: ON",
"log_flip_vertical_off": "Vertical flip: OFF",
"camera_preset_small": "Small",
"camera_preset_medium": "Medium",
"camera_preset_large": "Large",
"overlay_label": "Label: {label}",
"overlay_label_idx": "Label: {label} | Index: {idx}",
"log_view_changed": "View changed: {view}",
"log_view_on_show_error": "View on_show error: {err}",
"log_camera_open_failed": "Cannot open camera index {idx}.",
"log_camera_opened": "Camera opened (index={idx}).",
"log_camera_consumer_error": "Camera consumer error: {err}",
"log_detection_error": "Detection error: {err}",
"log_plot_error": "Plot error: {err}",
"log_history_keys": "History keys: {keys}",
"log_camera_size_preset": "Camera size preset: {preset}",
"log_training_exception": "Training failed: {err}",
"log_settings_loaded": "Loaded settings from: {path}",
"log_settings_load_failed": "Failed to load settings: {err}",
"log_settings_saved": "Saved settings to: {path}",
"log_settings_save_failed": "Failed to save settings: {err}",
"lbl_self_timer": "Self-timer",
"lbl_self_timer_seconds": "Seconds:",
"chk_self_timer_loop": "Loop",
"lbl_self_timer_countdown": "Capturing in {seconds}s…",
"btn_self_timer_start": "Start timer",
"btn_self_timer_cancel": "Cancel",
"err_timer_seconds": "Seconds must be a positive integer.",
"log_self_timer_started": "Self-timer started ({seconds}s).",
"log_self_timer_fired": "Self-timer: saving sample.",
"btn_restart_camera": "Restart/Enable camera",
"log_camera_restarted": "Camera has been restarted"
 ,"log_camera_restarting": "Restarting camera…"

 ,"plot_loss": "Loss"
 ,"plot_accuracy": "Accuracy"
 ,"plot_epoch": "Epoch"
 ,"plot_train": "Train"
 ,"plot_val": "Validation"
}