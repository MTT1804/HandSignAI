STRINGS: dict[str, str] = {
    # ---------- notebook tabs ----------
    "tab_collect":          "Data collection",
    "tab_train":            "Model training",
    "tab_detection":        "Sign detection",
    "tab_text":             "Text signing",
    "tab_instr":            "Instructions",

    # ---------- common buttons / labels ----------
    "btn_start":            "Start",
    "btn_stop":             "Stop",
    "btn_clear_screen":     "Clear screen",
    "lbl_interval":         "Interval (ms):",
    "lbl_threshold":        "Threshold:",

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
    "log_saved_image":      "Saved image to {path}.",
        "instructions_text": """\
INTERFACE GUIDE (ENGLISH)
=========================

1. DATA COLLECTION
   – Pick a camera, type a character, hit “Set letter”.
   – Press “Save data” or Enter to capture frames (saved to images/ + CSV).
   – Flip vertical with Tab; sliders fix Brightness, Contrast, Gamma, RGB.
   – Clear images folder and reset CSV delete all collected samples.

2. MODEL TRAINING
   – Paths to CSV, model and scaler are pre-filled.
   – Keep or change parameters (Test size 0.2, Epochs 30, etc.).
   – Click “Start training” and wait for 100%; model.h5 & scaler.pkl are created.

3. SIGN DETECTION
   – Click “Start Detection”, detected letters appear in the text box.
   – Adjust “Threshold” to filter out wrong guesses.
   – Increase “Interval (ms)” or lower model_complexity if slow.

4. TEXT PRACTICE
   – Choose a .txt file → “Load text” → “Start”.
   – Correctly recognised letters turn green; stats update live.

COMMON HICCUPS → QUICK FIXES
• No video?
  – Close other webcam apps or pick a different camera index.
• Blurry/dark image?
  – Increase Brightness/Contrast, tweak Gamma or RGB sliders.
• No hand detected?
  – Center your hand, improve lighting.
• Empty CSV?
  – Record at least one frame in Data Collection.
• Training errors (stratify, etc.)?
  – Ensure every class has samples; collect more if needed.
• Training too slow?
  – Lower Epochs or Batch size.
• Model won’t load?
  – Verify .h5 path and TensorFlow version.
• Scaler load fails?
  – Point to the correct scaler.pkl.
• Detection slow?
  – Increase Interval(ms) or decrease model_complexity.
• Random letters?
  – Raise Threshold or retrain with better data.
• Missing text_files folder?
  – Create it and add plain .txt files.
• “Load text” does nothing?
  – Ensure files are plain UTF‑8 text without BOM.
• App freezes/crashes?
  – Check other/logs.log and available RAM.
• Keyboard shortcuts unresponsive?
  – Focus the app window; use Tab=Flip, Space/Enter=Save, q=Quit.
""",
    "language_label": "Language/Język:",
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
"btn_restart_camera": "Restart camera\nif it doesn't work",
"log_camera_restarted": "Camera has been restarted"
}