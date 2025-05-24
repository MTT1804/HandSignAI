import os
import tkinter as tk
import threading
from tkinter import ttk, scrolledtext, messagebox
import cv2
import numpy as np
import mediapipe as mp
import joblib
import tensorflow as tf
from PIL import Image, ImageTk
import pandas as pd
from utils import disable_space_activation
from locales import tr

def show_top10_predictions_text(app, pred_prob):
    if not hasattr(app, "prob_window") or not app.prob_window.winfo_exists():
        app.prob_window = tk.Toplevel(app.root)
        app.prob_window.title(tr("win_top10"))
        app.prob_labels = []
        for _ in range(10):
            lbl = tk.Label(app.prob_window, text="", font=("Roboto", 12))
            lbl.pack(anchor="w", padx=10, pady=2)
            app.prob_labels.append(lbl)

    top10_indices = np.argsort(pred_prob)[::-1][:10]
    top10_probs = pred_prob[top10_indices]
    top10_classes = [app.classes[i] for i in top10_indices]

    for i in range(10):
        if i < len(top10_classes):
            znak = top10_classes[i]
            prob = top10_probs[i] * 100
            app.prob_labels[i].config(text=f"{i+1}. {znak}: {prob:.2f}%")
        else:
            app.prob_labels[i].config(text="")

def run_text_detection(app):
    csv_path = app.csv_file_var.get()
    if not csv_path or not os.path.exists(csv_path):
        app.log(tr("log_no_csv_path"))
        return

    try:
        df = pd.read_csv(csv_path)
        if "label" not in df.columns:
            app.log(tr("log_no_label_column"))
            return
        app.classes = sorted(df["label"].unique().tolist())
        app.log(tr("log_classes_found", classes=app.classes))
    except Exception as e:
        app.log(f"{tr('log_csv_read_error')}: {e}")
        return

    model = tf.keras.models.load_model(app.model_file_var.get())
    scaler = joblib.load(app.scaler_file_var.get())

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils
    cap = app.open_camera(app.current_camera_index)

    def update_text_detection():
        if not app.text_detection_running:
            cap.release()
            hands.close()
            return

        ret, frame = cap.read()
        if ret:
            if app.flip_horizontal:
                 frame = cv2.flip(frame, 1)
            if app.flip_vertical:
                 frame = cv2.flip(frame, 0)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            letter = ""

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                row = [coord for lm in hand_landmarks.landmark for coord in (lm.x, lm.y)]
                X_scaled = scaler.transform(np.array(row).reshape(1, -1))
                pred_prob = model.predict(X_scaled)[0]
                pred_class = int(np.argmax(pred_prob))
                max_prob = float(pred_prob[pred_class])

                try:
                    threshold = float(app.text_threshold_var.get())
                except ValueError:
                    threshold = 0.7

                if max_prob >= threshold:
                    letter = app.classes[pred_class]
                    if letter == "#":
                        letter = " "
                    if app.show_overlays_var.get():
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            imgtk = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            app.text_det_camera_label.imgtk = imgtk
            app.text_det_camera_label.configure(image=imgtk)

            if letter and letter != app.last_text_detected_letter:
                check_letter_correctness_and_stats(app, letter)
                show_top10_predictions_text(app, pred_prob)
                app.last_text_detected_letter = letter

        try:
            interval_ms = int(app.text_interval_var.get())
        except ValueError:
            interval_ms = 50
        app.root.after(interval_ms, update_text_detection)

    update_text_detection()

def check_letter_correctness_and_stats(app, recognised):
    if not app.text_file_content or app.current_char_index >= len(app.text_file_content):
        return
    expected = app.text_file_content[app.current_char_index]
    while expected in " \n\r" and app.current_char_index < len(app.text_file_content):
        app.current_char_index += 1
        if app.current_char_index < len(app.text_file_content):
            expected = app.text_file_content[app.current_char_index]
        else:
            return
    if recognised.upper() == expected.upper():
        start = f"1.0+{app.current_char_index}c"
        end = f"1.0+{app.current_char_index+1}c"
        app.text_detection_text.tag_add("correct", start, end)
        app.current_char_index += 1
        app.recognised_signs += 1
    else:
        app.failed_attempts += 1
    update_text_stats(app)

def update_text_stats(app):
    remain = app.total_signs - app.recognised_signs
    text_ok = tr("stat_correct", ok=app.recognised_signs, total=app.total_signs)
    text_fail = tr("stat_failed", fail=app.failed_attempts)
    text_remain = tr("stat_remaining", remain=remain)
    app.text_stats_label.config(text=text_ok + "\n" + text_fail + "\n" + text_remain)

def create_text_detection_tab(app):
    main_frame = ttk.Frame(app.tab_text_detection)
    main_frame.pack(fill=tk.BOTH, expand=True)

    left = ttk.Frame(main_frame)
    left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    app.text_cam_combo = ttk.Combobox(
        left,
        textvariable=app.camera_var,
        values=[str(i) for i in app.available_cameras],
        state="readonly",
        width=3
    )
    app.text_cam_combo.set(app.current_camera_index)
    app.text_cam_combo.pack(anchor="center", pady=(5,0))
    app.text_cam_combo.bind("<<ComboboxSelected>>", app.on_camera_select)

    app.text_restart_cam_btn = ttk.Button(
        left,
        text=tr("btn_restart_camera"),
        command=app.restart_camera
    )
    app.text_restart_cam_btn.pack(anchor="center", pady=(0,5))
    disable_space_activation(app.text_restart_cam_btn)
    right = ttk.Frame(main_frame, width=300)
    right.pack(side=tk.RIGHT, fill=tk.Y)

    app.text_det_camera_label = ttk.Label(left, text=tr("lbl_cam_preview_text"), font=("Roboto", 12))
    app.text_det_camera_label.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    app.text_detection_text = scrolledtext.ScrolledText(right, height=20, wrap=tk.WORD, font=("Roboto", 12))
    app.text_detection_text.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    app.text_detection_text.tag_config("correct", foreground="green", font=("Roboto", 12, "bold"))

    interval_frame = ttk.Frame(right)
    interval_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
    app.td_interval_label = ttk.Label(interval_frame, text=tr("lbl_interval"), font=("Roboto", 12))
    app.td_interval_label.pack(side=tk.LEFT, padx=5)
    app.text_interval_var = tk.StringVar(value="50")
    ttk.Entry(interval_frame, textvariable=app.text_interval_var, width=7, font=("Roboto", 12)).pack(side=tk.LEFT, padx=5)

    app.td_threshold_label = ttk.Label(interval_frame, text=tr("lbl_threshold"), font=("Roboto", 12))
    app.td_threshold_label.pack(side=tk.LEFT, padx=5)
    app.text_threshold_var = tk.StringVar(value="0.7")
    ttk.Entry(interval_frame, textvariable=app.text_threshold_var, width=5, font=("Roboto", 12)).pack(side=tk.LEFT, padx=5)

    file_frame = ttk.Frame(right)
    file_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
    app.td_choose_file_lbl = ttk.Label(file_frame, text=tr("lbl_select_text_file"), font=("Roboto", 12))
    app.td_choose_file_lbl.pack(side=tk.LEFT, padx=5)
    app.text_file_var = tk.StringVar()
    app.text_file_combo = ttk.Combobox(file_frame, textvariable=app.text_file_var, state="readonly")
    app.text_file_combo.pack(side=tk.LEFT, padx=5)

    text_files_dir = "text_files"
    if os.path.isdir(text_files_dir):
        app.text_file_combo["values"] = [f for f in os.listdir(text_files_dir) if os.path.isfile(os.path.join(text_files_dir, f))]
    else:
        app.log(tr("log_missing_dir", dir=text_files_dir))

    btn_frame = ttk.Frame(right)
    btn_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

    flip_h_btn = ttk.Button(
        btn_frame,
        text=tr("btn_flip_horizontal") or "Flip poziomo",
        command=app.toggle_flip_horizontal
    )
    flip_h_btn.pack(side=tk.LEFT, padx=5)
    disable_space_activation(flip_h_btn)

    flip_v_btn = ttk.Button(
        btn_frame,
        text=tr("btn_flip_vertical") or "Flip pionowo",
        command=app.toggle_flip_vertical
    )
    flip_v_btn.pack(side=tk.LEFT, padx=5)
    disable_space_activation(flip_v_btn)

    app.td_load_btn = ttk.Button(btn_frame, text=tr("btn_load_text"))
    app.td_load_btn.pack(side=tk.LEFT, padx=5)

    app.text_stats_label = ttk.Label(right, text="", font=("Roboto", 12))
    app.text_stats_label.pack(side=tk.TOP, fill=tk.X, pady=5)

    def original_load_text_file():
        file_name = app.text_file_var.get()
        file_path = os.path.join(text_files_dir, file_name)
        with open(file_path, "r", encoding="utf-8") as f:
            app.text_file_content = f.read()
        app.text_detection_text.config(state=tk.NORMAL)
        app.text_detection_text.delete("1.0", tk.END)
        app.text_detection_text.insert(tk.END, app.text_file_content)
        app.text_detection_text.config(state=tk.DISABLED)
        app.total_signs = sum(1 for ch in app.text_file_content if ch not in " \n\r")
        app.current_char_index = 0
        app.recognised_signs = 0
        app.failed_attempts = 0
        update_text_stats(app)
        app.log(tr("log_file_loaded", file=file_path))

    def validated_load_text_file():
        file_name = app.text_file_var.get().strip()
        if not file_name:
            messagebox.showerror(tr("dlg_error"), tr("err_no_filename"))
            return
        file_path = os.path.join(text_files_dir, file_name)
        if not os.path.exists(file_path):
            messagebox.showerror(tr("dlg_error"), tr("err_file_not_exists").format(file=file_path))
            return
        original_load_text_file()

    app.td_load_btn.config(command=validated_load_text_file)

    def original_start_td():
        if app.text_detection_running:
            return
        app.text_detection_running = True
        app.td_start_btn.config(state="disabled")
        app.td_stop_btn.config(state="normal")
        threading.Thread(target=run_text_detection, args=(app,), daemon=True).start()

    def validated_start_td():
        interval_txt = app.text_interval_var.get().strip()
        threshold_txt = app.text_threshold_var.get().strip()
        if not interval_txt or not threshold_txt:
            messagebox.showerror(tr("dlg_error"), tr("err_incomplete_input"))
            return
        try:
            interval_val = int(interval_txt)
            threshold_val = float(threshold_txt)
            if interval_val <= 0 or threshold_val <= 0 or threshold_val > 1:
                messagebox.showwarning(tr("dlg_warning"), tr("warn_invalid_range"))
                return
        except ValueError:
            messagebox.showerror(tr("dlg_error"), tr("err_invalid_numbers"))
            return
        if not app.text_file_content:
            messagebox.showerror(tr("dlg_error"), tr("err_text_file_not_loaded"))
            return
        original_start_td()

    def original_stop_td():
        app.text_detection_running = False
        app.td_start_btn.config(state="normal")
        app.td_stop_btn.config(state="disabled")

    app.td_start_btn = ttk.Button(btn_frame, text=tr("btn_start"), command=validated_start_td)
    app.td_start_btn.pack(side=tk.LEFT, padx=5)
    app.td_stop_btn = ttk.Button(btn_frame, text=tr("btn_stop"), state="disabled", command=original_stop_td)
    app.td_stop_btn.pack(side=tk.LEFT, padx=5)

    app.text_file_content = ""
    app.current_char_index = 0
    app.text_detection_running = False
    app.last_text_detected_letter = ""
    app.total_signs = 0
    app.recognised_signs = 0
    app.failed_attempts = 0
    update_text_stats(app)
    if not hasattr(app, "_i18n_widgets_text"):
        app._i18n_widgets_text = []
        app._i18n_widgets_text.extend([
        (app.td_interval_label,       "lbl_interval"),
        (app.td_threshold_label,      "lbl_threshold"),
        (app.td_choose_file_lbl,      "lbl_select_text_file"),
        (app.td_load_btn,             "btn_load_text"),
        (app.td_start_btn,            "btn_start"),
        (app.td_stop_btn,             "btn_stop"),
        (flip_h_btn,                  "btn_flip_horizontal"),
        (flip_v_btn,                  "btn_flip_vertical"),
    ])
