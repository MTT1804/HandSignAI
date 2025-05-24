import os
import cv2
import joblib
import numpy as np
import mediapipe as mp
from PIL import Image, ImageTk
import tensorflow as tf
import tkinter as tk
from tkinter import ttk, scrolledtext
import pandas as pd

from locales import tr  # internationalisation helper

def show_top10_predictions(app, pred_prob):
    """Create / update a small top‑10 window with class probabilities."""
    if not hasattr(app, "prob_window") or not app.prob_window.winfo_exists():
        app.prob_window = tk.Toplevel(app.root)
        app.prob_window.title(tr("win_top10"))
        app.prob_labels = []
        for _ in range(10):
            lbl = tk.Label(app.prob_window, text="", font=("Roboto", 12))
            lbl.pack(anchor="w", padx=10, pady=2)
            app.prob_labels.append(lbl)

    top10_idx = np.argsort(pred_prob)[::-1][:10]
    top10_probs = pred_prob[top10_idx]
    top10_classes = [app.classes[i] for i in top10_idx]

    for i in range(10):
        if i < len(top10_classes):
            app.prob_labels[i].config(
                text=f"{i+1}. {top10_classes[i]}: {top10_probs[i] * 100:.2f}%"
            )
        else:
            app.prob_labels[i].config(text="")

def run_detection(app):
    """Thread‑based detection loop (called from app.start_detection)."""
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
        app.log(f"CSV error: {e}")
        return

    model = tf.keras.models.load_model(app.model_file_var.get())
    scaler = joblib.load(app.scaler_file_var.get())

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    mp_drawing = mp.solutions.drawing_utils

    app.last_detected_letter = ""
    app.next_letter = ""

    cap = app.open_camera(app.current_camera_index)

    def update_detection():
        if not app.detection_running:
            cap.release()
            hands.close()
            return

        ret, frame = cap.read()
        if app.flip_horizontal:
            frame = cv2.flip(frame, 1)
        if app.flip_vertical:
            frame = cv2.flip(frame, 0)
        if ret:
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
                    threshold = float(app.threshold_var.get())
                except ValueError:
                    threshold = 0.7

                if max_prob >= threshold:
                    letter = app.classes[pred_class]
                    if letter == "#":
                        letter = " "
                    if app.show_overlays_var.get():
                        mp_drawing.draw_landmarks(
                            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                        )

            imgtk = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            app.det_camera_label.imgtk = imgtk
            app.det_camera_label.configure(image=imgtk)

            if letter and letter != app.last_detected_letter:
                app.last_detected_letter = letter
                app.next_letter = letter
                if not app.enter_mode_var.get():
                    app.det_text.insert(tk.END, letter)
                    app.det_text.see(tk.END)
                show_top10_predictions(app, pred_prob)

        try:
            interval_ms = int(app.interval_var.get())
        except ValueError:
            interval_ms = 1000

        app.root.after(interval_ms, update_detection)

    update_detection()


def on_enter(event, app):
    if app.enter_mode_var.get() and app.next_letter:
        app.det_text.insert(tk.END, app.next_letter)
        app.det_text.see(tk.END)
        app.next_letter = ""

def create_detection_tab(app):
    """Create widgets for the detection tab; save references for i18n."""

    app.det_main_frame = ttk.Frame(app.tab_detection)
    app.det_main_frame.pack(fill=tk.BOTH, expand=True)

    app.det_left_frame = ttk.Frame(app.det_main_frame)
    app.det_left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    app.det_right_frame = ttk.Frame(app.det_main_frame, width=300)
    app.det_right_frame.pack(side=tk.RIGHT, fill=tk.Y)

    app.det_camera_label = ttk.Label(app.det_left_frame, font=("Roboto", 12))
    app.det_camera_label.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    app.det_text = scrolledtext.ScrolledText(
        app.det_right_frame, height=20, wrap=tk.WORD, font=("Roboto", 12)
    )
    app.det_text.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    style = ttk.Style()
    style.configure("My.TButton", font=("Roboto", 10))

    control_frame = ttk.Frame(app.det_right_frame)
    control_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)

    interval_frame = ttk.Frame(control_frame)
    interval_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

    app.det_interval_label = ttk.Label(interval_frame, font=("Roboto", 12))
    app.det_interval_label.pack(side=tk.LEFT, padx=5)

    app.interval_var = tk.StringVar(value="1000")
    app.det_interval_entry = ttk.Entry(
        interval_frame, textvariable=app.interval_var, width=7, font=("Roboto", 12)
    )
    app.det_interval_entry.pack(side=tk.LEFT, padx=5)

    app.det_threshold_label = ttk.Label(interval_frame, font=("Roboto", 12))
    app.det_threshold_label.pack(side=tk.LEFT, padx=5)

    app.threshold_var = tk.StringVar(value="0.7")
    app.det_threshold_entry = ttk.Entry(
        interval_frame, textvariable=app.threshold_var, width=5, font=("Roboto", 12)
    )
    app.det_threshold_entry.pack(side=tk.LEFT, padx=5)

    app.enter_mode_var = tk.BooleanVar(value=False)
    app.det_enter_chk = ttk.Checkbutton(
        app.det_right_frame, variable=app.enter_mode_var
    )
    app.det_enter_chk.pack(side=tk.TOP, anchor="w", padx=10, pady=5)

    btn_frame = ttk.Frame(control_frame)
    btn_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

    app.start_det_button = ttk.Button(
        btn_frame, command=app.start_detection, style="My.TButton"
    )
    app.start_det_button.pack(side=tk.LEFT, padx=5)

    app.stop_det_button = ttk.Button(
        btn_frame, command=app.stop_detection, state="disabled", style="My.TButton"
    )
    app.stop_det_button.pack(side=tk.LEFT, padx=5)

    app.clear_screen_button = ttk.Button(
        btn_frame,
        command=lambda: app.det_text.delete("1.0", tk.END),
        style="My.TButton",
    )
    app.clear_screen_button.pack(side=tk.LEFT, padx=5)

    app.next_letter = ""
    app.root.bind_all("<Return>", lambda event: on_enter(event, app))


    if not hasattr(app, "_i18n_widgets_det"):
        app._i18n_widgets_det = []
    app._i18n_widgets_det.extend([
        (app.det_interval_label, "lbl_interval"),
        (app.det_threshold_label, "lbl_threshold"),
        (app.det_enter_chk, "chk_enter_mode"),
        (app.start_det_button, "btn_start_detection"),
        (app.stop_det_button, "btn_stop_detection"),
        (app.clear_screen_button, "btn_clear_screen"),
    ])

    for widget, key in app._i18n_widgets_det:
        if isinstance(widget, ttk.Checkbutton):
            widget.config(text=tr(key))
        else:
            widget.config(text=tr(key))
