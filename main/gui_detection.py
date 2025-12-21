import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk, scrolledtext, messagebox
from utils import disable_space_activation
from locales import tr

def create_detection_tab(app):
    app.tab_detection.grid_rowconfigure(0, weight=1)
    app.tab_detection.grid_columnconfigure(0, weight=1)

    app.det_main_frame = ttk.Frame(app.tab_detection)
    app.det_main_frame.grid(row=0, column=0, sticky="nsew")
    # 2-row layout:
    # - row 0: controls (buttons/entries)
    # - row 1: camera + text (text height driven by the camera image)
    app.det_main_frame.grid_rowconfigure(0, weight=0)
    app.det_main_frame.grid_rowconfigure(1, weight=0)
    app.det_main_frame.grid_columnconfigure(0, weight=2)
    app.det_main_frame.grid_columnconfigure(1, weight=1)

    # Top controls
    control_frame = ttk.Frame(app.det_main_frame)
    control_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))
    control_frame.grid_columnconfigure(0, weight=1)

    # Content (below controls)
    app.det_camera_label = ttk.Label(app.det_main_frame, font=("Roboto", 12), anchor="n")
    app.det_camera_label.grid(row=1, column=0, sticky="nw")

    app.det_text = scrolledtext.ScrolledText(
        app.det_main_frame,
        height=20,
        wrap=tk.WORD,
        font=("Roboto", 12),
    )
    # Keep text from stretching vertically; height is controlled by _sync_text_height_to_camera.
    app.det_text.grid(row=1, column=1, sticky="nw", padx=(10, 0))

    def _sync_text_height_to_camera(_event=None):
        # Match text widget height (in lines) to current camera label height (in pixels)
        # without scaling the camera image.
        h = app.det_camera_label.winfo_height()
        if h <= 1:
            return
        try:
            f = tkfont.Font(font=app.det_text["font"])
            line_h = max(1, int(f.metrics("linespace")))
        except Exception:
            line_h = 18
        lines = max(3, int(h / line_h))
        app.det_text.configure(height=lines)

    app.det_camera_label.bind("<Configure>", _sync_text_height_to_camera)

    interval_frame = ttk.Frame(control_frame)
    interval_frame.grid(row=0, column=0, sticky="ew", pady=5)

    app.det_interval_label = ttk.Label(interval_frame, text=tr("lbl_interval"), font=("Roboto", 12))
    app.det_interval_label.grid(row=0, column=0, sticky="w", padx=5)

    app.interval_var = tk.StringVar(value="1000")
    interval_entry = ttk.Entry(interval_frame, textvariable=app.interval_var, width=7, font=("Roboto", 12))
    interval_entry.grid(row=0, column=1, sticky="w", padx=5)

    app.det_threshold_label = ttk.Label(interval_frame, text=tr("lbl_threshold"), font=("Roboto", 12))
    app.det_threshold_label.grid(row=0, column=2, sticky="w", padx=5)

    app.threshold_var = tk.StringVar(value="0.7")
    threshold_entry = ttk.Entry(interval_frame, textvariable=app.threshold_var, width=5, font=("Roboto", 12))
    threshold_entry.grid(row=0, column=3, sticky="w", padx=5)

    if not hasattr(app, "enter_mode_var"):
        app.enter_mode_var = tk.BooleanVar(value=False)

    app.det_enter_chk = ttk.Checkbutton(
        control_frame,
        text=tr("chk_enter_mode"),
        variable=app.enter_mode_var,
    )
    app.det_enter_chk.grid(row=1, column=0, sticky="w", padx=10, pady=5)

    btn_frame = ttk.Frame(control_frame)
    btn_frame.grid(row=2, column=0, sticky="w", pady=5)

    # Camera controls moved to the bottom, next to other buttons
    app.det_cam_combo = ttk.Combobox(
        btn_frame,
        textvariable=app.camera_var,
        values=[str(i) for i in app.available_cameras],
        state="readonly",
        width=3
    )
    app.det_cam_combo.set(app.current_camera_index)
    app.det_cam_combo.grid(row=0, column=0, padx=(5, 5))
    app.det_cam_combo.bind("<<ComboboxSelected>>", app.on_camera_select)

    app.det_restart_cam_btn = ttk.Button(
        btn_frame,
        text=tr("btn_restart_camera").replace("\n", " "),
        command=app.restart_camera
    )
    app.det_restart_cam_btn.grid(row=0, column=1, padx=5)
    disable_space_activation(app.det_restart_cam_btn)

    flip_h_det = ttk.Button(
        btn_frame,
        text=tr("btn_flip_horizontal") or "Flip poziomo",
        command=app.toggle_flip_horizontal
    )
    flip_h_det.grid(row=0, column=2, padx=5)
    disable_space_activation(flip_h_det)

    flip_v_det = ttk.Button(
        btn_frame,
        text=tr("btn_flip_vertical") or "Flip pionowo",
        command=app.toggle_flip_vertical
    )
    flip_v_det.grid(row=0, column=3, padx=5)
    disable_space_activation(flip_v_det)

    original_start_detection_cmd = app.start_detection
    def validated_start_detection():
        interval_txt = app.interval_var.get().strip()
        threshold_txt = app.threshold_var.get().strip()
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
        original_start_detection_cmd()

    app.det_start_btn = ttk.Button(btn_frame, text=tr("btn_start_detection"), command=validated_start_detection)
    app.det_start_btn.grid(row=0, column=4, padx=5)

    original_stop_detection_cmd = app.stop_detection
    def validated_stop_detection():
        original_stop_detection_cmd()

    app.det_stop_btn = ttk.Button(btn_frame, text=tr("btn_stop_detection"), command=validated_stop_detection, state="disabled")
    app.det_stop_btn.grid(row=0, column=5, padx=5)

    app.det_clear_btn = ttk.Button(
        btn_frame,
        text=tr("btn_clear_screen"),
        command=lambda: app.det_text.delete("1.0", tk.END),
    )
    app.det_clear_btn.grid(row=0, column=6, padx=5)

    app.root.bind("<Return>", lambda event: on_enter(event, app))

def on_enter(event, app):
    if app.enter_mode_var.get() and app.next_letter:
        app.det_text.insert(tk.END, app.next_letter)
        app.det_text.see(tk.END)
        app.next_letter = ""
