import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from utils import disable_space_activation
from locales import tr

def create_detection_tab(app):
    app.det_main_frame = ttk.Frame(app.tab_detection)
    app.det_main_frame.pack(fill=tk.BOTH, expand=True)

    app.det_left_frame = ttk.Frame(app.det_main_frame)
    app.det_left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    app.det_cam_combo = ttk.Combobox(
        app.det_left_frame,
        textvariable=app.camera_var,
        values=[str(i) for i in app.available_cameras],
        state="readonly",
        width=3
    )
    app.det_cam_combo.set(app.current_camera_index)
    app.det_cam_combo.pack(anchor="center", pady=(5,0))
    app.det_cam_combo.bind("<<ComboboxSelected>>", app.on_camera_select)

    app.det_restart_cam_btn = ttk.Button(
        app.det_left_frame,
        text=tr("btn_restart_camera"),
        command=app.restart_camera
    )
    app.det_restart_cam_btn.pack(anchor="center", pady=(0,5))

    app.det_camera_label = ttk.Label(app.det_left_frame, font=("Roboto", 12))
    app.det_camera_label.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    app.det_right_frame = ttk.Frame(app.det_main_frame, width=300)
    app.det_right_frame.pack(side=tk.RIGHT, fill=tk.Y)

    app.det_text = scrolledtext.ScrolledText(
        app.det_right_frame,
        height=20,
        wrap=tk.WORD,
        font=("Roboto", 12),
    )
    app.det_text.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    control_frame = ttk.Frame(app.det_right_frame)
    control_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)

    interval_frame = ttk.Frame(control_frame)
    interval_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

    app.det_interval_label = ttk.Label(interval_frame, text=tr("lbl_interval"), font=("Roboto", 12))
    app.det_interval_label.pack(side=tk.LEFT, padx=5)

    app.interval_var = tk.StringVar(value="1000")
    interval_entry = ttk.Entry(interval_frame, textvariable=app.interval_var, width=7, font=("Roboto", 12))
    interval_entry.pack(side=tk.LEFT, padx=5)

    app.det_threshold_label = ttk.Label(interval_frame, text=tr("lbl_threshold"), font=("Roboto", 12))
    app.det_threshold_label.pack(side=tk.LEFT, padx=5)

    app.threshold_var = tk.StringVar(value="0.7")
    threshold_entry = ttk.Entry(interval_frame, textvariable=app.threshold_var, width=5, font=("Roboto", 12))
    threshold_entry.pack(side=tk.LEFT, padx=5)

    if not hasattr(app, "enter_mode_var"):
        app.enter_mode_var = tk.BooleanVar(value=False)

    app.det_enter_chk = ttk.Checkbutton(
        control_frame,
        text=tr("chk_enter_mode"),
        variable=app.enter_mode_var,
    )
    app.det_enter_chk.pack(side=tk.TOP, anchor="w", padx=10, pady=5)

    btn_frame = ttk.Frame(control_frame)
    btn_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

    flip_h_det = ttk.Button(
        btn_frame,
        text=tr("btn_flip_horizontal") or "Flip poziomo",
        command=app.toggle_flip_horizontal
    )
    flip_h_det.pack(side=tk.LEFT, padx=5)
    disable_space_activation(flip_h_det)

    flip_v_det = ttk.Button(
        btn_frame,
        text=tr("btn_flip_vertical") or "Flip pionowo",
        command=app.toggle_flip_vertical
    )
    flip_v_det.pack(side=tk.LEFT, padx=5)
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
    app.det_start_btn.pack(side=tk.LEFT, padx=5)

    original_stop_detection_cmd = app.stop_detection
    def validated_stop_detection():
        original_stop_detection_cmd()

    app.det_stop_btn = ttk.Button(btn_frame, text=tr("btn_stop_detection"), command=validated_stop_detection, state="disabled")
    app.det_stop_btn.pack(side=tk.LEFT, padx=5)

    app.det_clear_btn = ttk.Button(
        btn_frame,
        text=tr("btn_clear_screen"),
        command=lambda: app.det_text.delete("1.0", tk.END),
    )
    app.det_clear_btn.pack(side=tk.LEFT, padx=5)

    app.root.bind("<Return>", lambda event: on_enter(event, app))

def on_enter(event, app):
    if app.enter_mode_var.get() and app.next_letter:
        app.det_text.insert(tk.END, app.next_letter)
        app.det_text.see(tk.END)
        app.next_letter = ""
