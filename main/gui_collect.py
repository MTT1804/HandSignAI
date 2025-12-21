import tkinter as tk
from tkinter import ttk, messagebox
from utils import disable_space_activation
from locales import tr

def create_collect_tab(app):
    app.tab_collect.grid_rowconfigure(0, weight=1)
    app.tab_collect.grid_columnconfigure(0, weight=2)
    app.tab_collect.grid_columnconfigure(1, weight=1)
    app.tab_collect.grid_columnconfigure(2, weight=1)

    app.collect_left_frame = ttk.Frame(app.tab_collect, padding=4)
    app.collect_left_frame.grid(row=0, column=0, sticky="nsew")
    app.collect_left_frame.grid_rowconfigure(0, weight=1)
    app.collect_left_frame.grid_columnconfigure(0, weight=1)

    app.camera_label = ttk.Label(app.collect_left_frame, anchor="n", text="")
    app.camera_label.grid(row=0, column=0, sticky="nsew")

    app.collect_right_frame = ttk.Frame(app.tab_collect, padding=4)
    app.collect_right_frame.grid(row=0, column=1, columnspan=2, sticky="nsew")
    app.collect_right_frame.grid_columnconfigure(0, weight=1)
    app.collect_right_frame.grid_columnconfigure(1, weight=1)
    app.collect_right_frame.grid_rowconfigure(0, weight=1)
    app.collect_right_frame.grid_rowconfigure(1, weight=1)

    app.controls_frame = ttk.LabelFrame(app.collect_right_frame, text=tr("section_data_mgmt"))
    app.controls_frame.grid(row=0, column=0, sticky="nsew", padx=4, pady=(0, 6))

    app.mp_frame = ttk.LabelFrame(app.collect_right_frame, text=tr("section_mediapipe"))
    app.mp_section_frame = app.mp_frame
    app.mp_frame.grid(row=0, column=1, sticky="nsew", padx=4, pady=(0, 6))
    app.collect_right_frame.grid_columnconfigure(1, weight=1)

    _create_collect_controls(app)
    _create_mediapipe_frame(app)

    def _adjust_collect_layout(event=None):
        # Decide layout based on the available width of the right-side panel.
        # This better matches what the user sees (and avoids unnecessary vertical stacking).
        width = app.collect_right_frame.winfo_width()
        if width <= 1:
            width = app.tab_collect.winfo_width()

        # When there is enough space, keep MediaPipe next to data/image controls.
        if width >= 650:
            app.controls_frame.grid_configure(row=0, column=0, columnspan=1, sticky="nsew")
            app.mp_frame.grid_configure(row=0, column=1, columnspan=1, sticky="nsew")
            app.collect_right_frame.grid_rowconfigure(1, weight=0)
        else:
            app.controls_frame.grid_configure(row=0, column=0, columnspan=2, sticky="nsew")
            app.mp_frame.grid_configure(row=1, column=0, columnspan=2, sticky="nsew")
            app.collect_right_frame.grid_rowconfigure(1, weight=1)
        app.collect_right_frame.update_idletasks()

    app.tab_collect.bind("<Configure>", _adjust_collect_layout)
    _adjust_collect_layout()


def _create_collect_controls(app):
    for i in range(2):
        app.controls_frame.grid_columnconfigure(i, weight=1)

    row = 0
    app.col_lbl_choose_cam = ttk.Label(app.controls_frame, text=tr("lbl_choose_camera"))
    app.col_lbl_choose_cam.grid(row=row, column=0, sticky="w", padx=6, pady=(6, 2))
    app.camera_combo = ttk.Combobox(
        app.controls_frame,
        textvariable=app.camera_var,
        values=[str(i) for i in app.available_cameras],
        state="readonly",
        width=4
    )
    app.camera_combo.grid(row=row, column=1, sticky="ew", padx=6, pady=(6, 2))
    app.camera_combo.bind("<<ComboboxSelected>>", app.on_camera_select)

    row += 1
    app.restart_cam_btn = ttk.Button(
        app.controls_frame,
        text=tr("btn_restart_camera").replace("\n", " "),
        command=app.restart_camera
    )
    app.restart_cam_btn.grid(row=row, column=0, columnspan=2, sticky="ew", padx=6, pady=(0, 8))

    row += 1
    app.col_lbl_enter_label = ttk.Label(app.controls_frame, text=tr("lbl_enter_label"))
    app.col_lbl_enter_label.grid(row=row, column=0, columnspan=2, sticky="w", padx=6, pady=(0, 2))
    row += 1
    app.label_entry = ttk.Entry(app.controls_frame)
    app.label_entry.grid(row=row, column=0, columnspan=2, sticky="ew", padx=6, pady=(0, 6))

    original_set_label_cmd = app.col_btn_set_label['command'] if hasattr(app, "col_btn_set_label") else app.set_label
    def validated_set_label():
        label_text = app.label_entry.get().strip()
        if not label_text:
            messagebox.showerror(tr("dlg_error"), tr("err_label_empty"))
            return
        original_set_label_cmd()

    app.col_btn_set_label = ttk.Button(
        app.controls_frame,
        text=tr("btn_set_label"),
        command=validated_set_label
    )
    row += 1
    app.col_btn_set_label.grid(row=row, column=0, columnspan=2, sticky="ew", padx=6, pady=(0, 4))
    disable_space_activation(app.col_btn_set_label)

    original_save_data_cmd = app.col_btn_save['command'] if hasattr(app, "col_btn_save") else app.save_data
    def validated_save_data():
        label_text = app.label_entry.get().strip()
        if not label_text:
            messagebox.showwarning(tr("dlg_warning"), tr("warn_no_label"))
            return
        if not app.camera_combo.get():
            messagebox.showwarning(tr("dlg_warning"), tr("warn_no_camera"))
            return
        original_save_data_cmd()

    app.col_btn_save = ttk.Button(
        app.controls_frame,
        text=tr("btn_save_data"),
        command=validated_save_data
    )
    row += 1
    app.col_btn_save.grid(row=row, column=0, columnspan=2, sticky="ew", padx=6, pady=(0, 6))
    disable_space_activation(app.col_btn_save)

    app.col_btn_flip = ttk.Button(
        app.controls_frame,
        text=tr("btn_flip_horizontal") or "Flip poziomo",
        command=app.toggle_flip_horizontal
    )
    app.col_btn_flip_vertical = ttk.Button(
        app.controls_frame,
        text=tr("btn_flip_vertical") or "Flip pionowo",
        command=app.toggle_flip_vertical
    )
    row += 1
    app.col_btn_flip.grid(row=row, column=0, sticky="ew", padx=6, pady=(0, 4))
    app.col_btn_flip_vertical.grid(row=row, column=1, sticky="ew", padx=6, pady=(0, 4))
    disable_space_activation(app.col_btn_flip_vertical)
    disable_space_activation(app.col_btn_flip)

    row += 1
    app.col_section_data = ttk.Label(app.controls_frame, text=tr("section_data_mgmt"))
    app.col_section_data.grid(row=row, column=0, columnspan=2, sticky="w", padx=6, pady=(8, 2))

    original_clear_images_cmd = app.col_btn_clear_images['command'] if hasattr(app, "col_btn_clear_images") else app.clear_images
    def validated_clear_images():
        original_clear_images_cmd()

    app.col_btn_clear_images = ttk.Button(
        app.controls_frame,
        text=tr("btn_clear_images"),
        command=validated_clear_images
    )
    row += 1
    app.col_btn_clear_images.grid(row=row, column=0, sticky="ew", padx=6, pady=(0, 4))
    disable_space_activation(app.col_btn_clear_images)

    original_clear_csv_cmd = app.col_btn_clear_csv['command'] if hasattr(app, "col_btn_clear_csv") else app.clear_csv
    def validated_clear_csv():
        path = app.csv_file_var.get()
        if not path:
            messagebox.showerror(tr("dlg_error"), tr("err_csv_path_missing"))
            return
        original_clear_csv_cmd()


    app.col_btn_clear_csv = ttk.Button(
        app.controls_frame,
        text=tr("btn_reset_csv"),
        command=validated_clear_csv
    )
    app.col_btn_clear_csv.grid(row=row, column=1, sticky="ew", padx=6, pady=(0, 4))
    disable_space_activation(app.col_btn_clear_csv)

    row += 1
    app.col_section_reset = ttk.Label(app.controls_frame, text=tr("section_reset"))
    app.col_section_reset.grid(row=row, column=0, columnspan=2, sticky="w", padx=6, pady=(8, 2))

    app.col_btn_reset_defaults = ttk.Button(
        app.controls_frame,
        text=tr("btn_reset_defaults"),
        command=app.reset_to_defaults
    )
    row += 1
    app.col_btn_reset_defaults.grid(row=row, column=0, sticky="ew", padx=6, pady=(0, 4))
    disable_space_activation(app.col_btn_reset_defaults)

    original_quit_cmd = app.col_btn_quit['command'] if hasattr(app, "col_btn_quit") else app.quit_app
    def validated_quit():
        if messagebox.askokcancel(tr("dlg_confirm"), tr("dlg_quit_app")):
            original_quit_cmd()

    app.col_btn_quit = ttk.Button(
        app.controls_frame,
        text=tr("btn_quit"),
        command=validated_quit
    )
    app.col_btn_quit.grid(row=row, column=1, sticky="ew", padx=6, pady=(0, 4))
    disable_space_activation(app.col_btn_quit)
    row += 1
    app.col_section_img = ttk.Label(app.controls_frame, text=tr("section_img_settings"))
    app.col_section_img.grid(row=row, column=0, columnspan=2, sticky="w", padx=6, pady=(10, 4))

    def add_scale(label_attr, label_attr_name, var_name, value_attr_name, scale_attr_name, default, from_, to_, row_idx):
        label = ttk.Label(app.controls_frame, text=tr(label_attr))
        label.grid(row=row_idx, column=0, sticky="w", padx=6, pady=(0, 2))
        var = getattr(app, var_name, tk.IntVar(value=default))
        var.set(default)
        setattr(app, var_name, var)
        value_lbl = ttk.Label(app.controls_frame, text=str(default))
        value_lbl.grid(row=row_idx, column=1, sticky="e", padx=6)
        scale = ttk.Scale(
            app.controls_frame,
            from_=from_,
            to=to_,
            variable=var,
            orient=tk.HORIZONTAL,
            command=lambda e, v=var, lbl=value_lbl: app.update_scale_label(v, lbl)
        )
        scale.grid(row=row_idx+1, column=0, columnspan=2, sticky="ew", padx=6, pady=(0, 6))
        setattr(app, label_attr_name, label)
        setattr(app, value_attr_name, value_lbl)
        setattr(app, scale_attr_name, scale)

    row += 1
    add_scale("lbl_brightness", "col_lbl_brightness", "brightness_var", "brightness_value_label", "brightness_scale", 0, -100, 100, row)
    row += 2
    add_scale("lbl_contrast", "col_lbl_contrast", "contrast_var", "contrast_value_label", "contrast_scale", 100, 0, 200, row)
    row += 2
    add_scale("lbl_gamma", "col_lbl_gamma", "gamma_var", "gamma_value_label", "gamma_scale", 100, 1, 300, row)
    row += 2

    app.col_lbl_color_shift = ttk.Label(app.controls_frame, text=tr("lbl_color_shift"))
    app.col_lbl_color_shift.grid(row=row, column=0, columnspan=2, sticky="w", padx=6, pady=(4, 2))
    row += 1
    frame_rgb = ttk.Frame(app.controls_frame)
    frame_rgb.grid(row=row, column=0, columnspan=2, sticky="ew", padx=6, pady=(0, 6))
    frame_rgb.grid_columnconfigure(1, weight=1)

    app.col_lbl_R = ttk.Label(frame_rgb, text=tr("lbl_R"))
    app.col_lbl_R.grid(row=0, column=0, sticky=tk.W)
    app.r_var = tk.IntVar(value=0)
    app.r_value_label = ttk.Label(frame_rgb, text="0")
    app.r_value_label.grid(row=0, column=2, sticky=tk.W, padx=(5, 0))
    r_scale = ttk.Scale(
        frame_rgb,
        from_=-50,
        to=50,
        variable=app.r_var,
        orient=tk.HORIZONTAL,
        length=140,
        command=lambda e: app.update_scale_label(app.r_var, app.r_value_label)
    )
    r_scale.grid(row=0, column=1, padx=5, sticky="ew")

    app.col_lbl_G = ttk.Label(frame_rgb, text=tr("lbl_G"))
    app.col_lbl_G.grid(row=1, column=0, sticky=tk.W)
    app.g_var = tk.IntVar(value=0)
    app.g_value_label = ttk.Label(frame_rgb, text="0")
    app.g_value_label.grid(row=1, column=2, sticky=tk.W, padx=(5, 0))
    g_scale = ttk.Scale(
        frame_rgb,
        from_=-50,
        to=50,
        variable=app.g_var,
        orient=tk.HORIZONTAL,
        length=140,
        command=lambda e: app.update_scale_label(app.g_var, app.g_value_label)
    )
    g_scale.grid(row=1, column=1, padx=5, sticky="ew")

    app.col_lbl_B = ttk.Label(frame_rgb, text=tr("lbl_B"))
    app.col_lbl_B.grid(row=2, column=0, sticky=tk.W)
    app.b_var = tk.IntVar(value=0)
    app.b_value_label = ttk.Label(frame_rgb, text="0")
    app.b_value_label.grid(row=2, column=2, sticky=tk.W, padx=(5, 0))
    b_scale = ttk.Scale(
        frame_rgb,
        from_=-50,
        to=50,
        variable=app.b_var,
        orient=tk.HORIZONTAL,
        length=140,
        command=lambda e: app.update_scale_label(app.b_var, app.b_value_label)
    )
    b_scale.grid(row=2, column=1, padx=5, sticky="ew")


def _create_mediapipe_frame(app):
    for i in range(2):
        app.mp_frame.grid_columnconfigure(i, weight=1)

    row = 0
    app.mp_chk_static = ttk.Checkbutton(
        app.mp_frame,
        text=tr("chk_static_img_mode"),
        variable=app.static_image_mode_var
    )
    app.mp_chk_static.grid(row=row, column=0, columnspan=2, sticky="w", padx=6, pady=(6, 4))

    def add_mp_scale(label_attr, var, row_idx, from_, to_):
        label = ttk.Label(app.mp_frame, text=tr(label_attr))
        label.grid(row=row_idx, column=0, sticky="w", padx=6)
        value_lbl = ttk.Label(app.mp_frame, text=str(var.get()))
        value_lbl.grid(row=row_idx, column=1, sticky="e", padx=6)
        scale = ttk.Scale(
            app.mp_frame,
            from_=from_,
            to=to_,
            variable=var,
            orient=tk.HORIZONTAL,
            command=lambda e, v=var, lbl=value_lbl: app.update_scale_label(v, lbl)
        )
        scale.grid(row=row_idx+1, column=0, columnspan=2, sticky="ew", padx=6, pady=(0, 6))
        return label, value_lbl, scale

    row += 1
    app.mp_lbl_max_hands, app.max_hands_label, app.max_num_hands_scale = add_mp_scale(
        "lbl_max_num_hands", app.max_num_hands_var, row, 1, 4
    )
    row += 2
    app.mp_lbl_model_complexity, app.model_complexity_label, app.model_complexity_scale = add_mp_scale(
        "lbl_model_complexity", app.model_complexity_var, row, 0, 2
    )
    row += 2
    app.mp_lbl_min_det, app.min_detection_label, app.min_detection_scale = add_mp_scale(
        "lbl_min_det_conf", app.min_detection_confidence_var, row, 0, 100
    )
    row += 2
    app.mp_lbl_min_track, app.min_tracking_label, app.min_tracking_scale = add_mp_scale(
        "lbl_min_track_conf", app.min_tracking_confidence_var, row, 0, 100
    )
    row += 2

    app.mp_btn_apply = ttk.Button(
        app.mp_frame,
        text=tr("btn_apply_mp"),
        command=app.update_mediapipe_settings
    )
    app.mp_btn_apply.grid(row=row, column=0, columnspan=2, sticky="ew", padx=6, pady=(0, 6))
    disable_space_activation(app.mp_btn_apply)

    row += 1
    app.mp_chk_show_overlays = ttk.Checkbutton(
        app.mp_frame,
        text=tr("chk_show_overlays"),
        variable=app.show_overlays_var
    )
    app.mp_chk_show_overlays.grid(row=row, column=0, columnspan=2, sticky="w", padx=6, pady=(0, 8))

    row += 1
    app.mp_file_frame = ttk.LabelFrame(app.mp_frame, text=tr("frame_file_labels"))
    app.mp_file_frame.grid(row=row, column=0, columnspan=2, sticky="ew", padx=6, pady=(0, 6))
    app.mp_file_frame.grid_columnconfigure(1, weight=1)

    app.mp_lbl_csv = ttk.Label(app.mp_file_frame, text=tr("lbl_csv_file"))
    app.mp_lbl_csv.grid(row=0, column=0, sticky=tk.W, padx=4, pady=2)
    ttk.Entry(app.mp_file_frame, textvariable=app.csv_file_var, width=28).grid(
        row=0, column=1, sticky=tk.EW, padx=4, pady=2
    )

    app.mp_lbl_model = ttk.Label(app.mp_file_frame, text=tr("lbl_model_file"))
    app.mp_lbl_model.grid(row=1, column=0, sticky=tk.W, padx=4, pady=2)
    ttk.Entry(app.mp_file_frame, textvariable=app.model_file_var, width=28).grid(
        row=1, column=1, sticky=tk.EW, padx=4, pady=2
    )

    app.mp_lbl_scaler = ttk.Label(app.mp_file_frame, text=tr("lbl_scaler_file"))
    app.mp_lbl_scaler.grid(row=2, column=0, sticky=tk.W, padx=4, pady=2)
    ttk.Entry(app.mp_file_frame, textvariable=app.scaler_file_var, width=28).grid(
        row=2, column=1, sticky=tk.EW, padx=4, pady=2
    )