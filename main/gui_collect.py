import tkinter as tk
from tkinter import ttk, messagebox
from utils import disable_space_activation
from locales import tr

def create_collect_tab(app):
    app.collect_left_frame = ttk.Frame(app.tab_collect)
    app.collect_left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    app.camera_label = ttk.Label(app.collect_left_frame)
    app.camera_label.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    app.collect_right_frame = ttk.Frame(app.tab_collect, width=550)
    app.collect_right_frame.pack(side=tk.RIGHT, fill=tk.Y)

    app.controls_frame = ttk.Frame(app.collect_right_frame)
    app.controls_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

    app.mp_frame = ttk.Frame(app.collect_right_frame)
    app.mp_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)

    _create_collect_controls(app)
    _create_mediapipe_frame(app)


def _create_collect_controls(app):
    app.col_lbl_choose_cam = ttk.Label(app.controls_frame, text=tr("lbl_choose_camera"))
    app.col_lbl_choose_cam.pack(anchor="w", padx=5, pady=(5,0))

    app.camera_combo = ttk.Combobox(
        app.controls_frame,
        textvariable=app.camera_var,                       
        values=[str(i) for i in app.available_cameras],   
        state="readonly",
        width=3
    )
    app.camera_combo.pack(anchor="w", padx=5, pady=5)
    app.camera_combo.bind("<<ComboboxSelected>>", app.on_camera_select)

    app.restart_cam_btn = ttk.Button(
        app.controls_frame,
        text=tr("btn_restart_camera"),
        command=app.restart_camera
    )
    app.restart_cam_btn.pack(anchor="w", padx=5, pady=(0,10))

    app.col_lbl_enter_label = ttk.Label(app.controls_frame, text=tr("lbl_enter_label"))
    app.col_lbl_enter_label.pack(pady=5)

    app.label_entry = ttk.Entry(app.controls_frame)
    app.label_entry.pack(padx=5, pady=5)

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
    app.col_btn_set_label.pack(pady=5)
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
    app.col_btn_save.pack(pady=5)
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
    app.col_btn_flip_vertical.pack(pady=5)
    disable_space_activation(app.col_btn_flip_vertical)
    app.col_btn_flip.pack(pady=5)
    disable_space_activation(app.col_btn_flip)

    app.col_section_data = ttk.Label(app.controls_frame, text=tr("section_data_mgmt"))
    app.col_section_data.pack(pady=(10, 0))

    original_clear_images_cmd = app.col_btn_clear_images['command'] if hasattr(app, "col_btn_clear_images") else app.clear_images
    def validated_clear_images():
        original_clear_images_cmd()

    app.col_btn_clear_images = ttk.Button(
        app.controls_frame,
        text=tr("btn_clear_images"),
        command=validated_clear_images
    )
    app.col_btn_clear_images.pack(pady=5)
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
    app.col_btn_clear_csv.pack(pady=5)
    disable_space_activation(app.col_btn_clear_csv)

    app.col_section_reset = ttk.Label(app.controls_frame, text=tr("section_reset"))
    app.col_section_reset.pack(pady=(10, 0))

    app.col_btn_reset_defaults = ttk.Button(
        app.controls_frame,
        text=tr("btn_reset_defaults"),
        command=app.reset_to_defaults
    )
    app.col_btn_reset_defaults.pack(pady=5)
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
    app.col_btn_quit.pack(pady=5)
    disable_space_activation(app.col_btn_quit)

    app.col_section_img = ttk.Label(app.controls_frame, text=tr("section_img_settings"))
    app.col_section_img.pack(pady=(10, 5))

    app.col_lbl_brightness = ttk.Label(app.controls_frame, text=tr("lbl_brightness"))
    app.col_lbl_brightness.pack()
    app.brightness_var = tk.IntVar(value=0)
    app.brightness_value_label = ttk.Label(app.controls_frame, text="0")
    app.brightness_value_label.pack()
    app.brightness_scale = ttk.Scale(
        app.controls_frame,
        from_=-100,
        to=100,
        variable=app.brightness_var,
        orient=tk.HORIZONTAL,
        command=lambda e: app.update_scale_label(app.brightness_var, app.brightness_value_label)
    )
    app.brightness_scale.pack(padx=5)

    app.col_lbl_contrast = ttk.Label(app.controls_frame, text=tr("lbl_contrast"))
    app.col_lbl_contrast.pack()
    app.contrast_var = tk.IntVar(value=100)
    app.contrast_value_label = ttk.Label(app.controls_frame, text="100")
    app.contrast_value_label.pack()
    app.contrast_scale = ttk.Scale(
        app.controls_frame,
        from_=0,
        to=200,
        variable=app.contrast_var,
        orient=tk.HORIZONTAL,
        command=lambda e: app.update_scale_label(app.contrast_var, app.contrast_value_label)
    )
    app.contrast_scale.pack(padx=5)

    app.col_lbl_gamma = ttk.Label(app.controls_frame, text=tr("lbl_gamma"))
    app.col_lbl_gamma.pack()
    app.gamma_var = tk.IntVar(value=100)
    app.gamma_value_label = ttk.Label(app.controls_frame, text="100")
    app.gamma_value_label.pack()
    app.gamma_scale = ttk.Scale(
        app.controls_frame,
        from_=1,
        to=300,
        variable=app.gamma_var,
        orient=tk.HORIZONTAL,
        command=lambda e: app.update_scale_label(app.gamma_var, app.gamma_value_label)
    )
    app.gamma_scale.pack(padx=5)

    app.col_lbl_color_shift = ttk.Label(app.controls_frame, text=tr("lbl_color_shift"))
    app.col_lbl_color_shift.pack()
    frame_rgb = ttk.Frame(app.controls_frame)
    frame_rgb.pack(padx=5, pady=5, fill=tk.X)

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
        length=120,
        command=lambda e: app.update_scale_label(app.r_var, app.r_value_label)
    )
    r_scale.grid(row=0, column=1, padx=5)

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
        length=120,
        command=lambda e: app.update_scale_label(app.g_var, app.g_value_label)
    )
    g_scale.grid(row=1, column=1, padx=5)

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
        length=120,
        command=lambda e: app.update_scale_label(app.b_var, app.b_value_label)
    )
    b_scale.grid(row=2, column=1, padx=5)


def _create_mediapipe_frame(app):
    app.mp_section_title = ttk.Label(app.mp_frame, text=tr("section_mediapipe"))
    app.mp_section_title.pack(pady=(10, 5))

    app.mp_chk_static = ttk.Checkbutton(
        app.mp_frame,
        text=tr("chk_static_img_mode"),
        variable=app.static_image_mode_var
    )
    app.mp_chk_static.pack(padx=5, pady=2)

    app.mp_lbl_max_hands = ttk.Label(app.mp_frame, text=tr("lbl_max_num_hands"))
    app.mp_lbl_max_hands.pack()
    app.max_hands_label = ttk.Label(app.mp_frame, text=str(app.max_num_hands_var.get()))
    app.max_hands_label.pack()
    app.max_num_hands_scale = ttk.Scale(
        app.mp_frame,
        from_=1,
        to=4,
        variable=app.max_num_hands_var,
        orient=tk.HORIZONTAL,
        command=lambda e: app.update_scale_label(app.max_num_hands_var, app.max_hands_label)
    )
    app.max_num_hands_scale.pack(padx=5, pady=(0, 5))

    app.mp_lbl_model_complexity = ttk.Label(app.mp_frame, text=tr("lbl_model_complexity"))
    app.mp_lbl_model_complexity.pack()
    app.model_complexity_label = ttk.Label(app.mp_frame, text=str(app.model_complexity_var.get()))
    app.model_complexity_label.pack()
    app.model_complexity_scale = ttk.Scale(
        app.mp_frame,
        from_=0,
        to=2,
        variable=app.model_complexity_var,
        orient=tk.HORIZONTAL,
        command=lambda e: app.update_scale_label(app.model_complexity_var, app.model_complexity_label)
    )
    app.model_complexity_scale.pack(padx=5, pady=(0, 5))

    app.mp_lbl_min_det = ttk.Label(app.mp_frame, text=tr("lbl_min_det_conf"))
    app.mp_lbl_min_det.pack()
    app.min_detection_label = ttk.Label(app.mp_frame, text=str(app.min_detection_confidence_var.get()))
    app.min_detection_label.pack()
    app.min_detection_scale = ttk.Scale(
        app.mp_frame,
        from_=0,
        to=100,
        variable=app.min_detection_confidence_var,
        orient=tk.HORIZONTAL,
        command=lambda e: app.update_scale_label(app.min_detection_confidence_var, app.min_detection_label)
    )
    app.min_detection_scale.pack(padx=5, pady=(0, 5))

    app.mp_lbl_min_track = ttk.Label(app.mp_frame, text=tr("lbl_min_track_conf"))
    app.mp_lbl_min_track.pack()
    app.min_tracking_label = ttk.Label(app.mp_frame, text=str(app.min_tracking_confidence_var.get()))
    app.min_tracking_label.pack()
    app.min_tracking_scale = ttk.Scale(
        app.mp_frame,
        from_=0,
        to=100,
        variable=app.min_tracking_confidence_var,
        orient=tk.HORIZONTAL,
        command=lambda e: app.update_scale_label(app.min_tracking_confidence_var, app.min_tracking_label)
    )
    app.min_tracking_scale.pack(padx=5, pady=(0, 5))

    app.mp_btn_apply = ttk.Button(
        app.mp_frame,
        text=tr("btn_apply_mp"),
        command=app.update_mediapipe_settings
    )
    app.mp_btn_apply.pack(padx=5, pady=5)
    disable_space_activation(app.mp_btn_apply)

    app.mp_chk_overlays = ttk.Checkbutton(
        app.mp_frame,
        text=tr("chk_show_overlays"),
        variable=app.show_overlays_var
    )
    app.mp_chk_overlays.pack(padx=5, pady=5)

    app.mp_file_frame = ttk.LabelFrame(app.mp_frame, text=tr("frame_file_labels"))
    app.mp_file_frame.pack(fill=tk.X, pady=(10, 0))

    app.mp_lbl_csv = ttk.Label(app.mp_file_frame, text=tr("lbl_csv_file"))
    app.mp_lbl_csv.grid(row=0, column=0, sticky=tk.W, padx=2, pady=2)
    ttk.Entry(app.mp_file_frame, textvariable=app.csv_file_var, width=30).grid(
        row=0, column=1, sticky=tk.W, padx=2, pady=2
    )

    app.mp_lbl_model = ttk.Label(app.mp_file_frame, text=tr("lbl_model_file"))
    app.mp_lbl_model.grid(row=1, column=0, sticky=tk.W, padx=2, pady=2)
    ttk.Entry(app.mp_file_frame, textvariable=app.model_file_var, width=30).grid(
        row=1, column=1, sticky=tk.W, padx=2, pady=2
    )

    app.mp_lbl_scaler = ttk.Label(app.mp_file_frame, text=tr("lbl_scaler_file"))
    app.mp_lbl_scaler.grid(row=2, column=0, sticky=tk.W, padx=2, pady=2)
    ttk.Entry(app.mp_file_frame, textvariable=app.scaler_file_var, width=30).grid(
        row=2, column=1, sticky=tk.W, padx=2, pady=2
    )