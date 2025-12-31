from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox

import pandas as pd

from locales import tr

from ctk_app.common import _open_path_cross_platform, _parse_camera_index

class SettingsView(ctk.CTkFrame):
    def __init__(self, master, app: App):
        super().__init__(master)
        self.app = app

        self._content_frame: ctk.CTkFrame | None = None

        self._initial_theme_light = str(self.app.color_theme_light.get() or "")
        self._initial_theme_dark = str(self.app.color_theme_dark.get() or "")

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        frame = ctk.CTkFrame(self)
        frame.grid(row=0, column=0, sticky="nsew", padx=18, pady=18)
        frame.grid_rowconfigure(1, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            frame,
            text=tr("tab_settings"),
            font=ctk.CTkFont(size=18, weight="bold"),
        ).grid(row=0, column=0, sticky="w", padx=14, pady=(14, 8))

        scroll = ctk.CTkScrollableFrame(frame)
        scroll.grid(row=1, column=0, sticky="nsew", padx=14, pady=(0, 14))
        scroll.grid_columnconfigure(0, weight=1)
        outer = ctk.CTkFrame(scroll, fg_color="transparent")
        outer.pack(fill="x", pady=10)

        content = ctk.CTkFrame(outer)
        content.pack(anchor="n")
        self._content_frame = content

        content.grid_rowconfigure(0, weight=1)
        content.grid_columnconfigure(0, weight=1)
        content.grid_columnconfigure(1, weight=1)

        left_col = ctk.CTkFrame(content, fg_color="transparent")
        right_col = ctk.CTkFrame(content, fg_color="transparent")
        left_col.grid(row=0, column=0, sticky="n", padx=(0, 10))
        right_col.grid(row=0, column=1, sticky="n", padx=(10, 0))

        left_col.grid_columnconfigure(0, weight=1)
        right_col.grid_columnconfigure(0, weight=1)

        self._lang_display_to_code = {
            "Polski  🇵🇱": "pl",
            "English 🇬🇧": "en",
        }
        self._lang_code_to_display = {v: k for k, v in self._lang_display_to_code.items()}
        self.det_interval_var = tk.StringVar(value=str(app.det_interval_ms.get()))
        self.det_threshold_var = tk.StringVar(value=str(app.det_threshold))
        self._cam_display_var = tk.StringVar(value=f"Cam {app.camera_index_var.get()}")

        self._accent_palette = {
            "blue": ("#3B8ED0", "#1F6AA5"),
            "green": ("#2CC985", "#2FA572"),
            "dark-blue": ("#3a7ebf", "#1f538d"),
            "purple": ("#8E44AD", "#6C3483"),
            "red": ("#E74C3C", "#B03A2E"),
            "orange": ("#E67E22", "#B95E00"),
            "teal": ("#1ABC9C", "#148F77"),
            "pink": ("#E84393", "#B53471"),
            "yellow": ("#F1C40F", "#B7950B"),
            "lime": ("#7DCE13", "#4C8B0C"),
            "cyan": ("#00BCD4", "#00838F"),
            "indigo": ("#3F51B5", "#283593"),
            "gray": ("#7F8C8D", "#566573"),
            "amber": ("#FFB300", "#C58A00"),
        }
        self._accent_label = {
            "blue": tr("theme_blue"),
            "green": tr("theme_green"),
            "dark-blue": tr("theme_dark_blue"),
            "purple": tr("theme_purple"),
            "red": tr("theme_red"),
            "orange": tr("theme_orange"),
            "teal": tr("theme_teal"),
            "pink": tr("theme_pink"),
            "yellow": tr("theme_yellow"),
            "lime": tr("theme_lime"),
            "cyan": tr("theme_cyan"),
            "indigo": tr("theme_indigo"),
            "gray": tr("theme_gray"),
            "amber": tr("theme_amber"),
        }
        self._accent_display_to_key = {self._accent_label.get(k, k): k for k in self._accent_palette.keys()}
        self._accent_key_to_display = {v: k for k, v in self._accent_display_to_key.items()}
        accent_values = list(self._accent_display_to_key.keys())

        general = self._card_stack(left_col, tr("section_general"))
        paths = self._card_stack(left_col, tr("section_paths"))
        mp = self._card_stack(left_col, tr("section_mediapipe"))

        stats = self._card_stack(right_col, tr("section_stats"))
        quick = self._card_stack(right_col, tr("section_quick_open"))
        danger = self._card_stack(right_col, tr("section_danger_zone"))
        author = self._card_stack(right_col, tr("section_author"))

        form = ctk.CTkFrame(general, fg_color="transparent")
        form.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 12))
        form.grid_columnconfigure(1, weight=1)

        _CONTROL_WIDTH = 260
        _SWATCH_WIDTH = 18
        _SWATCH_GAP = 10
        _LANG_WIDTH = _CONTROL_WIDTH + _SWATCH_WIDTH + _SWATCH_GAP

        self.lang_menu = ctk.CTkOptionMenu(
            form,
            values=list(self._lang_display_to_code.keys()),
            command=self._on_language_changed,
            width=_LANG_WIDTH,
        )
        self._form_row_widget(form, 0, tr("language_label").rstrip(":"), self.lang_menu)
        self.lang_menu.set(self._lang_code_to_display.get(self.app.language_code.get(), "English 🇬🇧"))

        def _swatch_color_for(key: str, idx: int) -> str:
            return self._accent_palette.get(key, ("#3B8ED0", "#1F6AA5"))[idx]

        def _theme_picker(row_idx: int, title: str, key_var: tk.StringVar, swatch_index: int):
            inner = ctk.CTkFrame(form, fg_color="transparent")
            inner.grid_columnconfigure(2, weight=1)

            swatch = ctk.CTkFrame(
                inner,
                width=18,
                height=18,
                corner_radius=5,
                fg_color=_swatch_color_for(str(key_var.get() or "blue"), swatch_index),
            )
            swatch.grid(row=0, column=0, sticky="w")
            swatch.grid_propagate(False)

            display_var = tk.StringVar(
                value=self._accent_key_to_display.get(str(key_var.get() or ""), self._accent_key_to_display.get("blue", "blue"))
            )

            def _on_changed(display_value: str):
                k = self._accent_display_to_key.get(str(display_value), "blue")
                key_var.set(k)
                try:
                    swatch.configure(fg_color=_swatch_color_for(k, swatch_index))
                except Exception:
                    pass

            menu = ctk.CTkOptionMenu(inner, values=accent_values, variable=display_var, command=_on_changed, width=_CONTROL_WIDTH)
            menu.grid(row=0, column=1, sticky="w", padx=(10, 0))

            self._form_row_widget(form, row_idx, title.rstrip(":"), inner)

        _theme_picker(1, tr("lbl_theme_light"), self.app.color_theme_light, 0)
        _theme_picker(2, tr("lbl_theme_dark"), self.app.color_theme_dark, 1)

        self.overlay_switch = ctk.CTkSwitch(form, text=tr("chk_overlay"))
        self._form_row_widget(form, 3, "", self.overlay_switch)
        if app.show_overlays:
            self.overlay_switch.select()
        else:
            self.overlay_switch.deselect()

        reset_btn = ctk.CTkButton(form, text=tr("btn_reset_defaults"), width=_CONTROL_WIDTH, command=app.reset_to_defaults)
        self._form_row_widget(form, 4, "", reset_btn)

        save_btn = ctk.CTkButton(form, text=tr("btn_save_settings"), width=_CONTROL_WIDTH, command=self._apply_all_settings)
        self._form_row_widget(form, 5, "", save_btn)

        pform = ctk.CTkFrame(paths, fg_color="transparent")
        pform.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 14))
        pform.grid_columnconfigure(1, weight=1)
        self._form_row_entry(pform, 0, tr("lbl_csv_file"), self.app.csv_file_var, width=520)
        self._form_row_entry(pform, 1, tr("lbl_model_file"), self.app.model_file_var, width=520)
        self._form_row_entry(pform, 2, tr("lbl_scaler_file"), self.app.scaler_file_var, width=520)

        mpform = ctk.CTkFrame(mp, fg_color="transparent")
        mpform.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 14))
        mpform.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(mpform, text=tr("section_camera"), font=ctk.CTkFont(size=13, weight="bold"), anchor="w").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 6)
        )
        preset_values = [tr("camera_preset_small"), tr("camera_preset_medium"), tr("camera_preset_large")]
        self.camera_size_menu = ctk.CTkOptionMenu(mpform, values=preset_values, variable=self.app.camera_size_preset, width=260)
        self._form_row_widget(mpform, 1, tr("lbl_camera_size").rstrip(":"), self.camera_size_menu)

        cam_values = [f"Cam {i}" for i in app.available_cameras] if app.available_cameras else ["Cam 0"]
        self.cam_menu = ctk.CTkOptionMenu(mpform, values=cam_values, variable=self._cam_display_var, width=260)
        self._form_row_widget(mpform, 2, tr("lbl_camera").rstrip(":"), self.cam_menu)

        ctk.CTkButton(mpform, text=f"↻ {tr('btn_restart_camera')}", width=260, command=self._restart_camera).grid(
            row=3, column=1, sticky="w", pady=4
        )

        ctk.CTkLabel(mpform, text=tr("section_detection"), font=ctk.CTkFont(size=13, weight="bold"), anchor="w").grid(
            row=4, column=0, columnspan=2, sticky="w", pady=(6, 6)
        )
        self._form_row_entry(mpform, 5, tr("lbl_interval").rstrip(":"), self.det_interval_var, width=120)
        self._form_row_entry(mpform, 6, tr("lbl_threshold").rstrip(":"), self.det_threshold_var, width=120)
        self.enter_mode_switch = ctk.CTkSwitch(mpform, text=tr("chk_enter_mode"), variable=app.enter_mode)
        self.enter_mode_switch.grid(row=7, column=1, sticky="w", pady=4)

        self.static_switch = ctk.CTkSwitch(mpform, text=tr("chk_static_img_mode"), variable=app.static_image_mode)
        self.static_switch.grid(row=8, column=1, sticky="w", pady=4)
        self._form_row_entry(mpform, 9, tr("lbl_max_num_hands"), app.max_num_hands, width=120)
        self._form_row_entry(mpform, 10, tr("lbl_model_complexity"), app.model_complexity, width=120)
        self._form_row_entry(mpform, 11, tr("lbl_min_det_conf"), app.min_detection_confidence, width=120)
        self._form_row_entry(mpform, 12, tr("lbl_min_track_conf"), app.min_tracking_confidence, width=120)

        ctk.CTkButton(mpform, text=tr("btn_apply_mp"), width=260, command=self._apply_mediapipe).grid(
            row=13, column=1, sticky="w", pady=(10, 2)
        )

        qgrid_wrap = ctk.CTkFrame(quick, fg_color="transparent")
        qgrid_wrap.grid(row=1, column=0, sticky="ew", padx=14, pady=(4, 14))
        qgrid_wrap.grid_columnconfigure(0, weight=1)
        qgrid_wrap.grid_columnconfigure(2, weight=1)
        qgrid = ctk.CTkFrame(qgrid_wrap, fg_color="transparent")
        qgrid.grid(row=0, column=1)
        btn_h = 32
        ctk.CTkButton(qgrid, text=tr("btn_open_images_folder"), width=210, height=btn_h, command=self._open_images_folder).grid(
            row=0, column=0, padx=(0, 10), pady=(0, 10)
        )
        ctk.CTkButton(qgrid, text=tr("btn_open_models_folder"), width=210, height=btn_h, command=self._open_models_folder).grid(
            row=0, column=1, padx=(10, 0), pady=(0, 10)
        )
        ctk.CTkButton(qgrid, text=tr("btn_open_csv_file"), width=440, height=btn_h, command=self._open_csv_file).grid(
            row=1, column=0, columnspan=2
        )

        self.stats_samples = tk.StringVar(value="—")
        self.stats_classes = tk.StringVar(value="—")
        self.stats_images = tk.StringVar(value="—")
        self.stats_csv_size = tk.StringVar(value="—")
        self.stats_csv_mtime = tk.StringVar(value="—")
        self.stats_model_size = tk.StringVar(value="—")
        sform = ctk.CTkFrame(stats, fg_color="transparent")
        sform.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 14))
        sform.grid_columnconfigure(1, weight=1)
        self._stat_grid_line(sform, 0, tr("stat_samples"), self.stats_samples)
        self._stat_grid_line(sform, 1, tr("stat_classes"), self.stats_classes)
        self._stat_grid_line(sform, 2, tr("stat_images"), self.stats_images)
        self._stat_grid_line(sform, 3, tr("stat_csv_size"), self.stats_csv_size)
        self._stat_grid_line(sform, 4, tr("stat_csv_mtime"), self.stats_csv_mtime)
        self._stat_grid_line(sform, 5, tr("stat_model_size"), self.stats_model_size)

        dz = ctk.CTkFrame(danger, fg_color="transparent")
        dz.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 14))
        dz.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(dz, text=tr("danger_zone_desc"), justify="left", wraplength=360, anchor="w").grid(
            row=0, column=0, sticky="w", pady=(0, 10)
        )
        ctk.CTkButton(dz, text=tr("btn_delete_all_data"), width=240, command=self._delete_all_data).grid(row=1, column=0, sticky="w")

        about = ctk.CTkFrame(author, fg_color="transparent")
        about.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 14))
        ctk.CTkLabel(about, text=tr("author_name"), font=ctk.CTkFont(size=13, weight="bold"), anchor="w", justify="left").grid(
            row=0, column=0, sticky="w", pady=(0, 4)
        )
        ctk.CTkLabel(about, text=tr("author_info"), anchor="w", justify="left", wraplength=360).grid(
            row=1, column=0, sticky="w"
        )

        self._refresh_stats()

    def on_show(self) -> None:
        try:
            self.lang_menu.set(self._lang_code_to_display.get(self.app.language_code.get(), "English 🇬🇧"))
        except Exception:
            pass
        try:
            self.det_interval_var.set(str(self.app.det_interval_ms.get()))
            self.det_threshold_var.set(str(self.app.det_threshold))
        except Exception:
            pass
        try:
            self._cam_display_var.set(f"Cam {self.app.camera_index_var.get()}")
        except Exception:
            pass

        self._initial_theme_light = str(self.app.color_theme_light.get() or "")
        self._initial_theme_dark = str(self.app.color_theme_dark.get() or "")
        self._refresh_stats()

    def _card_stack(self, master, title: str) -> ctk.CTkFrame:
        card = ctk.CTkFrame(master)
        card.pack(fill="x", pady=10)
        card.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(card, text=title, font=ctk.CTkFont(size=14, weight="bold"), anchor="w").grid(
            row=0, column=0, sticky="ew", padx=14, pady=(12, 8)
        )
        return card

    def _form_row_widget(self, master, row: int, label: str, widget) -> None:
        ctk.CTkLabel(master, text=label, width=190, anchor="w").grid(row=row, column=0, sticky="w", pady=4)
        widget.grid(row=row, column=1, sticky="w", pady=4)

    def _form_row_entry(self, master, row: int, label: str, var, width: int = 300) -> None:
        ctk.CTkLabel(master, text=label, width=190, anchor="w").grid(row=row, column=0, sticky="w", pady=4)
        ctk.CTkEntry(master, textvariable=var, width=width).grid(row=row, column=1, sticky="w", pady=4)

    def _stat_grid_line(self, master, row: int, label: str, var: tk.StringVar) -> None:
        ctk.CTkLabel(master, text=label, width=160, anchor="w").grid(row=row, column=0, sticky="w", pady=3)
        ctk.CTkLabel(master, textvariable=var, anchor="w").grid(row=row, column=1, sticky="w", pady=3)

    def _apply_all_settings(self) -> None:
        try:
            interval_val = int((self.det_interval_var.get() or "").strip())
            threshold_val = float((self.det_threshold_var.get() or "").strip())
        except ValueError:
            messagebox.showerror(tr("dlg_error"), tr("err_invalid_numbers"))
            return
        if interval_val <= 0 or threshold_val <= 0 or threshold_val > 1:
            messagebox.showwarning(tr("dlg_warning"), tr("warn_invalid_range"))
            return

        try:
            self.app.det_interval_ms.set(interval_val)
            self.app.det_threshold = threshold_val
        except Exception:
            pass

        try:
            self.app._prepare_directories_and_csv()
        except Exception:
            pass

        try:
            self.app.apply_camera_layout_preset()
        except Exception:
            pass
        try:
            idx = _parse_camera_index(str(self._cam_display_var.get() or ""))
            if int(idx) != int(self.app.camera_index_var.get()):
                self.app.open_camera(int(idx))
        except Exception:
            pass

        try:
            self.app._init_mediapipe_hands()
        except Exception:
            pass
        try:
            self.app.reset_detector()
        except Exception:
            pass

        try:
            self.app.show_overlays = bool(self.overlay_switch.get())
        except Exception:
            pass

        try:
            self.app.save_settings_to_file()
        except Exception:
            pass

        theme_changed = (
            str(self.app.color_theme_light.get() or "") != self._initial_theme_light
            or str(self.app.color_theme_dark.get() or "") != self._initial_theme_dark
        )
        if theme_changed:
            self._apply_theme()
            return

        try:
            self.app.log(tr("log_settings_applied"))
        except Exception:
            pass
        try:
            self._refresh_stats()
        except Exception:
            pass

    def _on_language_changed(self, value: str) -> None:
        code = self._lang_display_to_code.get(value, "en")
        self.after(10, lambda: self.app.set_language(code))

    def _section_label(self, master, text: str) -> None:
        ctk.CTkLabel(master, text=text, font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", pady=(10, 6))

    def _stat_line(self, master, label: str, var: tk.StringVar) -> None:
        row = ctk.CTkFrame(master, fg_color="transparent")
        row.pack(fill="x", pady=(0, 4))
        ctk.CTkLabel(row, text=label, width=180, anchor="w").pack(side="left")
        ctk.CTkLabel(row, textvariable=var, anchor="w").pack(side="left", fill="x", expand=True)

    def _refresh_stats(self) -> None:
        csv_path = Path(str(self.app.csv_file_var.get() or "").strip())
        if not csv_path:
            return
        if not csv_path.is_absolute():
            csv_path = (self.app.base_dir / csv_path).resolve()

        samples = 0
        classes = 0
        try:
            if csv_path.exists():
                df = pd.read_csv(str(csv_path))
                samples = int(len(df.index))
                if "label" in df.columns:
                    classes = int(df["label"].nunique())
        except Exception:
            pass

        images_count = 0
        try:
            images_dir = Path(str(self.app.images_dir or "images"))
            if not images_dir.is_absolute():
                images_dir = (self.app.base_dir / images_dir).resolve()
            if images_dir.exists():
                for root, _dirs, files in os.walk(str(images_dir)):
                    for name in files:
                        ext = os.path.splitext(name)[1].lower()
                        if ext in (".jpg", ".jpeg", ".png", ".bmp"):
                            images_count += 1
        except Exception:
            pass

        size_str = "—"
        mtime_str = "—"
        try:
            if csv_path.exists():
                size_kb = os.path.getsize(str(csv_path)) / 1024.0
                size_str = f"{size_kb:.1f} KB"
                mtime_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(str(csv_path))))
        except Exception:
            pass

        model_size_str = "—"
        try:
            model_path = Path(str(self.app.model_file_var.get() or "").strip())
            if model_path and not model_path.is_absolute():
                model_path = (self.app.base_dir / model_path).resolve()
            if model_path.exists():
                mb = os.path.getsize(str(model_path)) / (1024.0 * 1024.0)
                model_size_str = f"{mb:.2f} MB"
        except Exception:
            pass

        self.stats_samples.set(str(samples))
        self.stats_classes.set(str(classes))
        self.stats_images.set(str(images_count))
        self.stats_csv_size.set(size_str)
        self.stats_csv_mtime.set(mtime_str)
        self.stats_model_size.set(model_size_str)

    def _kv_entry(self, master, label: str, var) -> None:
        row = ctk.CTkFrame(master, fg_color="transparent")
        row.pack(fill="x", pady=(0, 6))
        ctk.CTkLabel(row, text=label, width=160, anchor="w").pack(side="left")
        ctk.CTkEntry(row, textvariable=var).pack(side="left", fill="x", expand=True)

    def _apply_paths(self) -> None:
        self.app._prepare_directories_and_csv()
        self.app.reset_detector()
        self.app.log(tr("log_paths_applied"))
        self.app.save_settings_to_file()

    def _apply_theme(self) -> None:
        def _do_apply():
            self.app._apply_ui_theme()
            self.app.save_settings_to_file()
            self.app.rebuild_ui()

        self.after(10, _do_apply)

    def _open_path(self, path: Path, select_in_explorer: bool = False) -> None:
        try:
            _open_path_cross_platform(path, select_in_file_manager=select_in_explorer)
        except Exception as exc:
            try:
                self.app.log(tr("err_open_path", err=str(exc)))
            except Exception:
                pass

    def _open_images_folder(self) -> None:
        images_dir = Path(str(self.app.images_dir or "images").strip())
        if not images_dir.is_absolute():
            images_dir = (self.app.base_dir / images_dir).resolve()
        self._open_path(images_dir, select_in_explorer=False)

    def _open_models_folder(self) -> None:
        model_path = Path(str(self.app.model_file_var.get() or "").strip())
        if model_path and not model_path.is_absolute():
            model_path = (self.app.base_dir / model_path).resolve()
        models_dir = model_path.parent if model_path else (self.app.base_dir / "models")
        if not models_dir.is_absolute():
            models_dir = (self.app.base_dir / models_dir).resolve()
        self._open_path(models_dir, select_in_explorer=False)

    def _open_csv_file(self) -> None:
        csv_path = Path(str(self.app.csv_file_var.get() or "").strip())
        if not csv_path:
            return
        if not csv_path.is_absolute():
            csv_path = (self.app.base_dir / csv_path).resolve()
        if csv_path.exists():
            self._open_path(csv_path, select_in_explorer=False)
        else:
            self._open_path(csv_path.parent, select_in_explorer=False)

    def _resolve_path(self, p: str) -> Path:
        path = Path((p or "").strip())
        if not path.is_absolute():
            path = (self.app.base_dir / path).resolve()
        return path

    def _delete_all_data(self) -> None:
        csv_path = self._resolve_path(self.app.csv_file_var.get())
        model_path = self._resolve_path(self.app.model_file_var.get())
        scaler_path = self._resolve_path(self.app.scaler_file_var.get())

        images_dir = Path(str(self.app.images_dir or "images").strip() or "images")
        if not images_dir.is_absolute():
            images_dir = (self.app.base_dir / images_dir).resolve()

        messagebox.showwarning(
            tr("dlg_warning"),
            tr(
                "warn_delete_all_data",
                csv=str(csv_path),
                model=str(model_path),
                scaler=str(scaler_path),
                images=str(images_dir),
            ),
        )

        ok = messagebox.askyesno(
            tr("dlg_confirm"),
            tr(
                "confirm_delete_all_data",
                csv=str(csv_path),
                model=str(model_path),
                scaler=str(scaler_path),
                images=str(images_dir),
            ),
        )
        if not ok:
            self.app.log(tr("log_delete_all_cancelled"))
            return

        errors: list[str] = []

        def _try_remove_file(path: Path):
            try:
                if path.exists() and path.is_file():
                    path.unlink()
            except Exception as exc:
                errors.append(f"{path}: {exc}")

        _try_remove_file(csv_path)
        _try_remove_file(model_path)
        _try_remove_file(scaler_path)

        try:
            if images_dir.exists() and images_dir.is_dir():
                shutil.rmtree(str(images_dir))
        except Exception as exc:
            errors.append(f"{images_dir}: {exc}")

        if errors:
            messagebox.showerror(tr("dlg_error"), tr("err_delete_all_failed", err="\n".join(errors)))
        else:
            self.app.reset_detector()
            self.app.log(tr("log_delete_all_done"))
        try:
            self._refresh_stats()
        except Exception:
            pass

    def _on_camera_changed(self, value: str) -> None:
        idx = _parse_camera_index(value)
        self.app.open_camera(idx)

    def _on_camera_size_changed(self, _value: str) -> None:
        self.app.apply_camera_layout_preset()
        self.app.log(tr("log_camera_size_preset", preset=self.app.camera_size_preset.get()))

    def _restart_camera(self) -> None:
        self.app.restart_camera(int(self.app.camera_index_var.get()))

    def _apply_detection(self) -> None:
        try:
            interval_val = int((self.det_interval_var.get() or "").strip())
            threshold_val = float((self.det_threshold_var.get() or "").strip())
        except ValueError:
            messagebox.showerror(tr("dlg_error"), tr("err_invalid_numbers"))
            return
        if interval_val <= 0 or threshold_val <= 0 or threshold_val > 1:
            messagebox.showwarning(tr("dlg_warning"), tr("warn_invalid_range"))
            return
        self.app.det_interval_ms.set(interval_val)
        self.app.det_threshold = threshold_val
        self.app.log(tr("log_detection_applied"))
        self.app.save_settings_to_file()

    def _apply_mediapipe(self) -> None:
        self.app._init_mediapipe_hands()
        self.app.reset_detector()
        self.app.log(tr("log_mediapipe_applied"))
        self.app.save_settings_to_file()

    def _toggle_overlays(self) -> None:
        self.app.show_overlays = bool(self.overlay_switch.get())
        self.app.log(tr("log_overlays", state=("ON" if self.app.show_overlays else "OFF")))


