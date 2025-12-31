from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import threading
import time
import io
import sys
import ctypes
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import customtkinter as ctk

import cv2
import joblib
import numpy as np
import pandas as pd
import tkinter as tk
import tkinter.font as tkfont
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import importlib

from locales import tr, load

from ctk_app.common import (
    MAX_HANDS_SUPPORTED,
    _extract_multi_hand_features,
    _multi_hand_header,
    _theme_overlay_bg_hex,
    _theme_value,
)
from ctk_app.views import (
    DataCollectionView,
    DetectionView,
    InstructionsView,
    SettingsView,
    TextPracticeView,
    TrainingView,
)
from ctk_app.detector import HandSignDetector

@dataclass
class TrainingParams:
    epochs: int = 30
    batch_size: int = 32
    test_split: float = 0.2


class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.language_code = tk.StringVar(value="pl")
        try:
            load(self.language_code.get())
        except Exception:
            pass

        self.title(tr("app_title_ctk"))
        self.minsize(1100, 750)

        self.appearance_mode = tk.StringVar(value="Dark")
        self.color_theme_light = tk.StringVar(value="blue")
        self.color_theme_dark = tk.StringVar(value="dark-blue")

        try:
            self._maximize_window()
        except Exception:
            pass
        self.after(0, self._maximize_window)

        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)

        self.active_view: str | None = None
        self.training_params = TrainingParams()

        self.views: dict[str, ctk.CTkFrame] = {}

        self._log_buffer: list[str] = []

        self.deps_ready: bool = False
        self._startup_overlay: ctk.CTkFrame | None = None
        self._startup_label: ctk.CTkLabel | None = None

        self.root = self

        base_dir = Path(__file__).resolve().parents[1]
        self.base_dir = base_dir
        self.settings_file = (base_dir / "other" / "settings.json").resolve()
        self.csv_path = str((base_dir / "data" / "data.csv").resolve())
        self.model_path = str((base_dir / "models" / "model.h5").resolve())
        self.scaler_path = str((base_dir / "other" / "scaler.pkl").resolve())

        self.csv_file_var = tk.StringVar(value=str(Path("data") / "data.csv"))
        self.model_file_var = tk.StringVar(value=str(Path("models") / "model.h5"))
        self.scaler_file_var = tk.StringVar(value=str(Path("other") / "scaler.pkl"))

        self.test_size_var = tk.DoubleVar(value=0.2)
        self.random_state_var = tk.IntVar(value=42)
        self.epochs_var = tk.IntVar(value=self.training_params.epochs)
        self.batch_size_var = tk.IntVar(value=self.training_params.batch_size)
        self.patience_var = tk.IntVar(value=6)
        self.val_split_var = tk.DoubleVar(value=0.1)
        self.monitor_var = tk.StringVar(value="val_loss")
        self.progress_var = tk.DoubleVar(value=0.0)

        self.images_dir = "images"
        self.current_label: Optional[str] = None
        self.flip_horizontal: bool = False
        self.flip_vertical: bool = False
        self.show_overlays: bool = True
        self.last_frame: Optional[np.ndarray] = None
        self.last_frame_raw: Optional[np.ndarray] = None

        self.collect_self_timer_enabled = tk.BooleanVar(value=False)
        self.collect_self_timer_seconds = tk.IntVar(value=3)
        self.collect_self_timer_loop_enabled = tk.BooleanVar(value=False)

        self.collect_batch_save_overlay = tk.BooleanVar(value=True)

        self._collect_next_idx_label: str | None = None
        self._collect_next_idx: int | None = None

        self._last_saved_sample: dict | None = None

        self.brightness = tk.IntVar(value=0)
        self.contrast = tk.IntVar(value=100)
        self.gamma = tk.IntVar(value=100)
        self.shift_r = tk.IntVar(value=0)
        self.shift_g = tk.IntVar(value=0)
        self.shift_b = tk.IntVar(value=0)

        self.static_image_mode = tk.BooleanVar(value=False)
        self.max_num_hands = tk.IntVar(value=1)
        self.model_complexity = tk.IntVar(value=1)
        self.min_detection_confidence = tk.IntVar(value=50)
        self.min_tracking_confidence = tk.IntVar(value=50)
        self._mp_hands = None
        self._mp_drawing = None
        self._hands = None

        self.det_threshold: float = 0.7
        self.det_interval_ms = tk.IntVar(value=1000)
        self.enter_mode = tk.BooleanVar(value=False)

        self._load_settings_from_file()

        self._apply_ui_theme()

        self._cap: cv2.VideoCapture | None = None
        self._camera_after_id: str | None = None
        self._camera_index: int = 0
        self._camera_running: bool = False
        self._last_cam_consumer_log_ts: float = 0.0
        self._camera_last_consumer = None
        self._camera_last_interval_ms: int = 33
        self._camera_restart_in_progress: bool = False

        self._detector: HandSignDetector | None = None

        self.available_cameras = [0]
        self.camera_index_var = tk.IntVar(value=0)

        self.camera_size_preset = tk.StringVar(value=tr("camera_preset_medium"))

        self.console_visible = tk.BooleanVar(value=True)
        self.floating_console_btn: ctk.CTkButton | None = None

        self._prepare_directories_and_csv()

        self.after(0, self._maximize_window)
        self.after(120, self._maximize_window)
        self.after(350, self._maximize_window)

        self._show_startup_overlay(tr("startup_loading"))
        self.after(50, self._start_dependency_loader)

    def _show_startup_overlay(self, text: str) -> None:
        try:
            if self._startup_overlay is not None:
                return
            overlay = ctk.CTkFrame(self, corner_radius=0, fg_color=_theme_overlay_bg_hex())
            overlay.place(relx=0, rely=0, relwidth=1, relheight=1)
            overlay.lift()
            overlay.grid_propagate(False)

            label = ctk.CTkLabel(
                overlay,
                text=text,
                font=ctk.CTkFont(size=16, weight="bold"),
            )
            label.place(relx=0.5, rely=0.5, anchor="center")

            self._startup_overlay = overlay
            self._startup_label = label
        except Exception:
            pass

    def _set_startup_overlay_text(self, text: str) -> None:
        try:
            if self._startup_label is not None:
                self._startup_label.configure(text=text)
        except Exception:
            pass

    def _hide_startup_overlay(self) -> None:
        try:
            if self._startup_overlay is not None:
                self._startup_overlay.destroy()
        except Exception:
            pass
        self._startup_overlay = None
        self._startup_label = None

    def _start_dependency_loader(self) -> None:
        def _worker():
            try:
                self.after(0, lambda: self._set_startup_overlay_text(tr("startup_loading_tf")))
                globals()["tf"] = importlib.import_module("tensorflow")

                self.after(0, lambda: self._set_startup_overlay_text(tr("startup_loading_mp")))
                globals()["mp"] = importlib.import_module("mediapipe")

                self.after(0, lambda: self._set_startup_overlay_text(tr("startup_loading_training")))
                globals()["training"] = importlib.import_module("ctk_app.training_worker")

                self.after(0, self._on_startup_deps_ready)
            except Exception as exc:
                self.after(0, lambda: self._set_startup_overlay_text(tr("startup_loading_failed", err=str(exc))))

        threading.Thread(target=_worker, daemon=True).start()

    def _finish_startup_ui(self) -> None:
        self._build_sidebar()
        self._build_main_content()
        self._build_console()

        self._ensure_floating_console_button()
        if not bool(self.console_visible.get()):
            try:
                self.console_frame.grid_remove()
            except Exception:
                pass

        self.show_view("detekcja")
        self.log(tr("log_gui_started"))

        self.bind_all("<Return>", self._on_enter)
        self.bind_all("<space>", self._on_space)
        self.bind_all("<Control-l>", lambda _e: self.toggle_console())
        self.bind_all("<Control-L>", lambda _e: self.toggle_console())

        self._bind_keyboard_shortcuts()

    def _focus_is_text_input(self) -> bool:
        try:
            w = self.focus_get()
        except Exception:
            return False
        if w is None:
            return False

        try:
            if isinstance(w, (tk.Entry, tk.Text)):
                return True
        except Exception:
            pass

        try:
            cls = str(getattr(w, "winfo_class", lambda: "")() or "").lower()
            if any(k in cls for k in ("entry", "text", "spinbox", "combobox")):
                return True
        except Exception:
            pass
        return False

    def _bind_keyboard_shortcuts(self) -> None:
        if getattr(self, "_shortcuts_bound", False):
            return
        self._shortcuts_bound = True

        view_order = ["zbieranie", "detekcja", "trening", "tekst", "instrukcja", "ustawienia"]

        def _safe_call(fn):
            def _wrapped(event=None):
                try:
                    return fn(event)
                except Exception as exc:
                    try:
                        self.log(f"Shortcut error: {exc}")
                    except Exception:
                        pass
                    return "break"

            return _wrapped

        def _bind(seq: str, fn) -> None:
            try:
                self.bind_all(seq, _safe_call(fn), add="+")
            except Exception:
                pass

        def _go(view_key: str):
            def _f(_event=None):
                try:
                    self.show_view(view_key)
                except Exception:
                    pass
                return "break"

            return _f

        def _cycle(delta: int):
            def _f(_event=None):
                try:
                    cur = self.active_view or view_order[0]
                    idx = view_order.index(cur) if cur in view_order else 0
                    self.show_view(view_order[(idx + delta) % len(view_order)])
                except Exception:
                    pass
                return "break"

            return _f

        _bind("<F1>", _go("instrukcja"))
        _bind("<F2>", _go("zbieranie"))
        _bind("<F3>", _go("detekcja"))
        _bind("<F4>", _go("trening"))
        _bind("<F5>", _go("tekst"))
        _bind("<F6>", _go("ustawienia"))

        for i, key in enumerate(view_order, start=1):
            _bind(f"<Control-Key-{i}>", _go(key))
            _bind(f"<Control-{i}>", _go(key))

        _bind("<Control-Tab>", _cycle(+1))
        _bind("<Control-Shift-Tab>", _cycle(-1))

        _bind("<Control-q>", lambda _e: (self.on_app_close(), "break"))
        _bind("<Control-Q>", lambda _e: (self.on_app_close(), "break"))
        _bind("<Control-Shift-T>", lambda _e: (self._toggle_theme(), "break"))
        _bind("<Control-Shift-t>", lambda _e: (self._toggle_theme(), "break"))

        def _active_view_obj():
            try:
                return self.views.get(self.active_view or "")
            except Exception:
                return None

        def _toggle_switch(switch_widget) -> None:
            try:
                cur = int(switch_widget.get())
                if cur:
                    switch_widget.deselect()
                else:
                    switch_widget.select()
            except Exception:
                pass

        def _do_detection(action: str):
            v = _active_view_obj()
            if v is None or getattr(v, "_view_key", "") != "detekcja":
                return
            if action == "start" and hasattr(v, "_start_detection"):
                v._start_detection()
            elif action == "stop" and hasattr(v, "_stop_detection"):
                v._stop_detection()
            elif action == "clear" and hasattr(v, "_clear_buffer"):
                v._clear_buffer()
            elif action == "restart" and hasattr(v, "_restart_camera"):
                v._restart_camera()
            elif action == "flip_h" and hasattr(v, "_flip_h"):
                v._flip_h()
            elif action == "flip_v" and hasattr(v, "_flip_v"):
                v._flip_v()
            elif action == "enter_mode":
                try:
                    self.enter_mode.set(not bool(self.enter_mode.get()))
                except Exception:
                    pass

        def _do_collect(action: str):
            v = _active_view_obj()
            if v is None or getattr(v, "_view_key", "") != "zbieranie":
                return
            if action == "save":
                self.save_sample()
            elif action == "undo":
                self.undo_last_sample()
            elif action == "set_label" and hasattr(v, "_set_label"):
                v._set_label()
            elif action == "restart" and hasattr(v, "_restart_camera"):
                v._restart_camera()
            elif action == "timer_toggle" and hasattr(v, "_toggle_timer"):
                v._toggle_timer()
            elif action == "timer_enable" and hasattr(v, "_on_timer_toggle") and hasattr(v, "timer_switch"):
                try:
                    new_val = not bool(self.collect_self_timer_enabled.get())
                    self.collect_self_timer_enabled.set(new_val)
                    if new_val:
                        v.timer_switch.select()
                    else:
                        v.timer_switch.deselect()
                except Exception:
                    pass
                v._on_timer_toggle()
            elif action == "timer_loop" and hasattr(v, "timer_loop_chk"):
                try:
                    self.collect_self_timer_loop_enabled.set(not bool(self.collect_self_timer_loop_enabled.get()))
                    if bool(self.collect_self_timer_loop_enabled.get()):
                        v.timer_loop_chk.select()
                    else:
                        v.timer_loop_chk.deselect()
                except Exception:
                    pass
            elif action == "overlay" and hasattr(v, "overlay_switch"):
                _toggle_switch(v.overlay_switch)
                if hasattr(v, "_toggle_overlay"):
                    v._toggle_overlay()
            elif action == "flip_h" and hasattr(v, "_flip_h"):
                v._flip_h()
            elif action == "flip_v" and hasattr(v, "_flip_v"):
                v._flip_v()
            elif action == "batch_browse" and hasattr(v, "_browse_batch_folder"):
                v._browse_batch_folder()
            elif action == "batch_run" and hasattr(v, "_start_batch_processing"):
                v._start_batch_processing()

        def _do_training(action: str):
            v = _active_view_obj()
            if v is None or not isinstance(v, TrainingView):
                return
            if action == "start" and hasattr(v, "_on_start_training"):
                v._on_start_training()
            elif action == "advanced":
                try:
                    _toggle_switch(v.advanced_switch)
                except Exception:
                    pass
                if hasattr(v, "_toggle_advanced"):
                    v._toggle_advanced()
            elif action == "copy" and hasattr(v, "_copy_chart_to_clipboard"):
                v._copy_chart_to_clipboard()

        def _do_text(action: str):
            v = _active_view_obj()
            if v is None or not isinstance(v, TextPracticeView):
                return
            if action == "load" and hasattr(v, "_load_text"):
                v._load_text()
            elif action == "start" and hasattr(v, "_start"):
                v._start()
            elif action == "stop" and hasattr(v, "_stop"):
                v._stop()
            elif action == "restart" and hasattr(v, "_restart_camera"):
                v._restart_camera()
            elif action == "flip_h" and hasattr(v, "_flip_h"):
                v._flip_h()
            elif action == "flip_v" and hasattr(v, "_flip_v"):
                v._flip_v()
            elif action == "overlay" and hasattr(v, "overlay_switch"):
                _toggle_switch(v.overlay_switch)

        _bind("<Alt-s>", lambda _e: (_do_detection("start"), "break"))
        _bind("<Alt-S>", lambda _e: (_do_detection("start"), "break"))
        _bind("<Alt-x>", lambda _e: (_do_detection("stop"), "break"))
        _bind("<Alt-X>", lambda _e: (_do_detection("stop"), "break"))
        _bind("<Alt-c>", lambda _e: (_do_detection("clear"), "break"))
        _bind("<Alt-C>", lambda _e: (_do_detection("clear"), "break"))
        _bind("<Alt-r>", lambda _e: (_do_detection("restart"), "break"))
        _bind("<Alt-R>", lambda _e: (_do_detection("restart"), "break"))
        _bind("<Alt-h>", lambda _e: (_do_detection("flip_h"), "break"))
        _bind("<Alt-H>", lambda _e: (_do_detection("flip_h"), "break"))
        _bind("<Alt-v>", lambda _e: (_do_detection("flip_v"), "break"))
        _bind("<Alt-V>", lambda _e: (_do_detection("flip_v"), "break"))
        _bind("<Alt-e>", lambda _e: (_do_detection("enter_mode"), "break"))
        _bind("<Alt-E>", lambda _e: (_do_detection("enter_mode"), "break"))

        _bind("<Control-s>", lambda _e: (_do_collect("save"), "break"))
        _bind("<Control-S>", lambda _e: (_do_collect("save"), "break"))
        _bind("<Control-z>", lambda _e: (_do_collect("undo"), "break"))
        _bind("<Control-Z>", lambda _e: (_do_collect("undo"), "break"))
        _bind("<Control-Return>", lambda _e: (_do_collect("set_label"), "break"))
        _bind("<Control-KP_Enter>", lambda _e: (_do_collect("set_label"), "break"))
        _bind("<Alt-t>", lambda _e: (_do_collect("timer_toggle"), "break"))
        _bind("<Alt-T>", lambda _e: (_do_collect("timer_toggle"), "break"))
        _bind("<Alt-y>", lambda _e: (_do_collect("timer_enable"), "break"))
        _bind("<Alt-Y>", lambda _e: (_do_collect("timer_enable"), "break"))
        _bind("<Alt-l>", lambda _e: (_do_collect("timer_loop"), "break"))
        _bind("<Alt-L>", lambda _e: (_do_collect("timer_loop"), "break"))
        _bind("<Alt-o>", lambda _e: (_do_collect("overlay"), "break"))
        _bind("<Alt-O>", lambda _e: (_do_collect("overlay"), "break"))
        _bind("<Alt-b>", lambda _e: (_do_collect("batch_run"), "break"))
        _bind("<Alt-B>", lambda _e: (_do_collect("batch_run"), "break"))
        _bind("<Control-b>", lambda _e: (_do_collect("batch_browse"), "break"))
        _bind("<Control-B>", lambda _e: (_do_collect("batch_browse"), "break"))
        _bind("<Alt-H>", lambda _e: (_do_collect("flip_h"), "break"))
        _bind("<Alt-V>", lambda _e: (_do_collect("flip_v"), "break"))
        _bind("<Alt-R>", lambda _e: (_do_collect("restart"), "break"))

        _bind("<Alt-r>", lambda _e: (_do_training("start"), "break"))
        _bind("<Alt-R>", lambda _e: (_do_training("start"), "break"))
        _bind("<Alt-a>", lambda _e: (_do_training("advanced"), "break"))
        _bind("<Alt-A>", lambda _e: (_do_training("advanced"), "break"))
        _bind("<Alt-c>", lambda _e: (_do_training("copy"), "break"))
        _bind("<Alt-C>", lambda _e: (_do_training("copy"), "break"))

        _bind("<Alt-l>", lambda _e: (_do_text("load"), "break"))
        _bind("<Alt-L>", lambda _e: (_do_text("load"), "break"))
        _bind("<Alt-s>", lambda _e: (_do_text("start"), "break"))
        _bind("<Alt-S>", lambda _e: (_do_text("start"), "break"))
        _bind("<Alt-x>", lambda _e: (_do_text("stop"), "break"))
        _bind("<Alt-X>", lambda _e: (_do_text("stop"), "break"))
        _bind("<Alt-r>", lambda _e: (_do_text("restart"), "break"))
        _bind("<Alt-R>", lambda _e: (_do_text("restart"), "break"))
        _bind("<Alt-h>", lambda _e: (_do_text("flip_h"), "break"))
        _bind("<Alt-H>", lambda _e: (_do_text("flip_h"), "break"))
        _bind("<Alt-v>", lambda _e: (_do_text("flip_v"), "break"))
        _bind("<Alt-V>", lambda _e: (_do_text("flip_v"), "break"))
        _bind("<Alt-o>", lambda _e: (_do_text("overlay"), "break"))
        _bind("<Alt-O>", lambda _e: (_do_text("overlay"), "break"))

    def _on_startup_deps_ready(self) -> None:
        try:
            self._mp_hands = globals()["mp"].solutions.hands
            self._mp_drawing = globals()["mp"].solutions.drawing_utils
        except Exception:
            self._mp_hands = None
            self._mp_drawing = None

        try:
            self._init_mediapipe_hands()
        except Exception:
            pass

        try:
            cams = self._detect_cameras(max_cameras=5)
            if cams:
                self.available_cameras = cams
                try:
                    if int(self.camera_index_var.get()) not in cams:
                        self.camera_index_var.set(int(cams[0]))
                except Exception:
                    self.camera_index_var.set(int(cams[0]))
        except Exception:
            pass

        self.deps_ready = True

        try:
            self._finish_startup_ui()
        except Exception as exc:
            try:
                self._set_startup_overlay_text(tr("startup_loading_failed", err=str(exc)))
            except Exception:
                pass
            return

        try:
            self._refresh_camera_option_menus()
        except Exception:
            pass

        self._hide_startup_overlay()

    def _refresh_camera_option_menus(self) -> None:
        cam_values = [f"Cam {i}" for i in self.available_cameras] if self.available_cameras else ["Cam 0"]
        for v in getattr(self, "views", {}).values():
            try:
                menu = getattr(v, "cam_menu", None)
                if menu is not None:
                    menu.configure(values=cam_values)
                    menu.set(f"Cam {int(self.camera_index_var.get())}")
            except Exception:
                pass

    def _maximize_window(self) -> None:
        try:
            self.update_idletasks()
        except Exception:
            pass

        if sys.platform.startswith("win"):
            try:
                hwnd = int(self.winfo_id())
                if hwnd:
                    SW_MAXIMIZE = 3
                    ctypes.windll.user32.ShowWindow(hwnd, SW_MAXIMIZE)
                    self.update_idletasks()
            except Exception:
                pass

        try:
            self.state("zoomed")
            return
        except Exception:
            pass

        try:
            sw = int(self.winfo_screenwidth())
            sh = int(self.winfo_screenheight())
            self.geometry(f"{sw}x{sh}+0+0")
            self.update_idletasks()
        except Exception:
            pass

    def _apply_ui_theme(self) -> None:
        mode = (self.appearance_mode.get() or "Dark").strip().lower()
        if mode not in ("light", "dark"):
            mode = "dark"

        try:
            ctk.set_appearance_mode("Dark" if mode == "dark" else "Light")
        except Exception:
            pass

        base_theme = "dark-blue" if mode == "dark" else "blue"
        try:
            ctk.set_default_color_theme(base_theme)
        except Exception:
            pass

        accent_key = str(self.color_theme_dark.get() if mode == "dark" else self.color_theme_light.get() or "").strip().lower()

        accents: dict[str, tuple[str, str, str, str, str, str]] = {
            "blue": ("#3B8ED0", "#1F6AA5", "#36719F", "#144870", "#27577D", "#203A4F"),
            "green": ("#2CC985", "#2FA572", "#0C955A", "#106A43", "#0b6e3d", "#17472e"),
            "dark-blue": ("#3a7ebf", "#1f538d", "#325882", "#14375e", "#234567", "#1e2c40"),
            "purple": ("#8E44AD", "#6C3483", "#7D3C98", "#512E5F", "#5B2C6F", "#3C1C47"),
            "red": ("#E74C3C", "#B03A2E", "#C0392B", "#7B241C", "#922B21", "#641E16"),
            "orange": ("#E67E22", "#B95E00", "#CA6F1E", "#8A4500", "#A04000", "#6E2C00"),
            "teal": ("#1ABC9C", "#148F77", "#17A589", "#0E6251", "#117A65", "#0B5345"),
            "pink": ("#E84393", "#B53471", "#D63384", "#7D2252", "#A52A6D", "#5B163B"),
            "yellow": ("#F1C40F", "#B7950B", "#D4AC0D", "#7D6608", "#9A7D0A", "#5C4E06"),
            "lime": ("#7DCE13", "#4C8B0C", "#6BBE10", "#336007", "#5AA30D", "#254706"),
            "cyan": ("#00BCD4", "#00838F", "#00ACC1", "#005662", "#0097A7", "#004D56"),
            "indigo": ("#3F51B5", "#283593", "#3949AB", "#1A237E", "#303F9F", "#151B5A"),
            "gray": ("#7F8C8D", "#566573", "#707B7C", "#3E4A54", "#626F70", "#2E343B"),
            "amber": ("#FFB300", "#C58A00", "#E6A700", "#8F6400", "#D18F00", "#6A4A00"),
        }

        palette = accents.get(accent_key) or accents.get("blue")
        if not palette:
            return
        main_light, main_dark, hover_light, hover_dark, opt_hover_light, opt_hover_dark = palette

        try:
            t = ctk.ThemeManager.theme

            def _set(widget: str, key: str, value):
                try:
                    if widget in t and isinstance(t[widget], dict) and key in t[widget]:
                        t[widget][key] = value
                except Exception:
                    pass

            _set("CTkButton", "fg_color", [main_light, main_dark])
            _set("CTkButton", "hover_color", [hover_light, hover_dark])

            _set("CTkCheckBox", "fg_color", [main_light, main_dark])
            _set("CTkCheckBox", "hover_color", [hover_light, hover_dark])

            _set("CTkSwitch", "progress_color", [main_light, main_dark])

            _set("CTkRadioButton", "fg_color", [main_light, main_dark])
            _set("CTkRadioButton", "hover_color", [hover_light, hover_dark])

            _set("CTkProgressBar", "progress_color", [main_light, main_dark])

            _set("CTkSlider", "button_color", [main_light, main_dark])
            _set("CTkSlider", "button_hover_color", [hover_light, hover_dark])

            _set("CTkOptionMenu", "fg_color", [main_light, main_dark])
            _set("CTkOptionMenu", "button_color", [hover_light, hover_dark])
            _set("CTkOptionMenu", "button_hover_color", [opt_hover_light, opt_hover_dark])

            _set("CTkSegmentedButton", "selected_color", [main_light, main_dark])
            _set("CTkSegmentedButton", "selected_hover_color", [hover_light, hover_dark])
        except Exception:
            pass

        try:
            self.update_idletasks()
            w = int(self.winfo_screenwidth())
            h = int(self.winfo_screenheight())
            if w > 0 and h > 0:
                self.geometry(f"{w}x{h}+0+0")
        except Exception:
            pass

    def _settings_to_dict(self) -> dict:
        def _get_int(var, default: int) -> int:
            try:
                return int(var.get())
            except Exception:
                return default

        def _get_bool(var, default: bool) -> bool:
            try:
                return bool(var.get())
            except Exception:
                return default

        def _get_str(var, default: str) -> str:
            try:
                return str(var.get())
            except Exception:
                return default

        det_threshold = 0.7
        try:
            det_threshold = float(self.det_threshold)
        except Exception:
            det_threshold = 0.7

        return {
            "version": 1,
            "ui": {
                "appearance": str(self.appearance_mode.get() or "Dark"),
                "theme_light": str(self.color_theme_light.get() or "blue"),
                "theme_dark": str(self.color_theme_dark.get() or "dark-blue"),
            },
            "paths": {
                "csv": _get_str(self.csv_file_var, str(Path("data") / "data.csv")),
                "model": _get_str(self.model_file_var, str(Path("models") / "model.h5")),
                "scaler": _get_str(self.scaler_file_var, str(Path("other") / "scaler.pkl")),
                "images": str(self.images_dir or "images"),
            },
            "detection": {
                "interval_ms": _get_int(self.det_interval_ms, 1000),
                "threshold": det_threshold,
                "enter_mode": _get_bool(self.enter_mode, False),
            },
            "mediapipe": {
                "static_image_mode": _get_bool(self.static_image_mode, False),
                "max_num_hands": _get_int(self.max_num_hands, 1),
                "model_complexity": _get_int(self.model_complexity, 1),
                "min_detection_confidence": _get_int(self.min_detection_confidence, 50),
                "min_tracking_confidence": _get_int(self.min_tracking_confidence, 50),
            },
        }

    def save_settings_to_file(self) -> None:
        try:
            self.settings_file.parent.mkdir(parents=True, exist_ok=True)
            payload = self._settings_to_dict()
            with self.settings_file.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            self.log(tr("log_settings_saved", path=str(self.settings_file)))
        except Exception as exc:
            try:
                self.log(tr("log_settings_save_failed", err=str(exc)))
            except Exception:
                pass

    def _load_settings_from_file(self) -> None:
        try:
            path = self.settings_file
            if not path.exists():
                return
            with path.open("r", encoding="utf-8") as f:
                cfg = json.load(f) or {}

            ui = cfg.get("ui") or {}
            paths = cfg.get("paths") or {}
            det = cfg.get("detection") or {}
            mp_cfg = cfg.get("mediapipe") or {}

            try:
                v = str(ui.get("appearance") or "").strip().lower()
                if v in ("light", "dark"):
                    self.appearance_mode.set("Light" if v == "light" else "Dark")
            except Exception:
                pass
            try:
                v = str(ui.get("theme_light") or "").strip().lower()
                if v in (
                    "blue",
                    "green",
                    "dark-blue",
                    "purple",
                    "red",
                    "orange",
                    "teal",
                    "pink",
                    "yellow",
                    "lime",
                    "cyan",
                    "indigo",
                    "gray",
                    "amber",
                ):
                    self.color_theme_light.set(v)
            except Exception:
                pass
            try:
                v = str(ui.get("theme_dark") or "").strip().lower()
                if v in (
                    "blue",
                    "green",
                    "dark-blue",
                    "purple",
                    "red",
                    "orange",
                    "teal",
                    "pink",
                    "yellow",
                    "lime",
                    "cyan",
                    "indigo",
                    "gray",
                    "amber",
                ):
                    self.color_theme_dark.set(v)
            except Exception:
                pass

            csv_p = str(paths.get("csv") or "").strip()
            if csv_p:
                self.csv_file_var.set(csv_p)
            model_p = str(paths.get("model") or "").strip()
            if model_p:
                self.model_file_var.set(model_p)
            scaler_p = str(paths.get("scaler") or "").strip()
            if scaler_p:
                self.scaler_file_var.set(scaler_p)
            images_p = str(paths.get("images") or "").strip()
            if images_p:
                self.images_dir = images_p

            try:
                interval_ms = int(det.get("interval_ms"))
                if interval_ms > 0:
                    self.det_interval_ms.set(interval_ms)
            except Exception:
                pass
            try:
                threshold = float(det.get("threshold"))
                if 0.0 < threshold <= 1.0:
                    self.det_threshold = threshold
            except Exception:
                pass
            try:
                self.enter_mode.set(bool(det.get("enter_mode")))
            except Exception:
                pass

            try:
                self.static_image_mode.set(bool(mp_cfg.get("static_image_mode")))
            except Exception:
                pass
            try:
                v = int(mp_cfg.get("max_num_hands"))
                v = min(MAX_HANDS_SUPPORTED, max(1, v))
                self.max_num_hands.set(v)
            except Exception:
                pass
            try:
                v = int(mp_cfg.get("model_complexity"))
                v = max(0, min(2, v))
                self.model_complexity.set(v)
            except Exception:
                pass
            try:
                v = int(mp_cfg.get("min_detection_confidence"))
                v = max(0, min(100, v))
                self.min_detection_confidence.set(v)
            except Exception:
                pass
            try:
                v = int(mp_cfg.get("min_tracking_confidence"))
                v = max(0, min(100, v))
                self.min_tracking_confidence.set(v)
            except Exception:
                pass

            self.log(tr("log_settings_loaded", path=str(path)))
        except Exception as exc:
            try:
                self.log(tr("log_settings_load_failed", err=str(exc)))
            except Exception:
                pass

    def set_language(self, code: str) -> None:
        code = (code or "").strip().lower()
        if code not in ("pl", "en"):
            code = "en"
        if self.language_code.get() == code:
            return
        self.language_code.set(code)
        try:
            load(code)
        except Exception:
            pass
        try:
            self._localize_camera_size_preset()
        except Exception:
            pass
        self.rebuild_ui()

    def _camera_size_preset_key(self, preset_value: str) -> str:
        v = (preset_value or "").strip().lower()
        if v.startswith(("ma", "sm")):
            return "small"
        if v.startswith(("du", "la", "bi")):
            return "large"
        return "medium"

    def _localize_camera_size_preset(self) -> None:
        key = self._camera_size_preset_key(self.camera_size_preset.get())
        self.camera_size_preset.set(tr(f"camera_preset_{key}"))

    def rebuild_ui(self) -> None:
        prev_view = self.active_view or "detekcja"

        log_text = ""
        try:
            if hasattr(self, "console") and self.console is not None:
                self.console.configure(state="normal")
                log_text = self.console.get("1.0", "end")
                self.console.configure(state="disabled")
        except Exception:
            log_text = ""

        try:
            self.stop_camera_loop()
        except Exception:
            pass

        for attr in ("sidebar", "content", "console_frame"):
            try:
                w = getattr(self, attr, None)
                if w is not None:
                    w.destroy()
            except Exception:
                pass

        try:
            self.title(tr("app_title_ctk"))
        except Exception:
            pass

        self._build_sidebar()
        self._build_main_content()
        self._build_console()

        self._ensure_floating_console_button()

        if not bool(self.console_visible.get()):
            try:
                self.console_frame.grid_remove()
            except Exception:
                pass

        if log_text.strip():
            try:
                self.console.configure(state="normal")
                self.console.insert("end", log_text)
                self.console.see("end")
                self.console.configure(state="disabled")
            except Exception:
                pass

        self.show_view(prev_view)
        self.log(tr("log_language_changed", lang=self.language_code.get()))

    def _ensure_floating_console_button(self) -> None:
        try:
            if self.floating_console_btn is None or not self.floating_console_btn.winfo_exists():
                self.floating_console_btn = ctk.CTkButton(
                    self,
                    text=tr("btn_show_console"),
                    width=160,
                    height=36,
                    command=self.toggle_console,
                )
        except Exception:
            return

        try:
            if bool(self.console_visible.get()):
                self.floating_console_btn.place_forget()
            else:
                self.floating_console_btn.configure(text=tr("btn_show_console"))
                self.floating_console_btn.place(relx=1.0, rely=1.0, anchor="se", x=-18, y=-18)
        except Exception:
            pass

    def camera_layout_ratio(self) -> float:
        preset = (self.camera_size_preset.get() or "").strip().lower()
        if preset.startswith(("ma", "sm")):
            return 0.50
        if preset.startswith(("du", "la", "bi")):
            return 0.72
        return 0.62

    def camera_pane_minsize(self) -> int:
        preset = (self.camera_size_preset.get() or "").strip().lower()
        if preset.startswith(("ma", "sm")):
            return 240
        if preset.startswith(("du", "la", "bi")):
            return 420
        return 320

    def apply_camera_layout_preset(self) -> None:
        for view in self.views.values():
            if hasattr(view, "apply_camera_layout_preset"):
                try:
                    view.apply_camera_layout_preset()
                except Exception:
                    pass

    def reset_to_defaults(self) -> None:
        self.flip_horizontal = False
        self.flip_vertical = False
        self.show_overlays = True

        self.brightness.set(0)
        self.contrast.set(100)
        self.gamma.set(100)
        self.shift_r.set(0)
        self.shift_g.set(0)
        self.shift_b.set(0)

        self.static_image_mode.set(False)
        self.max_num_hands.set(1)
        self.model_complexity.set(1)
        self.min_detection_confidence.set(50)
        self.min_tracking_confidence.set(50)
        self._init_mediapipe_hands()

        self.det_threshold = 0.7
        self.det_interval_ms.set(1000)
        self.enter_mode.set(False)
        self.reset_detector()
        self.log(tr("log_reset_defaults"))

    def _build_sidebar(self) -> None:
        self.sidebar = ctk.CTkFrame(self, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(0, weight=0)
        self.sidebar.grid_rowconfigure(1, weight=1)
        self.sidebar.grid_rowconfigure(2, weight=0)

        header = ctk.CTkLabel(
            self.sidebar,
            text=tr("app_title_ctk"),
            font=ctk.CTkFont(size=18, weight="bold"),
        )
        header.grid(row=0, column=0, padx=18, pady=(18, 10), sticky="w")

        nav = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        nav.grid(row=1, column=0, padx=12, pady=12, sticky="nsew")
        nav.grid_columnconfigure(0, weight=1)

        self.nav_buttons: dict[str, ctk.CTkButton] = {}
        self._nav_label_base: dict[str, str] = {}
        self._nav_default_style: dict[str, dict] = {}

        def add_nav(key: str, text: str):
            btn = ctk.CTkButton(
                nav,
                text=text,
                height=44,
                command=lambda k=key: self.show_view(k),
            )
            btn.grid(pady=8, sticky="ew")
            self.nav_buttons[key] = btn
            self._nav_label_base[key] = text
            self._nav_default_style[key] = {
                "fg_color": btn.cget("fg_color"),
                "text_color": btn.cget("text_color"),
                "border_width": btn.cget("border_width"),
                "border_color": btn.cget("border_color"),
                "font": btn.cget("font"),
            }

        add_nav("zbieranie", tr("tab_collect"))
        add_nav("detekcja", tr("tab_detection"))
        add_nav("trening", tr("tab_train"))
        add_nav("tekst", tr("tab_text"))
        add_nav("instrukcja", tr("tab_instr"))
        add_nav("ustawienia", tr("tab_settings"))

        footer = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        footer.grid(row=2, column=0, padx=12, pady=(0, 14), sticky="ew")
        footer.grid_columnconfigure(0, weight=1)

        self.theme_switch = ctk.CTkSwitch(
            footer,
            text=tr("theme_switch"),
            command=self._toggle_theme,
        )
        self.theme_switch.grid(row=0, column=0, sticky="w")
        try:
            if ctk.get_appearance_mode().lower() == "dark":
                self.theme_switch.select()
            else:
                self.theme_switch.deselect()
        except Exception:
            self.theme_switch.select()

    def _toggle_theme(self) -> None:
        if self.theme_switch.get() == 1:
            self.appearance_mode.set("Dark")
            self.log(tr("log_theme", theme="Dark"))
        else:
            self.appearance_mode.set("Light")
            self.log(tr("log_theme", theme="Light"))

        self._apply_ui_theme()
        self.save_settings_to_file()

        self.rebuild_ui()

    def _build_main_content(self) -> None:
        self.content = ctk.CTkFrame(self)
        self.content.grid(row=0, column=1, sticky="nsew")
        self.content.grid_rowconfigure(0, weight=1)
        self.content.grid_columnconfigure(0, weight=1)

        self.views: dict[str, ctk.CTkFrame] = {
            "detekcja": DetectionView(self.content, app=self),
            "zbieranie": DataCollectionView(self.content, app=self),
            "trening": TrainingView(self.content, app=self),
            "tekst": TextPracticeView(self.content, app=self),
            "instrukcja": InstructionsView(self.content, app=self),
            "ustawienia": SettingsView(self.content, app=self),
        }

        for view in self.views.values():
            view.grid(row=0, column=0, sticky="nsew")

    def show_view(self, key: str) -> None:
        if key not in self.views:
            return

        if self.active_view in self.views:
            old_view = self.views[self.active_view]
            if hasattr(old_view, "on_hide"):
                try:
                    old_view.on_hide()
                except Exception:
                    pass

        self.views[key].tkraise()
        self.active_view = key

        active_fg = _theme_value("CTkButton", "hover_color", None)
        active_border = _theme_value("CTkButton", "text_color", None)
        for k, btn in self.nav_buttons.items():
            base_text = self._nav_label_base.get(k, btn.cget("text"))
            if k == key:
                btn.configure(
                    text=f"▸ {base_text}",
                    fg_color=active_fg,
                    border_width=2,
                    border_color=active_border,
                    font=ctk.CTkFont(size=13, weight="bold"),
                    state="normal",
                )
            else:
                defaults = self._nav_default_style.get(k, {})
                btn.configure(
                    text=base_text,
                    fg_color=defaults.get("fg_color"),
                    text_color=defaults.get("text_color"),
                    border_width=defaults.get("border_width"),
                    border_color=defaults.get("border_color"),
                    font=defaults.get("font"),
                    state="normal",
                )

        self.log(tr("log_view_changed", view=key))

        new_view = self.views[key]
        if hasattr(new_view, "on_show"):
            if not bool(getattr(self, "deps_ready", False)):
                return
            try:
                new_view.on_show()
            except Exception as exc:
                self.log(tr("log_view_on_show_error", err=str(exc)))

    def open_camera(self, index: int = 0) -> bool:
        self.close_camera()
        self._camera_index = int(index)
        self.camera_index_var.set(self._camera_index)

        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        cap = cv2.VideoCapture(self._camera_index, backend)
        if not cap.isOpened():
            self.log(tr("log_camera_open_failed", idx=self._camera_index))
            return False

        cap.read()
        self._cap = cap
        self.log(tr("log_camera_opened", idx=self._camera_index))
        return True

    def close_camera(self) -> None:
        self.stop_camera_loop()
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
        self._cap = None

    def start_camera_loop(self, consumer, interval_ms: int = 33) -> None:
        self._camera_last_consumer = consumer
        self._camera_last_interval_ms = int(interval_ms)
        if self._camera_running:
            return
        if self._cap is None or not self._cap.isOpened():
            if not self.open_camera(self._camera_index):
                return
        self._camera_running = True

        def _tick():
            if not self._camera_running:
                return
            if self._cap is None:
                self._camera_running = False
                return
            ok, frame = self._cap.read()
            if ok and frame is not None:
                try:
                    consumer(frame)
                except Exception as exc:
                    now = time.time()
                    if (now - self._last_cam_consumer_log_ts) > 1.0:
                        self._last_cam_consumer_log_ts = now
                        self.log(tr("log_camera_consumer_error", err=str(exc)))
            self._camera_after_id = self.after(interval_ms, _tick)

        _tick()

    def restart_camera(self, index: int | None = None) -> None:
        if self._camera_restart_in_progress:
            return
        self._camera_restart_in_progress = True

        idx = int(self._camera_index if index is None else index)
        self._camera_index = idx
        try:
            self.camera_index_var.set(idx)
        except Exception:
            pass

        self.stop_camera_loop()
        old_cap = self._cap
        self._cap = None
        if old_cap is not None:
            try:
                old_cap.release()
            except Exception:
                pass

        self.log(tr("log_camera_restarting"))

        def _worker_open():
            cap = None
            try:
                backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
                cap = cv2.VideoCapture(idx, backend)
                if not cap.isOpened():
                    try:
                        cap.release()
                    except Exception:
                        pass
                    cap = None
                else:
                    cap.read()
            except Exception:
                cap = None

            def _finish():
                self._camera_restart_in_progress = False
                if cap is None:
                    self.log(tr("log_camera_open_failed", idx=idx))
                    return
                self._cap = cap
                self.log(tr("log_camera_opened", idx=idx))
                if self._camera_last_consumer is not None and self.active_view is not None:
                    try:
                        self.start_camera_loop(self._camera_last_consumer, interval_ms=self._camera_last_interval_ms)
                    except Exception:
                        pass

            try:
                self.after(0, _finish)
            except Exception:
                self._camera_restart_in_progress = False
                try:
                    if cap is not None:
                        cap.release()
                except Exception:
                    pass

        threading.Thread(target=_worker_open, daemon=True).start()

    def stop_camera_loop(self) -> None:
        self._camera_running = False
        if self._camera_after_id is not None:
            try:
                self.after_cancel(self._camera_after_id)
            except Exception:
                pass
        self._camera_after_id = None

    def get_detector(self) -> "HandSignDetector":
        if self._detector is None:
            self._detector = HandSignDetector(app=self)
        return self._detector

    def reset_detector(self) -> None:
        self._detector = None

    def _detect_cameras(self, max_cameras: int = 5) -> list[int]:
        cams: list[int] = []
        for i in range(max_cameras):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
            if cap.isOpened():
                ok, _ = cap.read()
                if ok:
                    cams.append(i)
            cap.release()
        return cams

    def _prepare_directories_and_csv(self) -> None:
        csv_path = Path(self.csv_file_var.get())
        if not csv_path.parent.exists():
            csv_path.parent.mkdir(parents=True, exist_ok=True)
        img_dir = Path(self.images_dir)
        img_dir.mkdir(parents=True, exist_ok=True)

        if not csv_path.exists():
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(_multi_hand_header())

    def _init_mediapipe_hands(self) -> None:
        if self._mp_hands is None:
            return
        if self._hands is not None:
            try:
                self._hands.close()
            except Exception:
                pass
        self._hands = self._mp_hands.Hands(
            static_image_mode=bool(self.static_image_mode.get()),
            max_num_hands=min(MAX_HANDS_SUPPORTED, max(1, int(self.max_num_hands.get()))),
            model_complexity=int(self.model_complexity.get()),
            min_detection_confidence=float(self.min_detection_confidence.get()) / 100.0,
            min_tracking_confidence=float(self.min_tracking_confidence.get()) / 100.0,
        )

    def apply_image_adjustments(self, frame_bgr: np.ndarray) -> np.ndarray:
        frame = frame_bgr.astype(np.float32)
        beta = float(self.brightness.get())
        alpha = max(0.01, float(self.contrast.get()) / 100.0)
        frame = frame * alpha + beta

        frame[:, :, 2] += float(self.shift_r.get())
        frame[:, :, 1] += float(self.shift_g.get())
        frame[:, :, 0] += float(self.shift_b.get())

        frame = np.clip(frame, 0, 255)
        frame = frame / 255.0
        real_gamma = max(0.01, float(self.gamma.get()) / 100.0)
        frame = np.power(frame, 1.0 / real_gamma)
        frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
        return frame

    def set_label(self, label_text: str) -> None:
        label_text = (label_text or "").strip()
        if not label_text:
            messagebox.showerror(tr("dlg_error"), tr("err_label_empty"))
            return
        self.current_label = label_text.upper()
        self._refresh_collect_next_index()
        self.log(tr("log_label_selected", val=self.current_label))

    def _refresh_collect_next_index(self) -> None:
        label = (self.current_label or "").strip()
        if not label:
            self._collect_next_idx_label = None
            self._collect_next_idx = None
            return
        try:
            label_dir = Path(self.images_dir) / label
            label_dir.mkdir(parents=True, exist_ok=True)
            self._collect_next_idx = self._compute_next_image_index(label_dir, label)
            self._collect_next_idx_label = label
        except Exception:
            self._collect_next_idx_label = label
            self._collect_next_idx = None

    @staticmethod
    def _compute_next_image_index(label_dir: Path, label: str) -> int:
        max_idx = -1
        prefix = f"{label}_"
        for p in label_dir.glob("*.jpg"):
            name = p.stem
            if not name.startswith(prefix):
                continue
            tail = name[len(prefix) :]
            try:
                n = int(tail)
            except Exception:
                continue
            if n > max_idx:
                max_idx = n
        return max_idx + 1

    def get_collect_next_index(self) -> int | None:
        if not self.current_label:
            return None
        if self._collect_next_idx_label != self.current_label:
            self._refresh_collect_next_index()
        return self._collect_next_idx

    @staticmethod
    def _mediapipe_results_confidence(results) -> float:
        try:
            handedness = getattr(results, "multi_handedness", None)
            if handedness:
                scores: list[float] = []
                for h in handedness:
                    try:
                        scores.append(float(h.classification[0].score))
                    except Exception:
                        pass
                if scores:
                    return float(max(scores))
        except Exception:
            pass
        try:
            return 1.0 if results and getattr(results, "multi_hand_landmarks", None) else 0.0
        except Exception:
            return 0.0

    def _resolve_path(self, p: Path) -> Path:
        try:
            if p.is_absolute():
                return p
        except Exception:
            pass
        return (self.base_dir / p).resolve()

    def undo_last_sample(self) -> None:
        info = self._last_saved_sample
        if not info:
            self.log(tr("log_undo_none"))
            return

        try:
            csv_path = self._resolve_path(Path(str(self.csv_file_var.get() or "").strip()))
            if not csv_path.exists():
                self.log(tr("log_undo_failed", err=f"CSV missing: {csv_path}"))
                return

            rows: list[list[str]] = []
            with csv_path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                rows = list(reader)

            if len(rows) <= 1:
                self.log(tr("log_undo_none"))
                return

            rows.pop()
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerows(rows)

            img_path = Path(str(info.get("img_path") or "").strip())
            if img_path:
                img_path = self._resolve_path(img_path)
                try:
                    if img_path.exists():
                        img_path.unlink()
                except Exception:
                    pass

            self._last_saved_sample = None

            try:
                self._refresh_collect_next_index()
            except Exception:
                pass

            self.log(tr("log_undo_done"))
        except Exception as exc:
            self.log(tr("log_undo_failed", err=str(exc)))

    def save_sample(self) -> None:
        MIN_CONF = 0.60
        if self.current_label is None:
            self.log(tr("log_first_set_label"))
            return
        if self.last_frame_raw is None:
            self.log(tr("log_no_camera_data"))
            return
        if not bool(getattr(self, "deps_ready", True)):
            self.log(tr("err_deps_not_ready"))
            return
        if self._hands is None:
            self.log(tr("err_mediapipe_not_ready"))
            return

        frame = self.last_frame_raw.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._hands.process(frame_rgb)
        if not results.multi_hand_landmarks:
            self.log(tr("log_no_hand"))
            return

        conf = self._mediapipe_results_confidence(results)
        if conf < MIN_CONF:
            self.log(tr("log_hand_conf_too_low", conf=f"{conf:.2f}"))
            return

        row = _extract_multi_hand_features(results, max_hands=MAX_HANDS_SUPPORTED)

        label_dir = Path(self.images_dir) / self.current_label
        label_dir.mkdir(parents=True, exist_ok=True)
        idx = self._compute_next_image_index(label_dir, self.current_label)

        img_path = label_dir / f"{self.current_label}_{idx}.jpg"
        try:
            ok = bool(cv2.imwrite(str(img_path), frame))
        except Exception:
            ok = False
        if not ok:
            self.log(tr("log_save_image_failed", path=str(img_path)))
            return

        csv_path = Path(self.csv_file_var.get())
        with csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row + [self.current_label, idx])

        self.log(tr("log_saved_sample", label=self.current_label, idx=idx, path=str(csv_path)))

        try:
            self._last_saved_sample = {
                "label": str(self.current_label),
                "idx": int(idx),
                "img_path": str(img_path),
                "csv_path": str(csv_path),
            }
        except Exception:
            self._last_saved_sample = {"img_path": str(img_path), "csv_path": str(csv_path)}

        try:
            self._collect_next_idx_label = self.current_label
            self._collect_next_idx = int(idx) + 1
        except Exception:
            pass

    def clear_images(self) -> None:
        img_dir = Path(self.images_dir)
        if not img_dir.exists():
            self.log(tr("log_images_empty"))
            return
        if not messagebox.askokcancel(tr("dlg_confirm"), tr("dlg_sure_clear_images")):
            self.log(tr("log_action_cancelled"))
            return
        for p in img_dir.glob("**/*"):
            try:
                if p.is_file():
                    p.unlink()
            except Exception:
                pass
        for d in sorted([d for d in img_dir.glob("**/*") if d.is_dir()], key=lambda x: len(str(x)), reverse=True):
            try:
                d.rmdir()
            except Exception:
                pass
        self.log(tr("log_images_cleared"))

        self._collect_next_idx_label = None
        self._collect_next_idx = None

    def reset_csv(self) -> None:
        csv_path = Path(self.csv_file_var.get())
        if not messagebox.askokcancel(tr("dlg_confirm"), tr("dlg_sure_reset_csv", file=str(csv_path))):
            self.log(tr("log_action_cancelled"))
            return
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(_multi_hand_header())
        self.log(tr("log_csv_reset", path=str(csv_path)))

    def start_training(self) -> None:
        def _run():
            mod = globals().get("training")
            if mod is None:
                try:
                    globals()["training"] = importlib.import_module("ctk_app.training_worker")
                    mod = globals()["training"]
                except Exception as exc:
                    self.log(tr("log_training_exception", err=str(exc)))
                    return
            mod.run_training_in_thread(self)

        threading.Thread(target=_run, daemon=True).start()

    def show_training_plots(self, history) -> None:
        try:
            self._last_training_history = history
        except Exception:
            pass
        self.reset_detector()
        view = self.views.get("trening")
        if view and hasattr(view, "show_plots"):
            try:
                view.show_plots(history)
            except Exception as exc:
                self.log(tr("log_plot_error", err=str(exc)))

    def show_training_confusion_matrix(self, cm: np.ndarray, labels: list[str] | None = None) -> None:
        try:
            self._last_training_cm = np.asarray(cm)
            self._last_training_cm_labels = list(labels) if labels else None
        except Exception:
            pass

        view = self.views.get("trening")
        if view and hasattr(view, "show_confusion_matrix"):
            try:
                view.show_confusion_matrix(cm, labels)
            except Exception as exc:
                self.log(tr("log_plot_error", err=str(exc)))

    def _on_enter(self, _event=None):
        if self._focus_is_text_input():
            return

        if self.active_view == "zbieranie":
            try:
                self.save_sample()
            except Exception:
                pass
            return

        if self.active_view == "detekcja":
            view = self.views.get("detekcja")
            if view and hasattr(view, "on_enter"):
                view.on_enter()

    def _on_space(self, _event=None):
        if self._focus_is_text_input():
            return

        if self.active_view == "zbieranie":
            self.save_sample()

    def on_app_close(self) -> None:
        self.stop_camera_loop()
        self.close_camera()
        self.destroy()

    def _build_console(self) -> None:
        self.console_frame = ctk.CTkFrame(self, corner_radius=0)
        self.console_frame.grid(row=1, column=0, columnspan=2, sticky="nsew")

        self.console_frame.grid_rowconfigure(0, weight=0)
        self.console_frame.grid_rowconfigure(1, weight=1)
        self.console_frame.grid_columnconfigure(0, weight=1)

        header = ctk.CTkFrame(self.console_frame, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=14, pady=(10, 4))
        header.grid_columnconfigure(0, weight=1)

        title = ctk.CTkLabel(
            header,
            text=tr("console_title"),
            font=ctk.CTkFont(size=13, weight="bold"),
        )
        title.grid(row=0, column=0, sticky="w")

        self.console_toggle_btn = ctk.CTkButton(
            header,
            text=tr("btn_hide_console") if bool(self.console_visible.get()) else tr("btn_show_console"),
            width=120,
            height=28,
            command=self.toggle_console,
        )
        self.console_toggle_btn.grid(row=0, column=1, sticky="e")

        self.console = ctk.CTkTextbox(
            self.console_frame,
            height=150,
            wrap="word",
            font=("Consolas", 11),
        )
        self.console.grid(row=1, column=0, sticky="nsew", padx=14, pady=(0, 12))
        self.console.configure(state="disabled")

        try:
            if getattr(self, "_log_buffer", None):
                self.console.configure(state="normal")
                for buffered_line in self._log_buffer:
                    self.console.insert("end", buffered_line)
                self.console.see("end")
                self.console.configure(state="disabled")
                self._log_buffer.clear()
        except Exception:
            pass

        self._ensure_floating_console_button()

    def toggle_console(self) -> None:
        is_visible = bool(self.console_visible.get())
        if is_visible:
            try:
                self.console_frame.grid_remove()
            except Exception:
                return
            self.console_visible.set(False)
        else:
            try:
                self.console_frame.grid()
            except Exception:
                return
            self.console_visible.set(True)

        try:
            if hasattr(self, "console_toggle_btn") and self.console_toggle_btn is not None:
                self.console_toggle_btn.configure(
                    text=tr("btn_hide_console") if bool(self.console_visible.get()) else tr("btn_show_console")
                )
        except Exception:
            pass

        self._ensure_floating_console_button()

    def log(self, message: str) -> None:
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {message}\n"

        if not hasattr(self, "console") or self.console is None:
            try:
                self._log_buffer.append(line)
            except Exception:
                pass
            return

        self.console.configure(state="normal")
        self.console.insert("end", line)
        self.console.see("end")
        self.console.configure(state="disabled")

