import os
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import csv
import cv2
import numpy as np
import threading
import mediapipe as mp
import joblib
import tensorflow as tf
import platform
import time
from PIL import Image, ImageTk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from locales import tr, load as set_language
import training
import detection
import gui_collect
import gui_train
import gui_detection
import gui_instructions
import gui_text_detection
from utils import disable_space_activation, TextRedirector

class ScrollableFrame(ttk.Frame):
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.v_scroll = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.h_scroll = ttk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=self.v_scroll.set, xscrollcommand=self.h_scroll.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.v_scroll.grid(row=0, column=1, sticky="ns")
        self.h_scroll.grid(row=1, column=0, sticky="ew")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.scrollable_frame = ttk.Frame(self.canvas)
        self.scrollable_frame.bind(
            "<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        self._window = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        # Make the embedded frame track the canvas width so content can expand horizontally
        # (prevents large empty area on the right when the app window is wide).
        def _sync_width(event):
            self.canvas.itemconfigure(self._window, width=event.width)

        self.canvas.bind("<Configure>", _sync_width)

        system = platform.system()
        if system in ("Windows", "Darwin"):
            self.canvas.bind("<Enter>", lambda e: self.canvas.bind_all("<MouseWheel>", self._on_mousewheel))
            self.canvas.bind("<Leave>", lambda e: self.canvas.unbind_all("<MouseWheel>"))
        else:
            self.canvas.bind("<Enter>", lambda e: (
                self.canvas.bind_all("<Button-4>", lambda ev: self.canvas.yview_scroll(-1, "units")),
                self.canvas.bind_all("<Button-5>", lambda ev: self.canvas.yview_scroll(1, "units"))
            ))
            self.canvas.bind("<Leave>", lambda e: (
                self.canvas.unbind_all("<Button-4>"),
                self.canvas.unbind_all("<Button-5>")
            ))

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

class HandDataCollectorApp:
    def __init__(self, root):
        self.root = root
        self.theme_var = tk.StringVar(value="light")
        set_language("en")
        self.language_var = tk.StringVar(value="en")
        self.ui_scale = self._compute_ui_scale()
        try:
            # Shrink UI on high DPI/small screens so the layout stays accessible without excessive scrolling.
            self.root.tk.call("tk", "scaling", self.ui_scale)
        except tk.TclError:
            pass
        self.root.title(tr("main_window_title"))
        self.root.minsize(self._s(1100), self._s(750))
        try:
            self.root.state("zoomed")
        except tk.TclError:
            self.root.attributes("-fullscreen", True)
        self.root.rowconfigure(1, weight=1)
        self.root.columnconfigure(0, weight=1)
        self.top_bar = ttk.Frame(self.root, padding=(self._s(16), self._s(10)))
        self.top_bar.grid(row=0, column=0, sticky="ew")
        self.top_bar.columnconfigure(3, weight=1)
        self.brand_label = ttk.Label(self.top_bar, text="HandSignAI", style="Accent.TLabel")
        self.brand_label.grid(row=0, column=0, sticky="w")
        self.lang_label = ttk.Label(self.top_bar, text=tr("language_label"))
        self.lang_label.grid(row=0, column=1, padx=(16, 6))
        self.lang_combo = ttk.Combobox(
            self.top_bar,
            textvariable=self.language_var,
            values=["en", "pl"],
            state="readonly",
            width=6
        )
        self.lang_combo.grid(row=0, column=2, sticky="w")
        self.lang_combo.bind("<<ComboboxSelected>>", self.change_language)
        self.theme_label = ttk.Label(self.top_bar, text=tr("theme_label"))
        self.theme_label.grid(row=0, column=3, padx=(16, 6), sticky="e")
        self.theme_combo = ttk.Combobox(
            self.top_bar,
            textvariable=self.theme_var,
            values=["light", "dark", "high-contrast"],
            state="readonly",
            width=14
        )
        self.theme_combo.grid(row=0, column=4, sticky="w")
        self.theme_combo.bind("<<ComboboxSelected>>", self.change_theme)
        self.paned = ttk.Panedwindow(self.root, orient=tk.VERTICAL)
        self.paned.grid(row=1, column=0, sticky="nsew")
        self.content_frame = ttk.Frame(self.paned, padding=(self._s(12), self._s(12)))
        self.logs_frame = ttk.Frame(self.paned, padding=(self._s(12), self._s(8)))
        self.paned.add(self.content_frame, weight=5)
        self.paned.add(self.logs_frame, weight=2)
        self.content_frame.rowconfigure(0, weight=1)
        self.content_frame.columnconfigure(0, weight=1)
        self.interval_var = tk.StringVar(value="1000")
        self.enter_mode_var = tk.BooleanVar(value=False)
        self.show_overlays_var = tk.BooleanVar(value=True)
        self.logs_dir = "other"
        self.logs_file_path = os.path.join(self.logs_dir, "logs.log")
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir, exist_ok=True)
        self.log_file = open(self.logs_file_path, mode='w', encoding='utf-8')
        self.content_scroller = ScrollableFrame(self.content_frame)
        self.content_scroller.grid(row=0, column=0, sticky="nsew")
        self.content_scroller.grid_rowconfigure(0, weight=1)
        self.content_scroller.grid_columnconfigure(0, weight=1)

        # Allow notebook to expand within the embedded scrollable frame.
        self.content_scroller.scrollable_frame.grid_rowconfigure(0, weight=1)
        self.content_scroller.scrollable_frame.grid_columnconfigure(0, weight=1)
        self.notebook = ttk.Notebook(self.content_scroller.scrollable_frame)
        self.notebook.grid(row=0, column=0, sticky="nsew", padx=(0, self._s(10)), pady=(0, self._s(10)))
        self.tab_collect = ttk.Frame(self.notebook, padding=(10, 10))
        self.notebook.add(self.tab_collect, text=tr("tab_collect"))
        self.tab_train = ttk.Frame(self.notebook, padding=(10, 10))
        self.notebook.add(self.tab_train, text=tr("tab_train"))
        self.tab_detection = ttk.Frame(self.notebook, padding=(10, 10))
        self.notebook.add(self.tab_detection, text=tr("tab_detection"))
        self.tab_text_detection = ttk.Frame(self.notebook, padding=(10, 10))
        self.notebook.add(self.tab_text_detection, text=tr("tab_text"))
        self.tab_instructions = ttk.Frame(self.notebook, padding=(10, 10))
        self.notebook.add(self.tab_instructions, text=tr("tab_instr"))
        self.last_tab_index = 0
        self.notebook.bind('<<NotebookTabChanged>>', self.on_tab_changed)
        self.logs_frame.columnconfigure(0, weight=1)
        self.logs_frame.rowconfigure(1, weight=1)
        self.log_header = ttk.Label(self.logs_frame, text="Activity log", style="Accent.TLabel")
        self.log_header.grid(row=0, column=0, sticky="w")
        self.log_console = scrolledtext.ScrolledText(
            self.logs_frame, height=8, wrap=tk.WORD, font=("Segoe UI", self._s(11)), highlightthickness=0
        )
        self.log_console.grid(row=1, column=0, sticky="nsew", pady=(6, 0))
        self._init_style()
        self.csv_file_var = tk.StringVar(value='data/data.csv')
        self.model_file_var = tk.StringVar(value='models/model.h5')
        self.scaler_file_var = tk.StringVar(value='other/scaler.pkl')
        self.images_dir = 'images'
        self._prepare_directories_and_csv()
        self.current_label = None
        self.flip_vertical = False
        self.flip_horizontal = False
        self.new_index = None
        self.default_static_image_mode = False
        self.default_max_num_hands = 1
        self.default_model_complexity = 1
        self.default_min_detection_confidence = 0.5
        self.default_min_tracking_confidence = 0.5

        self.static_image_mode_var = tk.BooleanVar(value=self.default_static_image_mode)
        self.max_num_hands_var = tk.IntVar(value=self.default_max_num_hands)
        self.model_complexity_var = tk.IntVar(value=self.default_model_complexity)
        self.min_detection_confidence_var = tk.IntVar(value=int(self.default_min_detection_confidence * 100))
        self.min_tracking_confidence_var = tk.IntVar(value=int(self.default_min_tracking_confidence * 100))
        self.hands = None
        self._init_mediapipe_hands()
        self.available_cameras = self._detect_cameras(max_cameras=5)
        if not self.available_cameras:
            raise RuntimeError(tr("err_no_camera"))
        self.current_camera_index = self.available_cameras[0]
        self.camera_var = tk.StringVar(value=str(self.current_camera_index))
        gui_collect.create_collect_tab(self)
        gui_train.create_train_tab(self)
        gui_detection.create_detection_tab(self)
        gui_text_detection.create_text_detection_tab(self)
        gui_instructions.create_instructions_tab(self)
        self.cap = None
        self.root.after_idle(self._init_first_camera)
        self.root.bind_all("<space>", self._on_space_or_enter)
        self.root.bind_all("<Return>", self._on_space_or_enter)
        self.detection_running = False
        self.text_detection_running = False
        self.update_frame()

    def _init_style(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        self.style = style
        self._apply_theme(self.theme_var.get())

    def _compute_ui_scale(self) -> float:
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        base_w, base_h = 1366, 900
        scale = min(screen_w / base_w, screen_h / base_h)
        # Do not shrink below native size; allow a small bump on high-DPI displays.
        return max(1.0, min(scale, 1.1))

    def _s(self, value: float) -> int:
        return max(1, int(round(value * getattr(self, "ui_scale", 1.0))))

    def _apply_theme(self, mode):
        palettes = {
            "light": {
                "bg": "#f4f6fb",
                "fg": "#111827",
                "accent": "#2563eb",
                "panel": "#ffffff",
                "muted": "#6b7280",
            },
            "dark": {
                "bg": "#0f172a",
                "fg": "#e5e7eb",
                "accent": "#38bdf8",
                "panel": "#111827",
                "muted": "#9ca3af",
            },
            "high-contrast": {
                "bg": "#000000",
                "fg": "#ffffff",
                "accent": "#ffdd00",
                "panel": "#0a0a0a",
                "muted": "#f5f5f5",
            },
        }
        colors = palettes.get(mode, palettes["light"])
        base_font = ("Segoe UI", self._s(12))
        accent = colors["accent"]
        bg = colors["bg"]
        fg = colors["fg"]
        panel = colors["panel"]
        muted = colors["muted"]
        self.root.configure(bg=bg)
        self.style.configure("TFrame", background=bg)
        self.style.configure("TLabel", background=bg, foreground=fg, font=base_font)
        self.style.configure("Accent.TLabel", background=bg, foreground=accent, font=("Segoe UI Semibold", self._s(13)))
        self.style.configure("TButton", font=base_font, padding=(self._s(12), self._s(8)), background=panel, foreground=fg)
        self.style.configure("TCheckbutton", background=bg, foreground=fg, font=base_font)
        self.style.configure("TRadiobutton", background=bg, foreground=fg, font=base_font)

        # Inputs / groups
        self.style.configure(
            "TEntry",
            fieldbackground=panel,
            foreground=fg,
            padding=(self._s(8), self._s(6)),
        )
        self.style.map(
            "TEntry",
            foreground=[("disabled", muted)],
            fieldbackground=[("disabled", bg), ("readonly", panel)],
        )

        self.style.configure(
            "TCombobox",
            fieldbackground=panel,
            background=panel,
            foreground=fg,
            arrowcolor=fg,
            padding=(self._s(8), self._s(6)),
        )
        self.style.map(
            "TCombobox",
            foreground=[("disabled", muted), ("readonly", fg)],
            fieldbackground=[("disabled", bg), ("readonly", panel)],
            selectbackground=[("readonly", accent)],
            selectforeground=[("readonly", bg)],
        )

        self.style.configure(
            "TLabelframe",
            background=bg,
            padding=(self._s(10), self._s(10)),
        )
        self.style.configure(
            "TLabelframe.Label",
            background=bg,
            foreground=fg,
            font=("Segoe UI Semibold", self._s(12)),
        )

        self.style.configure("TScale", background=bg, troughcolor=panel)
        self.style.configure("Horizontal.TScale", background=bg, troughcolor=panel)
        self.style.configure("Vertical.TScale", background=bg, troughcolor=panel)

        self.style.configure("TProgressbar", troughcolor=panel, background=accent)
        self.style.configure("TScrollbar", troughcolor=panel)
        self.style.configure("TSeparator", background=panel)

        self.style.configure("TNotebook", background=bg, padding=self._s(4))
        self.style.configure("TNotebook.Tab", padding=(self._s(16), self._s(10)), font=("Segoe UI Semibold", self._s(12)), background=panel, foreground=fg)
        self.style.map("TNotebook.Tab", padding=[("selected", (self._s(18), self._s(12)))], background=[("selected", accent)], foreground=[("selected", bg)])

        # Buttons: give a clear hover/pressed state for dark/high-contrast.
        self.style.map(
            "TButton",
            foreground=[("disabled", muted), ("active", bg), ("pressed", bg)],
            background=[("active", accent), ("pressed", accent)],
        )
        for widget in (self.top_bar, self.content_frame, self.logs_frame, getattr(self, "content_scroller", None)):
            if widget:
                widget.configure(style="TFrame")

        # Canvas background (scrollable area) must match theme too.
        if hasattr(self, "content_scroller") and hasattr(self.content_scroller, "canvas"):
            try:
                self.content_scroller.canvas.configure(bg=bg)
            except tk.TclError:
                pass

        # Theme tk.Text-based widgets (ScrolledText uses a tk.Text under the hood)
        def _theme_text_widget(w):
            if not w:
                return
            try:
                w.configure(
                    bg=panel,
                    fg=fg,
                    insertbackground=fg,
                    selectbackground=accent,
                    selectforeground=bg,
                    highlightthickness=0,
                )
            except tk.TclError:
                pass

        _theme_text_widget(getattr(self, "log_console", None))
        _theme_text_widget(getattr(self, "det_text", None))
        _theme_text_widget(getattr(self, "text_detection_text", None))

    def change_theme(self, *_):
        self._apply_theme(self.theme_var.get())
    def restart_camera(self):
        # Stop any running loops first so they won't read from a released capture.
        if getattr(self, "detection_running", False):
            self.detection_running = False
            if hasattr(self, "det_start_btn"):
                self.det_start_btn.config(state="normal")
            if hasattr(self, "det_stop_btn"):
                self.det_stop_btn.config(state="disabled")

        if getattr(self, "text_detection_running", False):
            self.text_detection_running = False
            if hasattr(self, "td_start_btn"):
                self.td_start_btn.config(state="normal")
            if hasattr(self, "td_stop_btn"):
                self.td_stop_btn.config(state="disabled")

        if hasattr(self, 'cap') and self.cap and self.cap.isOpened():
            self.cap.release()
        self.cap = self.open_camera(self.current_camera_index)
        self.log(tr("log_camera_restarted"))
    def on_tab_changed(self, event):
        new = event.widget.index("current")
        old = self.last_tab_index

        # Keep the camera open when leaving Collect; we reuse it for preview
        # in other tabs.

        if old == self.notebook.index(self.tab_detection) and self.detection_running:
            self.detection_running = False
            self.det_start_btn.config(state="normal")
            self.det_stop_btn.config(state="disabled")

        if old == self.notebook.index(self.tab_text_detection) and getattr(self, "text_detection_running", False):
            self.text_detection_running = False
            if hasattr(self, "td_start_btn"):
                self.td_start_btn.config(state="normal")
            if hasattr(self, "td_stop_btn"):
                self.td_stop_btn.config(state="disabled")

        # Ensure the camera is open for tabs that can show a preview.
        if (new in (
                self.notebook.index(self.tab_collect),
                self.notebook.index(self.tab_detection),
                self.notebook.index(self.tab_text_detection),
            )
                and not getattr(self, "detection_running", False)
                and not getattr(self, "text_detection_running", False)
                and (self.cap is None or not self.cap.isOpened())):
            self.cap = self.open_camera(self.current_camera_index)

        self.last_tab_index = new

    def open_camera(self, index):
        dlg = tk.Toplevel(self.root)
        dlg.title(tr("wait_window_title")) 
        dlg.transient(self.root)
        
        lbl = tk.Label(dlg, text=tr("msg_wait_camera"))
        lbl.pack(padx=20, pady=20)

        self.root.update_idletasks()
        dlg.update_idletasks()

        w, h = dlg.winfo_width(), dlg.winfo_height()
        rw, rh = self.root.winfo_width(), self.root.winfo_height()
        if rw < 50 or rh < 50:
            sx = self.root.winfo_screenwidth()
            sy = self.root.winfo_screenheight()
            x = (sx - w) // 2
            y = (sy - h) // 2
        else:
            rx, ry = self.root.winfo_rootx(), self.root.winfo_rooty()
            x = rx + (rw - w) // 2
            y = ry + (rh - h) // 2

        dlg.geometry(f"{w}x{h}+{x}+{y}")

        def safe_grab():
            if dlg.winfo_viewable():
                try:
                    dlg.grab_set()
                except tk.TclError:
                    pass
            else:
                dlg.after(50, safe_grab)

        dlg.after(50, safe_grab)

        cap = open_capture(index)
        dlg.update()
        dlg.destroy()
        return cap


    def _init_first_camera(self):
        if self.cap is None:
            self.cap = self.open_camera(self.current_camera_index)
    # ------------------------------------------------------------------
    # language switching
    # ------------------------------------------------------------------
    def change_language(self, *_):
        new_lang = self.language_var.get()
        set_language(new_lang)
        self.refresh_ui_texts()

    def refresh_ui_texts(self):
        self.root.title(tr("main_window_title")) 

        if hasattr(self, "lang_label"):
            self.lang_label.config(text=tr("language_label"))
        if hasattr(self, "theme_label"):
            self.theme_label.config(text=tr("theme_label"))

        self.notebook.tab(self.tab_collect,        text=tr("tab_collect"))
        self.notebook.tab(self.tab_train,          text=tr("tab_train"))
        self.notebook.tab(self.tab_detection,      text=tr("tab_detection"))
        self.notebook.tab(self.tab_text_detection, text=tr("tab_text"))
        self.notebook.tab(self.tab_instructions,   text=tr("tab_instr"))
        if hasattr(self, "col_lbl_choose_cam"):
            self.col_lbl_choose_cam.config(text=tr("lbl_choose_camera"))
        if hasattr(self, "col_lbl_enter_label"):
            self.col_lbl_enter_label.config(text=tr("lbl_enter_label"))
        if hasattr(self, "col_btn_set_label"):
            self.col_btn_set_label.config(text=tr("btn_set_label"))
        if hasattr(self, "col_btn_save"):
            self.col_btn_save.config(text=tr("btn_save_data"))
        if hasattr(self, "col_btn_flip"):
            self.col_btn_flip.config(text=tr("btn_flip"))
        if hasattr(self, "col_section_data"):
            self.col_section_data.config(text=tr("section_data_mgmt"))
        if hasattr(self, "col_btn_clear_images"):
            self.col_btn_clear_images.config(text=tr("btn_clear_images"))
        if hasattr(self, "col_btn_clear_csv"):
            self.col_btn_clear_csv.config(text=tr("btn_reset_csv"))
        if hasattr(self, "col_section_reset"):
            self.col_section_reset.config(text=tr("section_reset"))
        if hasattr(self, "col_btn_reset_defaults"):
            self.col_btn_reset_defaults.config(text=tr("btn_reset_defaults"))
        if hasattr(self, "col_btn_quit"):
            self.col_btn_quit.config(text=tr("btn_quit"))
        if hasattr(self, "col_section_img"):
            self.col_section_img.config(text=tr("section_img_settings"))
        if hasattr(self, "col_lbl_brightness"):
            self.col_lbl_brightness.config(text=tr("lbl_brightness"))
        if hasattr(self, "col_lbl_contrast"):
            self.col_lbl_contrast.config(text=tr("lbl_contrast"))
        if hasattr(self, "col_lbl_gamma"):
            self.col_lbl_gamma.config(text=tr("lbl_gamma"))
        if hasattr(self, "col_lbl_color_shift"):
            self.col_lbl_color_shift.config(text=tr("lbl_color_shift"))
        if hasattr(self, "col_lbl_R"):
            self.col_lbl_R.config(text=tr("lbl_R"))
        if hasattr(self, "col_lbl_G"):
            self.col_lbl_G.config(text=tr("lbl_G"))
        if hasattr(self, "col_lbl_B"):
            self.col_lbl_B.config(text=tr("lbl_B"))

        if hasattr(self, "mp_section_frame"):
            self.mp_section_frame.config(text=tr("section_mediapipe"))
        if hasattr(self, "mp_chk_static"):
            self.mp_chk_static.config(text=tr("chk_static_img_mode"))
        if hasattr(self, "mp_lbl_max_hands"):
            self.mp_lbl_max_hands.config(text=tr("lbl_max_num_hands"))
        if hasattr(self, "mp_lbl_model_complex"):
            self.mp_lbl_model_complex.config(text=tr("lbl_model_complexity"))
        if hasattr(self, "mp_lbl_min_det"):
            self.mp_lbl_min_det.config(text=tr("lbl_min_det_conf"))
        if hasattr(self, "mp_lbl_min_track"):
            self.mp_lbl_min_track.config(text=tr("lbl_min_track_conf"))
        if hasattr(self, "mp_btn_apply"):
            self.mp_btn_apply.config(text=tr("btn_apply_mp"))
        if hasattr(self, "mp_chk_show_overlays"):
            self.mp_chk_show_overlays.config(text=tr("chk_show_overlays"))
        if hasattr(self, "mp_file_frame"):
            self.mp_file_frame.config(text=tr("frame_file_labels"))
        if hasattr(self, "mp_lbl_csv"):
            self.mp_lbl_csv.config(text=tr("lbl_csv_file"))
        if hasattr(self, "mp_lbl_model"):
            self.mp_lbl_model.config(text=tr("lbl_model_file"))
        if hasattr(self, "mp_lbl_scaler"):
            self.mp_lbl_scaler.config(text=tr("lbl_scaler_file"))

        if hasattr(self, "train_frame_label"):
            self.train_frame_label.config(text=tr("frame_train_config"))
        if hasattr(self, "train_lbl_test_size"):
            self.train_lbl_test_size.config(text=tr("lbl_test_size"))
        if hasattr(self, "train_lbl_random_state"):
            self.train_lbl_random_state.config(text=tr("lbl_random_state"))
        if hasattr(self, "train_lbl_epochs"):
            self.train_lbl_epochs.config(text=tr("lbl_epochs"))
        if hasattr(self, "train_lbl_batch_size"):
            self.train_lbl_batch_size.config(text=tr("lbl_batch_size"))
        if hasattr(self, "train_lbl_patience"):
            self.train_lbl_patience.config(text=tr("lbl_patience"))
        if hasattr(self, "train_btn_start"):
            self.train_btn_start.config(text=tr("btn_start_training"))
        if hasattr(self, "plot_frame"):
            self.plot_frame.config(text=tr("frame_train_plots"))
        if hasattr(self, "val_split_label"):
            self.val_split_label.config(text=tr("lbl_validation_split"))
        if hasattr(self, "monitor_label"):
            self.monitor_label.config(text=tr("lbl_monitor"))

        if hasattr(self, "det_interval_label"):
            self.det_interval_label.config(text=tr("lbl_interval"))
        if hasattr(self, "det_threshold_label"):
            self.det_threshold_label.config(text=tr("lbl_threshold"))
        if hasattr(self, "det_enter_chk"):
            self.det_enter_chk.config(text=tr("chk_enter_mode"))
        if hasattr(self, "det_start_btn"):
            self.det_start_btn.config(text=tr("btn_start_detection"))
        if hasattr(self, "det_stop_btn"):
            self.det_stop_btn.config(text=tr("btn_stop_detection"))
        if hasattr(self, "det_clear_btn"):
            self.det_clear_btn.config(text=tr("btn_clear_screen"))

        if hasattr(self, "txt_interval_label"):
            self.txt_interval_label.config(text=tr("lbl_interval"))
        if hasattr(self, "txt_threshold_label"):
            self.txt_threshold_label.config(text=tr("lbl_threshold"))
        if hasattr(self, "txt_lbl_file"):
            self.txt_lbl_file.config(text=tr("lbl_select_text_file"))
        if hasattr(self, "txt_load_btn"):
            self.txt_load_btn.config(text=tr("btn_load_text"))
        if hasattr(self, "txt_start_btn"):
            self.txt_start_btn.config(text=tr("btn_start"))
        if hasattr(self, "txt_stop_btn"):
            self.txt_stop_btn.config(text=tr("btn_stop"))
        if hasattr(self, "restart_cam_btn"):
            self.restart_cam_btn.config(text=tr("btn_restart_camera"))
        if hasattr(self, "det_restart_cam_btn"):
            self.det_restart_cam_btn.config(text=tr("btn_restart_camera"))
        if hasattr(self, "text_restart_cam_btn"):
            self.text_restart_cam_btn.config(text=tr("btn_restart_camera"))
            

        if hasattr(self, "instructions_box"):
            self.instructions_box.config(state=tk.NORMAL)
            self.instructions_box.delete("1.0", tk.END)
            self.instructions_box.insert(tk.END, tr("instructions_text"))
            self.instructions_box.config(state=tk.DISABLED)
        if hasattr(self, "_i18n_widgets_text"):
             for widget, key in self._i18n_widgets_text:
                 widget.config(text=tr(key))


    def _prepare_directories_and_csv(self):
        csv_path = self.csv_file_var.get()
        if not os.path.exists(os.path.dirname(csv_path)):
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        if not os.path.exists(self.images_dir):
            os.makedirs(self.images_dir)
        if not os.path.exists(csv_path):
            with open(csv_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                header = []
                for i in range(21):
                    header += [f'x{i}', f'y{i}']
                header += ['label', 'index']
                writer.writerow(header)


    def log(self, text):
        self.log_console.insert(tk.END, text + "\n")
        self.log_console.see(tk.END)
        print(text)
        if self.log_file:
            self.log_file.write(text + "\n")
            self.log_file.flush()

    def _init_mediapipe_hands(self):
        if self.hands is not None:
            self.hands.close()
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=self.static_image_mode_var.get(),
            max_num_hands=self.max_num_hands_var.get(),
            model_complexity=self.model_complexity_var.get(),
            min_detection_confidence=float(self.min_detection_confidence_var.get()) / 100.0,
            min_tracking_confidence=float(self.min_tracking_confidence_var.get()) / 100.0,
        )

    def _detect_cameras(self, max_cameras=5):
        cameras = []
        for i in range(max_cameras):
            temp_cap = cv2.VideoCapture(i)
            if temp_cap.isOpened():
                ret, _ = temp_cap.read()
                if ret:
                    cameras.append(i)
                temp_cap.release()
        return cameras

    def update_frame(self):
        current_tab_index = self.notebook.index(self.notebook.select())
        # Collect tab (0) - show preview with overlays and image adjustments
        if (
            current_tab_index == 0
            and self.cap
            and self.cap.isOpened()
            and not self.detection_running
            and not getattr(self, "text_detection_running", False)
        ):
            ret, frame = self.cap.read()
            if ret:
                if self.flip_horizontal:
                    frame = cv2.flip(frame, 1)
                if self.flip_vertical:
                    frame = cv2.flip(frame, 0)
                frame = self.apply_image_adjustments(frame)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(frame_rgb)

                draw_helpers = self.show_overlays_var.get()

                if draw_helpers and results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp.solutions.drawing_utils.draw_landmarks(
                            frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS
                        )

                if draw_helpers and self.current_label is not None:
                    cv2.putText(
                        frame,
                        tr("lbl_current_label", val=self.current_label),
                        (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

                if draw_helpers:
                    index_display = (
                        str(self.new_index) if self.new_index is not None else 'N/A'
                    )
                    cv2.putText(
                        frame,
                        tr("lbl_current_index", val=index_display),
                        (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                    )

                self.last_frame = frame.copy()
                frame_rgb_for_tk = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb_for_tk)
                imgtk = ImageTk.PhotoImage(image=img)
                self.camera_label.imgtk = imgtk
                self.camera_label.configure(image=imgtk)

        # Detection tab - show preview even before pressing Start
        if (
            hasattr(self, "det_camera_label")
            and current_tab_index == self.notebook.index(self.tab_detection)
            and self.cap
            and self.cap.isOpened()
            and not self.detection_running
        ):
            ret, frame = self.cap.read()
            if ret:
                if self.flip_horizontal:
                    frame = cv2.flip(frame, 1)
                if self.flip_vertical:
                    frame = cv2.flip(frame, 0)
                imgtk = ImageTk.PhotoImage(
                    image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                )
                self.det_camera_label.imgtk = imgtk
                self.det_camera_label.configure(image=imgtk)

        # Text signing tab - show preview even before starting text detection
        if (
            hasattr(self, "text_det_camera_label")
            and current_tab_index == self.notebook.index(self.tab_text_detection)
            and self.cap
            and self.cap.isOpened()
            and not getattr(self, "text_detection_running", False)
        ):
            ret, frame = self.cap.read()
            if ret:
                if self.flip_horizontal:
                    frame = cv2.flip(frame, 1)
                if self.flip_vertical:
                    frame = cv2.flip(frame, 0)
                imgtk = ImageTk.PhotoImage(
                    image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                )
                self.text_det_camera_label.imgtk = imgtk
                self.text_det_camera_label.configure(image=imgtk)

        self.root.after(20, self.update_frame)

    def apply_image_adjustments(self, frame):
        frame = frame.astype(np.float32)
        beta = float(self.brightness_var.get())
        alpha_percent = float(self.contrast_var.get())
        alpha = max(0.01, alpha_percent / 100.0)
        frame = frame * alpha + beta
        shift_r = float(self.r_var.get())
        shift_g = float(self.g_var.get())
        shift_b = float(self.b_var.get())
        frame[:, :, 2] += shift_r
        frame[:, :, 1] += shift_g
        frame[:, :, 0] += shift_b
        frame = np.clip(frame, 0, 255)
        frame = frame / 255.0
        real_gamma = max(0.01, float(self.gamma_var.get()) / 100.0)
        frame = np.power(frame, 1.0 / real_gamma)
        frame = frame * 255.0
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        return frame

    def quit_app(self):
        self.log(tr("log_closing"))
        if self.cap:
            self.cap.release()
        if self.log_file:
            self.log_file.close()
        self.root.destroy()

    def on_camera_select(self, event=None):
        new_index = int(self.camera_combo.get())
        if new_index != self.current_camera_index:
            self.log(tr("log_camera_switch", old=self.current_camera_index, new=new_index))
            self.cap.release()
            self.cap = self.open_camera(new_index)
            self.current_camera_index = new_index

    def set_label(self):
        label_text = self.label_entry.get().strip()
        if label_text:
            self.current_label = label_text.upper()
            self.log(tr("log_label_selected", val=self.current_label))
        else:
            self.log(tr("log_no_label"))

    def toggle_flip(self):
        self.flip_vertical = not self.flip_vertical
        status = tr("status_on") if self.flip_vertical else tr("status_off")
        self.log(tr("log_flip_status", val=status))
    def toggle_flip_horizontal(self):
        self.flip_horizontal = not self.flip_horizontal
        status = "ON" if self.flip_horizontal else "OFF"
        self.log(f"Przerzucanie obrazu poziomo: {status}")

    def toggle_flip_vertical(self):
        self.flip_vertical = not self.flip_vertical
        status = "ON" if self.flip_vertical else "OFF"
        self.log(f"Przerzucanie obrazu pionowo: {status}")

    def _on_space_or_enter(self, event):
        current_tab_index = self.notebook.index(self.notebook.select())
        if current_tab_index == 0:
            self.save_data()

    def update_scale_label(self, variable: tk.IntVar, label: ttk.Label):
        label.config(text=str(variable.get()))

    def reset_to_defaults(self):
        self.brightness_var.set(0)
        self.contrast_var.set(100)
        self.gamma_var.set(100)
        self.r_var.set(0)
        self.g_var.set(0)
        self.b_var.set(0)
        self.static_image_mode_var.set(self.default_static_image_mode)
        self.max_num_hands_var.set(self.default_max_num_hands)
        self.model_complexity_var.set(self.default_model_complexity)
        self.min_detection_confidence_var.set(int(self.default_min_detection_confidence * 100))
        self.min_tracking_confidence_var.set(int(self.default_min_tracking_confidence * 100))

        self.update_scale_label(self.brightness_var, self.brightness_value_label)
        self.update_scale_label(self.contrast_var, self.contrast_value_label)
        self.update_scale_label(self.gamma_var, self.gamma_value_label)
        self.update_scale_label(self.r_var, self.r_value_label)
        self.update_scale_label(self.g_var, self.g_value_label)
        self.update_scale_label(self.b_var, self.b_value_label)
        self.update_scale_label(self.max_num_hands_var, self.max_hands_label)
        self.update_scale_label(self.model_complexity_var, self.model_complexity_label)
        self.update_scale_label(self.min_detection_confidence_var, self.min_detection_label)
        self.update_scale_label(self.min_tracking_confidence_var, self.min_tracking_label)

        self._init_mediapipe_hands()
        self.log(tr("log_reset_defaults"))

    def update_mediapipe_settings(self):
        self._init_mediapipe_hands()
        self.log(tr("log_mp_updated"))

    def start_training(self):
        self.progress_var.set(0.0)
        t = threading.Thread(target=training.run_training_in_thread, args=(self,))
        t.start()

    def start_detection(self):
        if self.detection_running:
            return
        self.detection_running = True
        self.det_start_btn.config(state="disabled")
        self.det_stop_btn.config(state="normal")

        t = threading.Thread(target=detection.run_detection, args=(self,))
        t.start()

    def stop_detection(self):
        self.detection_running = False
        self.det_start_btn.config(state="normal")
        self.det_stop_btn.config(state="disabled")

    def save_data(self):
        if self.current_label is None:
            self.log(tr("log_first_set_label"))
            return
        if not hasattr(self, 'last_frame'):
            self.log(tr("log_no_camera_data"))
            return
        frame_rgb = cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            row = []
            for lm in hand.landmark:
                row.append(lm.x)
                row.append(lm.y)
            label_dir = os.path.join(self.images_dir, self.current_label)
            if not os.path.exists(label_dir):
                os.makedirs(label_dir)
            max_index = -1
            for file_name in os.listdir(label_dir):
                if file_name.startswith(f"{self.current_label}_") and file_name.endswith(".jpg"):
                    try:
                        number_part = file_name[len(f"{self.current_label}_"):-4]
                        num = int(number_part)
                        if num > max_index:
                            max_index = num
                    except ValueError:
                        pass
            new_index = max_index + 1
            self.new_index = new_index
            row.append(self.current_label)
            row.append(str(new_index))
            csv_path = self.csv_file_var.get()
            with open(csv_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
            with open(csv_path, 'r') as f:
                total_lines = sum(1 for _ in f) - 1
            self.log("#" * 20)
            self.log(tr("log_saved_sample", label=self.current_label, idx=new_index, path=csv_path))
            self.log(tr("log_csv_total", total=total_lines))
            image_filename = f"{self.current_label}_{new_index}.jpg"
            image_path = os.path.join(label_dir, image_filename)
            cv2.imwrite(image_path, self.last_frame)
            self.log(tr("log_saved_image", path=image_path))
            count_files = len([
                fname for fname in os.listdir(label_dir)
                if fname.startswith(f"{self.current_label}_") and fname.endswith(".jpg")
            ])
            self.log(tr("log_folder_count", label=self.current_label, count=count_files))
            self.log("#" * 20)
        else:
            self.log(tr("log_no_hand"))

    def clear_images(self):
        if not os.path.exists(self.images_dir):
            self.log(tr("log_images_empty"))
            return
        confirm = messagebox.askyesno(tr("dlg_confirm"), tr("dlg_sure_clear_images"))
        if confirm:
            import shutil
            shutil.rmtree(self.images_dir)
            os.makedirs(self.images_dir)
            self.log(tr("log_images_cleared"))
        else:
            self.log(tr("log_action_cancelled"))

    def clear_csv(self):
        csv_path = self.csv_file_var.get()
        if not os.path.exists(csv_path):
            self.log(tr("log_csv_missing", path=csv_path))
            return
        confirm = messagebox.askyesno(tr("dlg_confirm"), tr("dlg_sure_reset_csv", file=csv_path))
        if confirm:
            with open(csv_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                header = []
                for i in range(21):
                    header += [f'x{i}', f'y{i}']
                header += ['label', 'index']
                writer.writerow(header)
            self.log(tr("log_csv_reset", path=csv_path))
        else:
            self.log(tr("log_action_cancelled"))
    def show_training_plots(self, history):
        for child in self.plot_frame.winfo_children():
            child.destroy()

        fig = Figure(figsize=(5, 4))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        ax1.plot(history.history['accuracy'], label='train acc')
        ax1.plot(history.history['val_accuracy'], label='val acc')
        ax1.set_title('Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.legend()

        ax2.plot(history.history['loss'], label='train loss')
        ax2.plot(history.history['val_loss'], label='val loss')
        ax2.set_title('Loss')
        ax2.set_xlabel('Epoch')
        ax2.legend()

        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

def open_capture(index: int, warmup_sec: float = 2.0) -> cv2.VideoCapture:
    sys = platform.system()
    if sys == "Windows":
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    elif sys == "Linux":
        backends = [cv2.CAP_V4L2, cv2.CAP_ANY]
    else:
        backends = [cv2.CAP_ANY]

    for be in backends:
        cap = cv2.VideoCapture(index, be)
        if not cap.isOpened():
            cap.release()
            continue

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        t0 = time.time()
        while time.time() - t0 < warmup_sec:
            ok, _ = cap.read()
            if ok:
                return cap
            time.sleep(0.05)

        cap.release()

    raise RuntimeError(f"Cannot open camera index {index}")
def main():
    root = tk.Tk()
    app = HandDataCollectorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
