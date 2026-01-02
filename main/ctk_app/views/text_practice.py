from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import customtkinter as ctk

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

from locales import tr

from ctk_app.common import (
    _appearance_index,
    _letterbox_to,
    _parse_camera_index,
    _theme_frame_bg_hex,
    _theme_overlay_bg_hex,
    _theme_textbox_bg_hex,
    _theme_textbox_fg_hex,
    _theme_value,
)

class TextPracticeView(ctk.CTkFrame):
    def __init__(self, master, app: App):
        super().__init__(master)
        self.app = app

        self._content: str = ""
        self._idx: int = 0
        self._ok: int = 0
        self._fail: int = 0
        self._running: bool = False
        self._enabled: bool = False
        self._detection_ready: bool = False
        self._view_key: str = "tekst"
        self._prereq_after_id: str | None = None
        self._last_pred_ts: float = 0.0
        self.interval_ms = tk.IntVar(value=50)
        self.threshold = tk.DoubleVar(value=0.7)

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        root = ctk.CTkFrame(self)
        root.grid(row=0, column=0, sticky="nsew", padx=18, pady=18)
        root.grid_rowconfigure(0, weight=1)
        root.grid_columnconfigure(0, weight=1)

        self._bg_hex = _theme_frame_bg_hex()
        self.paned = tk.PanedWindow(
            root,
            orient="horizontal",
            bd=0,
            sashwidth=6,
            sashrelief="raised",
            bg=self._bg_hex,
        )
        self.paned.grid(row=0, column=0, sticky="nsew")

        left = ctk.CTkFrame(self.paned)
        right_container = ctk.CTkFrame(self.paned)
        self._left_pane = left
        self._right_pane = right_container
        self.paned.add(left, minsize=int(self.app.camera_pane_minsize()))
        self.paned.add(right_container, minsize=320)

        left.grid_rowconfigure(0, weight=1)
        left.grid_columnconfigure(0, weight=1)
        self.video_area = ctk.CTkFrame(left)
        self.video_area.grid(row=0, column=0, sticky="nsew")
        self.video_area.grid_rowconfigure(0, weight=1)
        self.video_area.grid_columnconfigure(0, weight=1)
        self.video_label = tk.Label(self.video_area, text=tr("cam_preview_placeholder"), bg=self._bg_hex)
        self.video_label.grid(row=0, column=0, sticky="nsew", padx=16, pady=16)
        self._imgtk: ImageTk.PhotoImage | None = None
        self._img_size: tuple[int, int] | None = None

        right_container.grid_rowconfigure(0, weight=1)
        right_container.grid_columnconfigure(0, weight=1)
        right = ctk.CTkScrollableFrame(right_container)
        right.grid(row=0, column=0, sticky="nsew")
        right.grid_rowconfigure(3, weight=1)
        right.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(right, text=tr("tab_text"), font=ctk.CTkFont(size=18, weight="bold")).grid(
            row=0, column=0, sticky="w", padx=14, pady=(14, 8)
        )

        self.text_files_dir = Path("text_files")
        files = []
        if self.text_files_dir.is_dir():
            files = [p.name for p in self.text_files_dir.glob("*.txt")]
        self.file_menu = ctk.CTkOptionMenu(right, values=files or ["(brak plików)"])
        self.file_menu.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 8))

        controls = ctk.CTkFrame(right, fg_color="transparent")
        controls.grid(row=2, column=0, sticky="ew", padx=14, pady=(0, 8))
        controls.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(controls, text=tr("lbl_interval").rstrip(":")).grid(row=0, column=0, sticky="w")
        self.interval_str = tk.StringVar(value=str(self.interval_ms.get()))
        ctk.CTkEntry(controls, textvariable=self.interval_str, width=90).grid(row=0, column=1, sticky="ew", padx=(8, 0))

        ctk.CTkLabel(controls, text=tr("lbl_threshold").rstrip(":")).grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.threshold_str = tk.StringVar(value=str(self.threshold.get()))
        ctk.CTkEntry(controls, textvariable=self.threshold_str, width=90).grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=(6, 0))

        btns = ctk.CTkFrame(right, fg_color="transparent")
        btns.grid(row=3, column=0, sticky="ew", padx=14, pady=(0, 8))
        btns.grid_columnconfigure(0, weight=1)
        btns.grid_columnconfigure(1, weight=1)
        btns.grid_columnconfigure(2, weight=1)

        self.load_btn = ctk.CTkButton(btns, text=f"↑ {tr('btn_load_text')}", command=self._load_text)
        self.load_btn.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self.start_btn = ctk.CTkButton(btns, text=f"▶ {tr('btn_start')}", command=self._start)
        self.start_btn.grid(row=0, column=1, sticky="ew", padx=6)
        self.stop_btn = ctk.CTkButton(btns, text=f"■ {tr('btn_stop')}", command=self._stop)
        self.stop_btn.grid(row=0, column=2, sticky="ew", padx=(6, 0))

        cam_row = ctk.CTkFrame(right, fg_color="transparent")
        cam_row.grid(row=4, column=0, sticky="ew", padx=14, pady=(0, 8))
        cam_row.grid_columnconfigure(2, weight=1)
        cam_values = [f"Cam {i}" for i in app.available_cameras] if app.available_cameras else ["Cam 0"]
        self.cam_menu = ctk.CTkOptionMenu(cam_row, values=cam_values, width=70, command=self._on_camera_changed)
        self.cam_menu.grid(row=0, column=0, padx=(0, 6))
        self.cam_menu.set(f"Cam {app.camera_index_var.get()}")
        ctk.CTkButton(cam_row, text=f"↻ {tr('btn_restart')}", width=100, command=self._restart_camera).grid(row=0, column=1, padx=6)
        ctk.CTkButton(cam_row, text=f"↔ {tr('btn_flip_h')}", width=90, command=self._flip_h).grid(row=0, column=2, padx=6, sticky="w")
        ctk.CTkButton(cam_row, text=f"↕ {tr('btn_flip_v')}", width=90, command=self._flip_v).grid(row=0, column=3, padx=(6, 0), sticky="w")

        overlay_row = ctk.CTkFrame(right, fg_color="transparent")
        overlay_row.grid(row=5, column=0, sticky="ew", padx=14, pady=(0, 8))
        overlay_row.grid_columnconfigure(0, weight=1)
        overlay_row.grid_columnconfigure(1, weight=0)

        self.overlay_switch = ctk.CTkSwitch(overlay_row, text=tr("chk_overlay"))
        self.overlay_switch.grid(row=0, column=0, sticky="w")
        if app.show_overlays:
            self.overlay_switch.select()
        else:
            self.overlay_switch.deselect()

        self.hands_var = tk.StringVar(value=tr("lbl_hands_detected", n=0))
        self.hands_label = ctk.CTkLabel(overlay_row, textvariable=self.hands_var)
        self.hands_label.grid(row=0, column=1, sticky="e")

        self.latency_var = tk.StringVar(value=tr("lbl_last_classification_times", mp_ms="—", model_ms="—", total_ms="—"))
        self.latency_label = ctk.CTkLabel(right, textvariable=self.latency_var)
        self.latency_label.grid(row=6, column=0, sticky="w", padx=14, pady=(0, 8))

        self.text_widget = tk.Text(right, wrap="word", height=10)
        self.text_widget.grid(row=7, column=0, sticky="nsew", padx=14, pady=(0, 8))
        self._apply_text_widget_theme()
        self.text_widget.configure(state="disabled")
        self.text_widget.tag_config("correct", underline=1)

        ctk.CTkLabel(
            right,
            text=tr("lbl_top10"),
            font=ctk.CTkFont(size=13, weight="bold"),
        ).grid(row=8, column=0, sticky="w", padx=14, pady=(0, 6))

        self.topk = ctk.CTkTextbox(right, height=160, font=("Consolas", 11))
        self.topk.grid(row=9, column=0, sticky="ew", padx=14, pady=(0, 8))
        self.topk.configure(state="disabled")

        self.stats = ctk.CTkLabel(right, text="", justify="left")
        self.stats.grid(row=10, column=0, sticky="ew", padx=14, pady=(0, 14))

        self._blocker = ctk.CTkFrame(self, corner_radius=0, fg_color=_theme_overlay_bg_hex())
        self._blocker.place(relx=0, rely=0, relwidth=1, relheight=1)
        self._blocker.grid_columnconfigure(0, weight=1)
        self._blocker.grid_rowconfigure(0, weight=1)
        card = ctk.CTkFrame(self._blocker)
        card.grid(row=0, column=0, padx=28, pady=28, sticky="n")
        card.grid_columnconfigure(0, weight=1)

        self._blocker_title = ctk.CTkLabel(card, text=tr("det_missing_files_title"), font=ctk.CTkFont(size=18, weight="bold"))
        self._blocker_title.grid(row=0, column=0, sticky="w", padx=18, pady=(18, 8))
        self._blocker_body_var = tk.StringVar(value="")
        self._blocker_body = ctk.CTkLabel(card, textvariable=self._blocker_body_var, justify="left", wraplength=720)
        self._blocker_body.grid(row=1, column=0, sticky="w", padx=18, pady=(0, 18))
        self._hide_blocker()

        self.after(80, self.apply_camera_layout_preset)

    def _apply_text_widget_theme(self) -> None:
        bg = _theme_textbox_bg_hex()
        fg = _theme_textbox_fg_hex()
        sel_bg = _theme_value("CTkButton", "fg_color", None)
        if isinstance(sel_bg, (list, tuple)):
            sel_bg = sel_bg[_appearance_index()]
        sel_bg = str(sel_bg) if sel_bg else bg

        try:
            self.text_widget.configure(
                bg=bg,
                fg=fg,
                insertbackground=fg,
                selectbackground=sel_bg,
                selectforeground=fg,
                highlightthickness=0,
                bd=0,
                relief="flat",
            )
        except Exception:
            pass

    def apply_camera_layout_preset(self) -> None:
        try:
            self.paned.paneconfigure(self._left_pane, minsize=int(self.app.camera_pane_minsize()))
        except Exception:
            pass
        try:
            self.paned.update_idletasks()
            pw = int(self.paned.winfo_width())
            if pw <= 2:
                self.after(120, self.apply_camera_layout_preset)
                return
            ratio = float(self.app.camera_layout_ratio())
            min_left = int(self.app.camera_pane_minsize())
            min_right = 320
            x = int(pw * ratio)
            max_x = max(min_left, pw - min_right)
            x = max(min_left, min(max_x, x))
            self.paned.sash_place(0, x, 0)
        except Exception:
            pass

    def on_show(self) -> None:
        self._running = True
        self.apply_camera_layout_preset()
        self._sync_detection_readiness()
        self._schedule_prereq_check()
        self.app.start_camera_loop(self._consume_frame, interval_ms=33)

    def on_hide(self) -> None:
        self._running = False
        self._enabled = False
        self.app.stop_camera_loop()
        self._cancel_prereq_check()
        self._imgtk = None
        self._img_size = None
        self._bg_hex = _theme_frame_bg_hex()
        self.video_label.configure(image="", text=tr("cam_preview_placeholder"), bg=self._bg_hex)
        self._hide_blocker()
        try:
            self.hands_var.set(tr("lbl_hands_detected", n=0))
        except Exception:
            pass
        try:
            self.latency_var.set(tr("lbl_last_classification_times", mp_ms="—", model_ms="—", total_ms="—"))
        except Exception:
            pass

    def _load_text(self) -> None:
        name = self.file_menu.get()
        if not name or name.startswith("("):
            messagebox.showerror(tr("dlg_error"), tr("err_no_filename"))
            return
        path = self.text_files_dir / name
        if not path.exists():
            messagebox.showerror(tr("dlg_error"), tr("err_file_not_exists", file=str(path)))
            return
        self._content = path.read_text(encoding="utf-8")
        self._idx = 0
        self._ok = 0
        self._fail = 0
        self.text_widget.configure(state="normal")
        self.text_widget.delete("1.0", "end")
        self.text_widget.insert("end", self._content)
        self.text_widget.configure(state="disabled")
        self._update_stats()
        self.app.log(tr("log_file_loaded", file=str(path)))

    def _consume_frame(self, frame_bgr) -> None:
        frame = frame_bgr
        if self.app.flip_horizontal:
            frame = cv2.flip(frame, 1)
        if self.app.flip_vertical:
            frame = cv2.flip(frame, 0)

        now = time.time()
        try:
            self.interval_ms.set(max(1, int((self.interval_str.get() or "").strip())))
        except Exception:
            self.interval_ms.set(50)
        try:
            self.threshold.set(float((self.threshold_str.get() or "").strip()))
        except Exception:
            self.threshold.set(0.7)
        self.app.show_overlays = bool(self.overlay_switch.get())

        do_predict = (
            self._enabled
            and self._detection_ready
            and (now - self._last_pred_ts) * 1000.0 >= float(self.interval_ms.get())
        )
        letter = ""
        if do_predict:
            self._last_pred_ts = now
            try:
                detector = self.app.get_detector()
                pred_text, conf, frame_out, pred_prob, hand_count = detector.process_frame(
                    frame,
                    threshold=float(self.threshold.get()),
                )
                frame = frame_out
                try:
                    mp_ms = getattr(detector, "last_mediapipe_ms", None)
                    model_ms = getattr(detector, "last_model_ms", None)
                    if pred_text and pred_text != "—" and mp_ms is not None and model_ms is not None:
                        total_ms = float(mp_ms) + float(model_ms)
                        self.latency_var.set(
                            tr(
                                "lbl_last_classification_times",
                                mp_ms=f"{float(mp_ms):.0f}",
                                model_ms=f"{float(model_ms):.0f}",
                                total_ms=f"{float(total_ms):.0f}",
                            )
                        )
                except Exception:
                    pass
                try:
                    self.hands_var.set(tr("lbl_hands_detected", n=int(hand_count)))
                except Exception:
                    pass
                if pred_text and pred_text != "—":
                    letter = pred_text
                    self._check_letter(letter)
                if pred_prob is not None:
                    self._update_topk(pred_prob)
            except Exception as exc:
                pass

        try:
            self._update_preview(frame)
        except Exception:
            pass

    def _hide_blocker(self) -> None:
        try:
            self._blocker.place_forget()
        except Exception:
            pass

    def _show_blocker(self, message: str) -> None:
        try:
            self._blocker.configure(fg_color=_theme_overlay_bg_hex())
        except Exception:
            pass
        try:
            self._blocker_body_var.set(message)
        except Exception:
            pass
        try:
            self._blocker.place(relx=0, rely=0, relwidth=1, relheight=1)
        except Exception:
            pass

    def _detection_prereq_status(self) -> tuple[bool, list[str]]:
        missing: list[str] = []
        csv_path = (self.app.csv_file_var.get() or "").strip()
        model_path = (self.app.model_file_var.get() or "").strip()
        scaler_path = (self.app.scaler_file_var.get() or "").strip()

        if not csv_path or not os.path.exists(csv_path):
            missing.append(tr("det_missing_csv", path=csv_path or "(empty)"))
        else:
            try:
                with open(csv_path, "r", encoding="utf-8", newline="") as f:
                    header_line = f.readline().strip()
                if not header_line:
                    missing.append(tr("det_invalid_csv", path=csv_path))
                else:
                    header_cols = [c.strip() for c in header_line.split(",")]
                    if "label" not in header_cols:
                        missing.append(tr("det_invalid_csv", path=csv_path))
            except Exception:
                missing.append(tr("det_invalid_csv", path=csv_path))

        if not model_path or not os.path.exists(model_path):
            missing.append(tr("det_missing_model", path=model_path or "(empty)"))
        if not scaler_path or not os.path.exists(scaler_path):
            missing.append(tr("det_missing_scaler", path=scaler_path or "(empty)"))

        return (len(missing) == 0), missing

    def _sync_detection_readiness(self) -> None:
        ok, missing = self._detection_prereq_status()
        self._detection_ready = bool(ok)

        try:
            self.start_btn.configure(state="normal" if ok else "disabled")
            self.stop_btn.configure(state="normal" if ok else "disabled")
        except Exception:
            pass

        if ok:
            self._hide_blocker()
            return

        self._enabled = False
        msg = tr(
            "det_missing_files_body",
            files="\n".join(f"- {m}" for m in missing),
            step1=tr("tab_collect"),
            step2=tr("tab_train"),
        )
        self._show_blocker(msg)

    def _schedule_prereq_check(self) -> None:
        self._cancel_prereq_check()

        def _tick():
            self._prereq_after_id = None
            try:
                if self.app.active_view != self._view_key:
                    return
                self._sync_detection_readiness()
            finally:
                try:
                    if self.app.active_view == self._view_key:
                        self._prereq_after_id = self.after(1000, _tick)
                except Exception:
                    self._prereq_after_id = None

        try:
            self._prereq_after_id = self.after(250, _tick)
        except Exception:
            self._prereq_after_id = None

    def _cancel_prereq_check(self) -> None:
        if not self._prereq_after_id:
            return
        try:
            self.after_cancel(self._prereq_after_id)
        except Exception:
            pass
        self._prereq_after_id = None

        self._update_preview(frame)

    def _check_letter(self, recognised: str) -> None:
        if not self._content or self._idx >= len(self._content):
            return

        while self._idx < len(self._content) and self._content[self._idx] in " \n\r\t":
            self._idx += 1
        if self._idx >= len(self._content):
            return

        expected = self._content[self._idx]
        if recognised.upper() == expected.upper():
            start = f"1.0+{self._idx}c"
            end = f"1.0+{self._idx+1}c"
            self.text_widget.configure(state="normal")
            self.text_widget.tag_add("correct", start, end)
            self.text_widget.configure(state="disabled")
            self._idx += 1
            self._ok += 1
        else:
            self._fail += 1
        self._update_stats()

    def _update_stats(self) -> None:
        total = sum(1 for ch in self._content if ch not in " \n\r\t") if self._content else 0
        remain = max(0, total - self._ok)
        self.stats.configure(
            text="\n".join(
                [
                    tr("stat_correct", ok=self._ok, total=total),
                    tr("stat_failed", fail=self._fail),
                    tr("stat_remaining", remain=remain),
                ]
            )
        )

    def _update_preview(self, frame_bgr) -> None:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(frame_rgb)
        w = max(200, int(self.video_area.winfo_width()) - 24)
        h = max(160, int(self.video_area.winfo_height()) - 24)
        if w <= 1 or h <= 1:
            w, h = 800, 450
        self._bg_hex = _theme_frame_bg_hex()
        pil = _letterbox_to(pil, w, h, self._bg_hex)
        size = (w, h)
        if self._imgtk is None or self._img_size != size:
            self._img_size = size
            self._imgtk = ImageTk.PhotoImage(pil)
            self.video_label.configure(image=self._imgtk, text="", bg=self._bg_hex)
        else:
            self._imgtk.paste(pil)

    def _start(self) -> None:
        self._sync_detection_readiness()
        if not self._detection_ready:
            return
        self._enabled = True
        self.app.log(tr("log_text_practice_started"))

    def _stop(self) -> None:
        self._enabled = False
        self.app.log(tr("log_text_practice_stopped"))

    def _on_camera_changed(self, value: str) -> None:
        idx = _parse_camera_index(value)
        self.app.open_camera(idx)

    def _restart_camera(self) -> None:
        self.app.restart_camera(int(self.app.camera_index_var.get()))

    def _flip_h(self) -> None:
        self.app.flip_horizontal = not self.app.flip_horizontal
        self.app.log(tr("log_flip_horizontal_on") if self.app.flip_horizontal else tr("log_flip_horizontal_off"))

    def _flip_v(self) -> None:
        self.app.flip_vertical = not self.app.flip_vertical
        self.app.log(tr("log_flip_vertical_on") if self.app.flip_vertical else tr("log_flip_vertical_off"))

    def _set_topk_text(self, text: str) -> None:
        self.topk.configure(state="normal")
        self.topk.delete("1.0", "end")
        self.topk.insert("end", text)
        self.topk.configure(state="disabled")

    def _update_topk(self, pred_prob: np.ndarray) -> None:
        try:
            detector = self.app.get_detector()
            classes = detector.get_classes()
            top10_indices = np.argsort(pred_prob)[::-1][:10]
            lines = []
            for i, idx in enumerate(top10_indices, start=1):
                cls = classes[int(idx)] if int(idx) < len(classes) else str(idx)
                prob = float(pred_prob[int(idx)]) * 100.0
                lines.append(f"{i}. {cls}: {prob:.2f}%")
            self._set_topk_text("\n".join(lines))
        except Exception:
            pass


