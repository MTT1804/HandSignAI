from __future__ import annotations

import os
import threading
import time

import customtkinter as ctk

import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

from locales import tr

from ctk_app.common import _letterbox_to, _parse_camera_index, _theme_frame_bg_hex, _theme_overlay_bg_hex

class DetectionView(ctk.CTkFrame):
    def __init__(self, master, app: App):
        super().__init__(master)
        self.app = app

        self._detection_enabled: bool = True
        self._detection_ready: bool = False
        self._view_key: str = "detekcja"
        self._prereq_after_id: str | None = None
        self._next_letter: str = ""
        self._last_letter: str = ""
        self._last_pred_ts: float = 0.0
        self._last_topk_ts: float = 0.0

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        root = ctk.CTkFrame(self)
        root.grid(row=0, column=0, sticky="nsew", padx=18, pady=18)
        root.grid_rowconfigure(0, weight=0)
        root.grid_rowconfigure(1, weight=1)
        root.grid_columnconfigure(0, weight=1)

        controls = ctk.CTkFrame(root)
        controls.grid(row=0, column=0, columnspan=2, sticky="ew", padx=14, pady=(14, 10))
        for i in range(12):
            controls.grid_columnconfigure(i, weight=0)
        controls.grid_columnconfigure(11, weight=1)

        ctk.CTkLabel(controls, text=tr("lbl_interval")).grid(row=0, column=0, padx=(10, 6), pady=8, sticky="w")
        self.interval_var = tk.StringVar(value=str(self.app.det_interval_ms.get()))
        ctk.CTkEntry(controls, width=90, textvariable=self.interval_var).grid(row=0, column=1, padx=(0, 10), pady=8)

        ctk.CTkLabel(controls, text=tr("lbl_threshold")).grid(row=0, column=2, padx=(0, 6), pady=8, sticky="w")
        self.threshold_var = tk.StringVar(value=str(self.app.det_threshold))
        ctk.CTkEntry(controls, width=80, textvariable=self.threshold_var).grid(row=0, column=3, padx=(0, 10), pady=8)

        self.start_btn = ctk.CTkButton(controls, text=f"▶ {tr('btn_start')}", width=90, command=self._start_detection)
        self.start_btn.grid(row=0, column=4, padx=(0, 8), pady=8)
        self.stop_btn = ctk.CTkButton(controls, text=f"■ {tr('btn_stop')}", width=90, command=self._stop_detection)
        self.stop_btn.grid(row=0, column=5, padx=(0, 8), pady=8)

        self.clear_btn = ctk.CTkButton(controls, text=f"⌫ {tr('btn_clear')}", width=110, command=self._clear_buffer)
        self.clear_btn.grid(row=0, column=6, padx=(0, 8), pady=8)

        cam_values = [f"Cam {i}" for i in app.available_cameras] if app.available_cameras else ["Cam 0"]
        self.cam_menu = ctk.CTkOptionMenu(controls, values=cam_values, width=70, command=self._on_camera_changed)
        self.cam_menu.grid(row=0, column=7, padx=(0, 8), pady=8)
        self.cam_menu.set(f"Cam {app.camera_index_var.get()}")

        self.restart_btn = ctk.CTkButton(controls, text=f"↻ {tr('btn_restart')}", width=110, command=self._restart_camera)
        self.restart_btn.grid(row=0, column=8, padx=(0, 8), pady=8)

        self.flip_h_btn = ctk.CTkButton(controls, text=f"↔ {tr('btn_flip_h')}", width=90, command=self._flip_h)
        self.flip_h_btn.grid(row=0, column=9, padx=(0, 8), pady=8)
        self.flip_v_btn = ctk.CTkButton(controls, text=f"↕ {tr('btn_flip_v')}", width=90, command=self._flip_v)
        self.flip_v_btn.grid(row=0, column=10, padx=(0, 8), pady=8)

        self._bg_hex = _theme_frame_bg_hex()
        self.paned = tk.PanedWindow(
            root,
            orient="horizontal",
            bd=0,
            sashwidth=6,
            sashrelief="raised",
            bg=self._bg_hex,
        )
        self.paned.grid(row=1, column=0, sticky="nsew")

        left = ctk.CTkFrame(self.paned)
        right_container = ctk.CTkFrame(self.paned)
        self._left_pane = left
        self._right_pane = right_container
        self.paned.add(left, minsize=240)
        self.paned.add(right_container, minsize=300)

        left.grid_rowconfigure(0, weight=1)
        left.grid_rowconfigure(1, weight=0)
        left.grid_columnconfigure(0, weight=1)

        self.video_area = ctk.CTkFrame(left)
        self.video_area.grid(row=0, column=0, sticky="nsew")
        self.video_area.grid_rowconfigure(0, weight=1)
        self.video_area.grid_columnconfigure(0, weight=1)

        self.video_label = tk.Label(self.video_area, text=tr("cam_preview_placeholder"), bg=self._bg_hex)
        self.video_label.grid(row=0, column=0, sticky="nsew", padx=16, pady=16)
        self._imgtk: ImageTk.PhotoImage | None = None
        self._img_size: tuple[int, int] | None = None

        self.confidence_bar = ctk.CTkProgressBar(left)
        self.confidence_bar.grid(row=1, column=0, sticky="ew", pady=(12, 0), padx=16)
        self.confidence_bar.set(0.0)

        right_container.grid_rowconfigure(0, weight=1)
        right_container.grid_columnconfigure(0, weight=1)
        right = ctk.CTkScrollableFrame(right_container)
        right.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        right.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            right,
            text=tr("ctk_pred_title"),
            font=ctk.CTkFont(size=13, weight="bold"),
        ).grid(row=0, column=0, sticky="w", padx=14, pady=(14, 8))

        self.pred_var = tk.StringVar(value="—")
        self.hands_var = tk.StringVar(value=tr("lbl_hands_detected", n=0))
        self.pred_label = ctk.CTkLabel(
            right,
            textvariable=self.pred_var,
            font=ctk.CTkFont(size=56, weight="bold"),
        )
        self.pred_label.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 14))

        self.buffer = ctk.CTkTextbox(right, height=140, font=("Consolas", 12))
        self.buffer.grid(row=2, column=0, sticky="ew", padx=14, pady=(0, 14))
        self.buffer.insert("end", "")

        self.enter_mode_switch = ctk.CTkSwitch(
            right,
            text=tr("chk_enter_mode"),
            variable=app.enter_mode,
        )
        try:
            self.enter_mode_switch.configure(wraplength=260, justify="left")
        except Exception:
            pass
        self.enter_mode_switch.grid(row=3, column=0, sticky="w", padx=14, pady=(0, 14))

        self.hands_label = ctk.CTkLabel(right, textvariable=self.hands_var)
        self.hands_label.grid(row=4, column=0, sticky="w", padx=14, pady=(0, 10))

        ctk.CTkLabel(
            right,
            text=tr("lbl_top10"),
            font=ctk.CTkFont(size=13, weight="bold"),
        ).grid(row=5, column=0, sticky="w", padx=14, pady=(0, 6))

        self.topk = ctk.CTkTextbox(right, height=170, font=("Consolas", 11))
        self.topk.grid(row=6, column=0, sticky="ew", padx=14, pady=(0, 14))
        self.topk.configure(state="disabled")

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
            min_right = 300
            x = int(pw * ratio)
            max_x = max(min_left, pw - min_right)
            x = max(min_left, min(max_x, x))
            self.paned.sash_place(0, x, 0)
        except Exception:
            pass

    def set_prediction(self, text: str, confidence: float) -> None:
        self.pred_var.set(text)
        self.confidence_bar.set(max(0.0, min(1.0, confidence)))

    def on_show(self) -> None:
        self._detection_enabled = True
        self.apply_camera_layout_preset()
        self._sync_detection_readiness()
        self._schedule_prereq_check()
        self.app.start_camera_loop(self._consume_frame, interval_ms=33)

    def on_hide(self) -> None:
        self.app.stop_camera_loop()
        self._cancel_prereq_check()
        self._imgtk = None
        self._img_size = None
        self._bg_hex = _theme_frame_bg_hex()
        try:
            self.paned.configure(bg=self._bg_hex)
        except Exception:
            pass
        self.video_label.configure(image="", text=tr("cam_preview_placeholder"), bg=self._bg_hex)
        self.set_prediction("—", 0.0)
        self.hands_var.set(tr("lbl_hands_detected", n=0))
        self._set_topk_text("")
        self._hide_blocker()

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

    def _start_detection(self) -> None:
        self._sync_detection_readiness()
        if not self._detection_ready:
            return
        interval_txt = (self.interval_var.get() or "").strip()
        threshold_txt = (self.threshold_var.get() or "").strip()
        try:
            interval_val = int(interval_txt)
            threshold_val = float(threshold_txt)
        except ValueError:
            messagebox.showerror(tr("dlg_error"), tr("err_bad_interval_threshold"))
            return
        if interval_val <= 0 or threshold_val <= 0 or threshold_val > 1:
            messagebox.showwarning(tr("dlg_warning"), tr("warn_interval_threshold_range"))
            return
        self.app.det_interval_ms.set(interval_val)
        self.app.det_threshold = threshold_val
        self._detection_enabled = True
        self.app.log(tr("log_detection_started"))

    def _stop_detection(self) -> None:
        self._detection_enabled = False
        self._next_letter = ""
        self.app.log(tr("log_detection_stopped"))

    def _clear_buffer(self) -> None:
        try:
            self.buffer.delete("1.0", "end")
        except Exception:
            pass
        self._last_letter = ""
        self._next_letter = ""
        self._set_topk_text("")

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

    def on_enter(self) -> None:
        if self.app.enter_mode.get() and self._next_letter:
            self.buffer.insert("end", self._next_letter)
            self.buffer.see("end")
            self._next_letter = ""

    def _consume_frame(self, frame_bgr) -> None:
        frame = frame_bgr
        if self.app.flip_horizontal:
            frame = cv2.flip(frame, 1)
        if self.app.flip_vertical:
            frame = cv2.flip(frame, 0)

        now = time.time()
        do_predict = (
            self._detection_enabled
            and self._detection_ready
            and (now - self._last_pred_ts) * 1000.0 >= float(self.app.det_interval_ms.get())
        )

        hand_count: int | None = None
        try:
            if do_predict:
                detector = self.app.get_detector()
                pred_text, confidence, frame_out, pred_prob, hand_count = detector.process_frame(
                    frame,
                    threshold=self.app.det_threshold,
                )
                self._last_pred_ts = now
            else:
                pred_text, confidence, frame_out, pred_prob = "—", 0.0, frame, None
        except Exception as exc:
            self.app.log(tr("log_detection_error", err=str(exc)))
            pred_text, confidence, frame_out, pred_prob, hand_count = "—", 0.0, frame, None, 0

        self._update_preview(frame_out)
        if hand_count is not None:
            self.hands_var.set(tr("lbl_hands_detected", n=int(hand_count)))
        if pred_text and pred_text != "—" and pred_text != self._last_letter:
            self._last_letter = pred_text
            self._next_letter = pred_text
            if not self.app.enter_mode.get():
                self.buffer.insert("end", pred_text)
                self.buffer.see("end")
        if pred_prob is not None and (now - self._last_topk_ts) > 0.2:
            self._last_topk_ts = now
            self._update_topk(pred_prob)
        self.set_prediction(pred_text if pred_text != "—" else self.pred_var.get(), confidence)

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

        self._detection_enabled = False
        msg = tr(
            "det_missing_files_body",
            files="\n".join(f"- {m}" for m in missing),
            step1=tr("tab_collect"),
            step2=tr("tab_train"),
        )
        self._show_blocker(msg)

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


