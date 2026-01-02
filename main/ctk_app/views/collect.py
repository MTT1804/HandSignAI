from __future__ import annotations

import csv
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
    MAX_HANDS_SUPPORTED,
    _extract_multi_hand_features,
    _letterbox_to,
    _parse_camera_index,
    _theme_frame_bg_hex,
)

class DataCollectionView(ctk.CTkFrame):
    def __init__(self, master, app: App):
        super().__init__(master)
        self.app = app

        self._timer_after_id: str | None = None
        self._timer_remaining_s: int = 0
        self._batch_running: bool = False
        self._batch_thread: threading.Thread | None = None

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
        controls = ctk.CTkScrollableFrame(right_container)
        controls.grid(row=0, column=0, sticky="nsew")
        controls.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(controls, text=tr("tab_collect"), font=ctk.CTkFont(size=18, weight="bold")).grid(
            row=0, column=0, sticky="w", padx=14, pady=(14, 4)
        )

        self.hands_var = tk.StringVar(value=tr("lbl_hands_detected", n=0))
        ctk.CTkLabel(controls, textvariable=self.hands_var).grid(row=1, column=0, sticky="w", padx=14, pady=(0, 10))

        cam_values = [f"Cam {i}" for i in app.available_cameras] if app.available_cameras else ["Cam 0"]
        self.cam_menu = ctk.CTkOptionMenu(
            controls,
            values=cam_values,
            command=self._on_camera_changed,
        )
        self.cam_menu.grid(row=2, column=0, sticky="ew", padx=14, pady=(0, 10))
        self.cam_menu.set(f"Cam {app.camera_index_var.get()}")

        self.restart_btn = ctk.CTkButton(controls, text=f"↻ {tr('btn_restart_camera')}", command=self._restart_camera)
        self.restart_btn.grid(row=3, column=0, sticky="ew", padx=14, pady=(0, 10))

        self.label_entry = ctk.CTkEntry(controls, placeholder_text=tr("placeholder_label"))
        self.label_entry.grid(row=4, column=0, sticky="ew", padx=14, pady=(0, 8))
        self.set_label_btn = ctk.CTkButton(controls, text=f"✎ {tr('btn_set_label')}", command=self._set_label)
        self.set_label_btn.grid(row=5, column=0, sticky="ew", padx=14, pady=(0, 10))

        save_box = ctk.CTkFrame(controls, fg_color="transparent")
        save_box.grid(row=6, column=0, sticky="ew", padx=14, pady=(0, 10))
        save_box.grid_columnconfigure(0, weight=1)

        self.save_btn = ctk.CTkButton(save_box, text=f"↓ {tr('btn_save_sample')}", command=app.save_sample)
        self.save_btn.grid(row=0, column=0, sticky="ew", pady=(0, 6))

        self.augment_chk = ctk.CTkCheckBox(
            save_box,
            text=tr("chk_collect_augmentation"),
            variable=self.app.collect_augment_enabled,
        )
        self.augment_chk.grid(row=1, column=0, sticky="w")

        self.mirror_chk = ctk.CTkCheckBox(
            save_box,
            text=tr("chk_collect_mirror"),
            variable=self.app.collect_mirror_enabled,
        )
        self.mirror_chk.grid(row=2, column=0, sticky="w")

        self.undo_btn = ctk.CTkButton(controls, text=f"↩ {tr('btn_undo_last')}", command=app.undo_last_sample)
        self.undo_btn.grid(row=7, column=0, sticky="ew", padx=14, pady=(0, 10))

        batch_frame = ctk.CTkFrame(controls)
        batch_frame.grid(row=9, column=0, sticky="ew", padx=14, pady=(0, 10))
        batch_frame.grid_columnconfigure(0, weight=1)
        batch_frame.grid_columnconfigure(1, weight=0)

        ctk.CTkLabel(batch_frame, text=tr("section_batch"), font=ctk.CTkFont(size=14, weight="bold")).grid(
            row=0, column=0, columnspan=2, sticky="w", padx=10, pady=(10, 6)
        )

        self.batch_dir_var = tk.StringVar(value="")
        self.batch_dir_entry = ctk.CTkEntry(batch_frame, textvariable=self.batch_dir_var, placeholder_text=tr("placeholder_batch_folder"))
        self.batch_dir_entry.grid(row=1, column=0, sticky="ew", padx=(10, 6), pady=(0, 6))
        self.batch_browse_btn = ctk.CTkButton(batch_frame, text=tr("btn_browse"), width=90, command=self._browse_batch_folder)
        self.batch_browse_btn.grid(row=1, column=1, sticky="e", padx=(0, 10), pady=(0, 6))

        self.batch_overlay_chk = ctk.CTkCheckBox(
            batch_frame,
            text=tr("chk_batch_save_overlay"),
            variable=self.app.collect_batch_save_overlay,
        )
        self.batch_overlay_chk.grid(row=2, column=0, columnspan=2, sticky="w", padx=10, pady=(0, 6))

        self.batch_run_btn = ctk.CTkButton(batch_frame, text=tr("btn_batch_process"), command=self._start_batch_processing)
        self.batch_run_btn.grid(row=3, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 10))

        timer_frame = ctk.CTkFrame(controls)
        timer_frame.grid(row=8, column=0, sticky="ew", padx=14, pady=(0, 10))
        timer_frame.grid_columnconfigure(0, weight=1)
        timer_frame.grid_columnconfigure(1, weight=0)

        self.timer_switch = ctk.CTkSwitch(
            timer_frame,
            text=tr("lbl_self_timer"),
            variable=self.app.collect_self_timer_enabled,
            command=self._on_timer_toggle,
        )
        self.timer_switch.grid(row=0, column=0, sticky="w", padx=10, pady=(10, 6))

        self.timer_loop_chk = ctk.CTkCheckBox(
            timer_frame,
            text=tr("chk_self_timer_loop"),
            variable=self.app.collect_self_timer_loop_enabled,
        )
        self.timer_loop_chk.grid(row=0, column=1, sticky="e", padx=10, pady=(10, 6))

        seconds_row = ctk.CTkFrame(timer_frame, fg_color="transparent")
        seconds_row.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 8))
        seconds_row.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(seconds_row, text=tr("lbl_self_timer_seconds")).grid(row=0, column=0, sticky="w", padx=(0, 8))

        self.timer_seconds_var = tk.StringVar(value=str(int(self.app.collect_self_timer_seconds.get())))
        self.timer_seconds_entry = ctk.CTkEntry(seconds_row, textvariable=self.timer_seconds_var, width=90)
        self.timer_seconds_entry.grid(row=0, column=1, sticky="w")

        self.timer_countdown_var = tk.StringVar(value="")
        self.timer_countdown_lbl = ctk.CTkLabel(timer_frame, textvariable=self.timer_countdown_var)
        self.timer_countdown_lbl.grid(row=2, column=0, sticky="w", padx=10, pady=(0, 10))

        self.timer_btn = ctk.CTkButton(timer_frame, text=tr("btn_self_timer_start"), command=self._toggle_timer)
        self.timer_btn.grid(row=2, column=1, sticky="e", padx=10, pady=(0, 10))

        self.overlay_switch = ctk.CTkSwitch(
            controls,
            text=tr("chk_overlay"),
            command=self._toggle_overlay,
        )
        self.overlay_switch.grid(row=10, column=0, sticky="w", padx=14, pady=(0, 8))
        if self.app.show_overlays:
            self.overlay_switch.select()
        else:
            self.overlay_switch.deselect()

        flip_row = ctk.CTkFrame(controls, fg_color="transparent")
        flip_row.grid(row=10, column=0, sticky="ew", padx=(14, 22), pady=(0, 10))
        flip_row.grid_columnconfigure(0, weight=1)
        flip_row.grid_columnconfigure(1, weight=1)
        ctk.CTkButton(flip_row, text=tr("btn_flip_h"), command=self._flip_h).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ctk.CTkButton(flip_row, text=tr("btn_flip_v"), command=self._flip_v).grid(row=0, column=1, sticky="ew", padx=(6, 0))

        self._add_slider(controls, 11, tr("lbl_brightness"), app.brightness, -100, 100)
        self._add_slider(controls, 12, tr("lbl_contrast"), app.contrast, 0, 200)
        self._add_slider(controls, 13, tr("lbl_gamma"), app.gamma, 1, 300)

        reset_row = ctk.CTkFrame(controls, fg_color="transparent")
        reset_row.grid(row=14, column=0, sticky="ew", padx=14, pady=(10, 14))
        reset_row.grid_columnconfigure(0, weight=1)
        reset_row.grid_columnconfigure(1, weight=1)
        ctk.CTkButton(reset_row, text=f"⌫ {tr('btn_clear_images')}", command=app.clear_images).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ctk.CTkButton(reset_row, text=f"↻ {tr('btn_reset_csv')}", command=app.reset_csv).grid(row=0, column=1, sticky="ew", padx=(6, 0))

        self._running = False

        self._on_timer_toggle()

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
        if not self._batch_running:
            self.app.start_camera_loop(self._consume_frame, interval_ms=33)

    def on_hide(self) -> None:
        self._running = False
        self._cancel_timer()
        self._cancel_batch_if_running()
        self.app.stop_camera_loop()
        self._imgtk = None
        self._img_size = None
        self._bg_hex = _theme_frame_bg_hex()
        self.video_label.configure(image="", text=tr("cam_preview_placeholder"), bg=self._bg_hex)
        try:
            self.hands_var.set(tr("lbl_hands_detected", n=0))
        except Exception:
            pass

    def _on_camera_changed(self, value: str) -> None:
        idx = _parse_camera_index(value)
        self.app.open_camera(idx)

    def _restart_camera(self) -> None:
        self.app.restart_camera(int(self.app.camera_index_var.get()))

    def _set_label(self) -> None:
        self.app.set_label(self.label_entry.get())

    def _on_timer_toggle(self) -> None:
        enabled = bool(self.app.collect_self_timer_enabled.get())
        try:
            self.timer_seconds_entry.configure(state="normal" if enabled else "disabled")
        except Exception:
            pass
        try:
            self.timer_btn.configure(state="normal" if enabled else "disabled")
        except Exception:
            pass
        try:
            self.timer_loop_chk.configure(state="normal" if enabled else "disabled")
        except Exception:
            pass
        if not enabled:
            self._cancel_timer()

    def _toggle_timer(self) -> None:
        if self._timer_after_id is not None:
            self._cancel_timer()
        else:
            self._start_timer()

    def _start_timer(self) -> None:
        if not bool(self.app.collect_self_timer_enabled.get()):
            return

        try:
            seconds = int((self.timer_seconds_var.get() or "").strip())
            if seconds <= 0:
                raise ValueError
        except Exception:
            messagebox.showerror(tr("dlg_error"), tr("err_timer_seconds"))
            return

        try:
            self.app.collect_self_timer_seconds.set(seconds)
        except Exception:
            pass

        self._timer_remaining_s = seconds
        self.app.log(tr("log_self_timer_started", seconds=seconds))
        try:
            self.timer_btn.configure(text=tr("btn_self_timer_cancel"))
        except Exception:
            pass
        self._tick_timer()

    def _tick_timer(self) -> None:
        if not self._running:
            self._cancel_timer()
            return
        if self._timer_remaining_s <= 0:
            self._timer_after_id = None
            self.timer_countdown_var.set("")
            self.app.log(tr("log_self_timer_fired"))
            try:
                self.app.save_sample()
            except Exception:
                self._cancel_timer()
                return

            if bool(self.app.collect_self_timer_loop_enabled.get()) and bool(self.app.collect_self_timer_enabled.get()):
                self._start_timer()
                return

            try:
                self.timer_btn.configure(text=tr("btn_self_timer_start"))
            except Exception:
                pass
            return

        self.timer_countdown_var.set(tr("lbl_self_timer_countdown", seconds=int(self._timer_remaining_s)))
        self._timer_remaining_s -= 1
        self._timer_after_id = self.after(1000, self._tick_timer)

    def _cancel_timer(self) -> None:
        if self._timer_after_id is not None:
            try:
                self.after_cancel(self._timer_after_id)
            except Exception:
                pass
        self._timer_after_id = None
        self._timer_remaining_s = 0
        try:
            self.timer_countdown_var.set("")
            self.timer_btn.configure(text=tr("btn_self_timer_start"))
        except Exception:
            pass

    def _browse_batch_folder(self) -> None:
        try:
            folder = filedialog.askdirectory()
        except Exception:
            folder = ""
        if folder:
            self.batch_dir_var.set(folder)

    def _set_batch_running_ui(self, running: bool) -> None:
        self._batch_running = bool(running)
        state = "disabled" if self._batch_running else "normal"
        for w in (self.batch_dir_entry, self.batch_browse_btn, self.batch_overlay_chk, self.batch_run_btn):
            try:
                w.configure(state=state)
            except Exception:
                pass

    def _enter_batch_preview_mode(self) -> None:
        try:
            self.app.stop_camera_loop()
        except Exception:
            pass

    def _exit_batch_preview_mode(self) -> None:
        if not self._running:
            return
        try:
            self.app.start_camera_loop(self._consume_frame, interval_ms=33)
        except Exception:
            pass

    def _show_batch_preview_frame(self, frame_bgr, hand_count: int | None = None) -> None:
        try:
            if hand_count is not None:
                self.hands_var.set(tr("lbl_hands_detected", n=int(hand_count)))
        except Exception:
            pass
        try:
            self._update_preview(frame_bgr)
        except Exception:
            pass

    def _cancel_batch_if_running(self) -> None:
        if self._batch_running:
            self._set_batch_running_ui(False)

    def _start_batch_processing(self) -> None:
        if self._batch_running:
            return
        if not bool(getattr(self.app, "deps_ready", True)):
            messagebox.showerror(tr("dlg_error"), tr("err_deps_not_ready"))
            return
        if self.app._hands is None:
            messagebox.showerror(tr("dlg_error"), tr("err_mediapipe_not_ready"))
            return
        if not self.app.current_label:
            messagebox.showerror(tr("dlg_error"), tr("err_label_empty"))
            return

        folder = (self.batch_dir_var.get() or "").strip()
        if not folder or not os.path.isdir(folder):
            messagebox.showerror(tr("dlg_error"), tr("err_batch_folder"))
            return

        self._enter_batch_preview_mode()

        self._set_batch_running_ui(True)
        label = str(self.app.current_label)
        save_overlay = bool(self.app.collect_batch_save_overlay.get())

        MIN_CONF = 0.60

        def _worker():
            processed = 0
            saved = 0
            skipped = 0
            last_preview_ts = 0.0
            try:
                self.app._prepare_directories_and_csv()
                src = Path(folder)
                files: list[Path] = []
                for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
                    files.extend(src.rglob(ext))
                files = sorted(set(files), key=lambda p: str(p.relative_to(src)).lower())

                if not files:
                    self.app.log(tr("log_batch_no_images", folder=str(src)))
                    return

                label_dir = Path(self.app.images_dir) / label
                label_dir.mkdir(parents=True, exist_ok=True)

                next_idx = self.app._compute_next_image_index(label_dir, label)

                csv_path = Path(self.app.csv_file_var.get())
                with csv_path.open("a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    for p in files:
                        processed += 1
                        img = cv2.imread(str(p))
                        if img is None:
                            skipped += 1
                            continue

                        frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        results = self.app._hands.process(frame_rgb)
                        if not results or not results.multi_hand_landmarks:
                            skipped += 1
                            continue

                        conf = self.app._mediapipe_results_confidence(results)
                        if conf < MIN_CONF:
                            skipped += 1
                            continue

                        row = _extract_multi_hand_features(results, max_hands=MAX_HANDS_SUPPORTED)
                        writer.writerow(row + [label, int(next_idx)])

                        out = img.copy()
                        preview = out if save_overlay else img.copy()
                        try:
                            if self.app._mp_drawing is not None and self.app._mp_hands is not None:
                                for hand_landmarks in results.multi_hand_landmarks:
                                    self.app._mp_drawing.draw_landmarks(preview, hand_landmarks, self.app._mp_hands.HAND_CONNECTIONS)
                                if save_overlay and preview is not out:
                                    for hand_landmarks in results.multi_hand_landmarks:
                                        self.app._mp_drawing.draw_landmarks(out, hand_landmarks, self.app._mp_hands.HAND_CONNECTIONS)
                        except Exception:
                            pass
                        try:
                            cv2.putText(
                                preview,
                                tr("overlay_label_idx", label=label, idx=int(next_idx)),
                                (10, 24),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.55,
                                (0, 255, 0),
                                1,
                            )
                        except Exception:
                            pass
                        if save_overlay and preview is not out:
                            try:
                                cv2.putText(
                                    out,
                                    tr("overlay_label_idx", label=label, idx=int(next_idx)),
                                    (10, 24),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.55,
                                    (0, 255, 0),
                                    1,
                                )
                            except Exception:
                                pass

                        try:
                            now_ts = time.time()
                            if now_ts - last_preview_ts >= 0.05:
                                last_preview_ts = now_ts
                                hc = len(results.multi_hand_landmarks) if results and results.multi_hand_landmarks else 0
                                self.after(0, lambda fr=preview.copy(), _hc=int(hc): self._show_batch_preview_frame(fr, _hc))
                        except Exception:
                            pass

                        img_path = label_dir / f"{label}_{int(next_idx)}.jpg"
                        try:
                            ok = bool(cv2.imwrite(str(img_path), out))
                            if ok:
                                saved += 1
                                self.app._last_saved_sample = {
                                    "label": str(label),
                                    "idx": int(next_idx),
                                    "img_path": str(img_path),
                                    "csv_path": str(csv_path),
                                }
                        except Exception:
                            pass

                        next_idx += 1

                        if processed % 25 == 0:
                            self.app.log(tr("log_batch_progress", done=processed, total=len(files), saved=saved, skipped=skipped))

                try:
                    self.app._collect_next_idx_label = label
                    self.app._collect_next_idx = int(next_idx)
                except Exception:
                    pass
                self.app.log(tr("log_batch_done", total=len(files), saved=saved, skipped=skipped))
            except Exception as exc:
                self.app.log(tr("log_batch_failed", err=str(exc)))
            finally:
                try:
                    self.after(0, lambda: (self._set_batch_running_ui(False), self._exit_batch_preview_mode()))
                except Exception:
                    pass

        self.app.log(tr("log_batch_started", folder=folder, label=label))
        self._batch_thread = threading.Thread(target=_worker, daemon=True)
        self._batch_thread.start()

    def _flip_h(self) -> None:
        self.app.flip_horizontal = not self.app.flip_horizontal
        self.app.log(tr("log_flip_horizontal_on") if self.app.flip_horizontal else tr("log_flip_horizontal_off"))

    def _flip_v(self) -> None:
        self.app.flip_vertical = not self.app.flip_vertical
        self.app.log(tr("log_flip_vertical_on") if self.app.flip_vertical else tr("log_flip_vertical_off"))

    def _toggle_overlay(self) -> None:
        self.app.show_overlays = bool(self.overlay_switch.get())
        self.app.log(tr("log_overlays", state=tr("status_on") if self.app.show_overlays else tr("status_off")))

    def _consume_frame(self, frame_bgr) -> None:
        frame = frame_bgr
        if self.app.flip_horizontal:
            frame = cv2.flip(frame, 1)
        if self.app.flip_vertical:
            frame = cv2.flip(frame, 0)
        frame = self.app.apply_image_adjustments(frame)
        self.app.last_frame_raw = frame.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.app._hands.process(frame_rgb)

        try:
            cnt = len(results.multi_hand_landmarks) if results and results.multi_hand_landmarks else 0
            self.hands_var.set(tr("lbl_hands_detected", n=int(cnt)))
        except Exception:
            pass
        if self.app.show_overlays and results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.app._mp_drawing.draw_landmarks(frame, hand_landmarks, self.app._mp_hands.HAND_CONNECTIONS)
        if self.app.show_overlays and self.app.current_label:
            next_idx = self.app.get_collect_next_index()
            txt = (
                tr("overlay_label_idx", label=self.app.current_label, idx=int(next_idx))
                if next_idx is not None
                else tr("overlay_label", label=self.app.current_label)
            )
            cv2.putText(frame, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)

        self.app.last_frame = frame
        self._update_preview(frame)

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

    def _add_slider(self, parent, row: int, label: str, var: tk.IntVar, from_: int, to_: int) -> None:
        box = ctk.CTkFrame(parent, fg_color="transparent")
        box.grid(row=row, column=0, sticky="ew", padx=14, pady=(0, 6))
        box.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(box, text=label).grid(row=0, column=0, sticky="w")
        val_lbl = ctk.CTkLabel(box, text=str(var.get()))
        val_lbl.grid(row=0, column=2, sticky="e")
        slider = ctk.CTkSlider(
            box,
            from_=from_,
            to=to_,
            number_of_steps=max(1, to_ - from_),
            command=lambda v: (var.set(int(v)), val_lbl.configure(text=str(int(v)))),
        )
        slider.set(var.get())
        slider.grid(row=0, column=1, sticky="ew", padx=10)


