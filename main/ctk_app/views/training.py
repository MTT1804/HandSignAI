from __future__ import annotations

import threading
import time
import io
import sys

import customtkinter as ctk

import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from locales import tr

from ctk_app.common import (
    _copy_pil_image_to_clipboard_windows,
    _mpl_safe_color,
    _theme_frame_bg_hex,
    _theme_text_hex,
    _to_hex_color,
)

class TrainingView(ctk.CTkFrame):
    def __init__(self, master, app: App):
        super().__init__(master)
        self.app = app

        self._history = None
        self._plot_canvas: FigureCanvasTkAgg | None = None
        self._plot_widget = None
        self._plot_fig: Figure | None = None
        self._resize_after_id: str | None = None
        self._last_plot_px: tuple[int, int] | None = None

        self._cm: np.ndarray | None = None
        self._cm_labels: list[str] | None = None
        self._cm_canvas: FigureCanvasTkAgg | None = None
        self._cm_widget = None
        self._cm_fig: Figure | None = None
        self._cm_resize_after_id: str | None = None
        self._last_cm_px: tuple[int, int] | None = None

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        scroll = ctk.CTkScrollableFrame(self)
        scroll.grid(row=0, column=0, sticky="nsew", padx=18, pady=18)
        scroll.grid_columnconfigure(0, weight=1)

        root = ctk.CTkFrame(scroll)
        root.grid(row=0, column=0, sticky="nsew")
        root.grid_rowconfigure(0, weight=0)
        root.grid_rowconfigure(1, weight=0)
        root.grid_rowconfigure(2, weight=1)
        root.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            root,
            text=tr("tab_train"),
            font=ctk.CTkFont(size=18, weight="bold"),
        ).grid(row=0, column=0, sticky="w", padx=14, pady=(14, 8))

        self.start_btn = ctk.CTkButton(
            root,
            text=tr("btn_start_training"),
            height=52,
            font=ctk.CTkFont(size=16, weight="bold"),
            command=self._on_start_training,
        )
        self.start_btn.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 10))

        self.advanced_switch = ctk.CTkSwitch(
            root,
            text=tr("chk_show_advanced"),
            command=self._toggle_advanced,
        )
        self.advanced_switch.grid(row=2, column=0, sticky="w", padx=14, pady=(0, 8))

        self.advanced_frame = ctk.CTkFrame(root)
        self.advanced_frame.grid(row=3, column=0, sticky="ew", padx=14, pady=(0, 12))
        self.advanced_frame.grid_columnconfigure(1, weight=1)

        self.epochs_var = tk.StringVar(value=str(app.training_params.epochs))
        self.batch_var = tk.StringVar(value=str(app.training_params.batch_size))
        self.split_var = tk.StringVar(value=str(app.training_params.test_split))

        self.patience_var = tk.StringVar(value=str(app.patience_var.get()))
        self.val_split_var = tk.StringVar(value=str(app.val_split_var.get()))
        self.monitor_var = tk.StringVar(value=str(app.monitor_var.get()))
        self.random_state_var = tk.StringVar(value=str(app.random_state_var.get()))

        self._add_kv(self.advanced_frame, 0, tr("lbl_epochs"), self.epochs_var)
        self._add_kv(self.advanced_frame, 1, tr("lbl_batch_size"), self.batch_var)
        self._add_kv(self.advanced_frame, 2, tr("lbl_test_size"), self.split_var)
        self._add_kv(self.advanced_frame, 3, tr("lbl_patience"), self.patience_var)
        self._add_kv(self.advanced_frame, 4, tr("lbl_val_split"), self.val_split_var)
        self._add_kv(self.advanced_frame, 5, tr("lbl_monitor"), self.monitor_var)
        self._add_kv(self.advanced_frame, 6, tr("lbl_random_state"), self.random_state_var)

        self.advanced_frame.grid_remove()

        chart = ctk.CTkFrame(root)
        chart.grid(row=4, column=0, sticky="nsew", padx=14, pady=(0, 14))
        chart.grid_rowconfigure(0, weight=1)
        chart.grid_columnconfigure(0, weight=1)
        try:
            chart.configure(height=560)
            chart.grid_propagate(False)
        except Exception:
            pass

        self._chart_bg = _theme_frame_bg_hex()
        chart.grid_rowconfigure(1, weight=0)

        self._chart_host = tk.Frame(chart, bg=self._chart_bg, highlightthickness=0, bd=0, height=500)
        self._chart_host.grid(row=0, column=0, sticky="nsew", padx=16, pady=(16, 10))
        self._chart_host.grid_rowconfigure(0, weight=1)
        self._chart_host.grid_columnconfigure(0, weight=1)
        try:
            self._chart_host.grid_propagate(False)
        except Exception:
            pass

        try:
            self._chart_host.bind("<Configure>", self._on_chart_configure)
        except Exception:
            pass

        self._chart_placeholder = tk.Label(
            self._chart_host,
            text=tr("training_chart_placeholder"),
            bg=self._chart_bg,
            fg=_theme_text_hex(),
            justify="center",
            font=("Segoe UI", 14, "bold"),
        )
        self._chart_placeholder.grid(row=0, column=0, sticky="nsew")

        chart_controls = ctk.CTkFrame(chart, fg_color="transparent")
        chart_controls.grid(row=1, column=0, sticky="ew", padx=16, pady=(0, 12))
        chart_controls.grid_columnconfigure(0, weight=1)

        self.copy_chart_btn = ctk.CTkButton(
            chart_controls,
            text=tr("btn_copy_chart"),
            width=200,
            command=self._copy_chart_to_clipboard,
        )
        self.copy_chart_btn.grid(row=0, column=1, sticky="e")
        try:
            self.copy_chart_btn.configure(state="disabled")
        except Exception:
            pass

        cm_box = ctk.CTkFrame(root)
        cm_box.grid(row=5, column=0, sticky="nsew", padx=14, pady=(0, 14))
        cm_box.grid_columnconfigure(0, weight=1)
        cm_box.grid_rowconfigure(1, weight=1)
        try:
            cm_box.configure(height=420)
            cm_box.grid_propagate(False)
        except Exception:
            pass

        cm_header = ctk.CTkFrame(cm_box, fg_color="transparent")
        cm_header.grid(row=0, column=0, sticky="ew", padx=16, pady=(16, 10))
        cm_header.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(cm_header, text=tr("training_cm_title"), font=ctk.CTkFont(size=14, weight="bold")).grid(
            row=0, column=0, sticky="w"
        )
        self.copy_cm_btn = ctk.CTkButton(cm_header, text=tr("btn_copy_cm"), width=240, command=self._copy_cm_to_clipboard)
        self.copy_cm_btn.grid(row=0, column=1, sticky="e")
        try:
            self.copy_cm_btn.configure(state="disabled")
        except Exception:
            pass

        self._cm_bg = _theme_frame_bg_hex()
        self._cm_host = tk.Frame(cm_box, bg=self._cm_bg, highlightthickness=0, bd=0)
        self._cm_host.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0, 16))
        self._cm_host.grid_rowconfigure(0, weight=1)
        self._cm_host.grid_columnconfigure(0, weight=1)
        try:
            self._cm_host.bind("<Configure>", self._on_cm_configure)
        except Exception:
            pass

        self._cm_placeholder = tk.Label(
            self._cm_host,
            text=tr("training_cm_placeholder"),
            bg=self._cm_bg,
            fg=_theme_text_hex(),
            justify="center",
            font=("Segoe UI", 13, "bold"),
        )
        self._cm_placeholder.grid(row=0, column=0, sticky="nsew")

        self.progress = ctk.CTkProgressBar(root)
        self.progress.grid(row=6, column=0, sticky="ew", padx=14, pady=(0, 14))
        self.progress.set(0.0)

        def _poll_progress():
            self.progress.set(max(0.0, min(1.0, float(app.progress_var.get()) / 100.0)))
            self.after(200, _poll_progress)

        _poll_progress()

        root.grid_rowconfigure(4, weight=1)
        root.grid_rowconfigure(5, weight=1)

    def on_show(self) -> None:
        if self._history is None and hasattr(self.app, "_last_training_history"):
            try:
                self._history = getattr(self.app, "_last_training_history", None)
            except Exception:
                self._history = None
        if self._history is not None and self._plot_canvas is None:
            try:
                self.show_plots(self._history)
            except Exception:
                pass

        if self._cm is None and hasattr(self.app, "_last_training_cm"):
            try:
                self._cm = getattr(self.app, "_last_training_cm", None)
                self._cm_labels = getattr(self.app, "_last_training_cm_labels", None)
            except Exception:
                self._cm = None
                self._cm_labels = None
        if self._cm is not None and self._cm_canvas is None:
            try:
                self.show_confusion_matrix(self._cm, self._cm_labels)
            except Exception:
                pass

    def _on_chart_configure(self, event=None) -> None:
        if self._history is None or self._plot_fig is None:
            return

        try:
            w = int(getattr(event, "width", 0) or self._chart_host.winfo_width())
            h = int(getattr(event, "height", 0) or self._chart_host.winfo_height())
        except Exception:
            return

        if w < 480 or h < 320:
            return

        if self._last_plot_px is not None:
            lw, lh = self._last_plot_px
            if abs(w - lw) < 40 and abs(h - lh) < 40:
                return

        if self._resize_after_id is not None:
            try:
                self.after_cancel(self._resize_after_id)
            except Exception:
                pass
            self._resize_after_id = None

        def _do_redraw():
            self._resize_after_id = None
            try:
                if self._history is not None:
                    self.show_plots(self._history)
            except Exception as exc:
                self.app.log(tr("log_plot_error", err=str(exc)))

        try:
            self._resize_after_id = self.after(200, _do_redraw)
        except Exception:
            self._resize_after_id = None

    def _on_cm_configure(self, event=None) -> None:
        if self._cm is None or self._cm_fig is None:
            return

        try:
            w = int(getattr(event, "width", 0) or self._cm_host.winfo_width())
            h = int(getattr(event, "height", 0) or self._cm_host.winfo_height())
        except Exception:
            return

        if w < 520 or h < 240:
            return

        if self._last_cm_px is not None:
            lw, lh = self._last_cm_px
            if abs(w - lw) < 40 and abs(h - lh) < 40:
                return

        if self._cm_resize_after_id is not None:
            try:
                self.after_cancel(self._cm_resize_after_id)
            except Exception:
                pass
            self._cm_resize_after_id = None

        def _do_redraw():
            self._cm_resize_after_id = None
            try:
                if self._cm is not None:
                    self.show_confusion_matrix(self._cm, self._cm_labels)
            except Exception as exc:
                self.app.log(tr("log_plot_error", err=str(exc)))

        try:
            self._cm_resize_after_id = self.after(200, _do_redraw)
        except Exception:
            self._cm_resize_after_id = None

    def show_confusion_matrix(self, cm: np.ndarray, labels: list[str] | None = None) -> None:
        self._cm = np.asarray(cm)
        self._cm_labels = list(labels) if labels else None

        try:
            if self._cm_widget is not None:
                self._cm_widget.destroy()
        except Exception:
            pass
        self._cm_widget = None
        self._cm_canvas = None
        self._cm_fig = None

        try:
            self._cm_placeholder.configure(text=tr("training_cm_placeholder"), fg=_theme_text_hex(), bg=self._cm_bg)
            self._cm_placeholder.grid()
        except Exception:
            pass
        try:
            self.copy_cm_btn.configure(state="disabled")
        except Exception:
            pass

        try:
            self._cm_host.update_idletasks()
            w_px = int(self._cm_host.winfo_width())
            h_px = int(self._cm_host.winfo_height())
        except Exception:
            w_px, h_px = 900, 420
        w_px = max(640, w_px)
        h_px = max(280, h_px)
        self._last_cm_px = (w_px, h_px)

        dpi = 100
        bg = _to_hex_color(_theme_frame_bg_hex(), fallback="#111111", tk_widget=self._cm_host)
        txt = _to_hex_color(_theme_text_hex(), fallback="#ffffff", tk_widget=self._cm_host)
        bg = _mpl_safe_color(bg, "#111111")
        txt = _mpl_safe_color(txt, "#ffffff")

        fig = Figure(figsize=(w_px / dpi, h_px / dpi), dpi=dpi, constrained_layout=True)
        fig.patch.set_facecolor(bg)
        ax = fig.add_subplot(1, 1, 1)
        ax.set_facecolor(bg)
        ax.tick_params(colors=txt)
        for spine in ax.spines.values():
            spine.set_color(txt)

        mat = self._cm
        n = int(mat.shape[0]) if mat.ndim == 2 else 0
        if n <= 0:
            return

        im = ax.imshow(mat, interpolation="nearest", cmap="Blues")
        cbar = fig.colorbar(im, ax=ax, shrink=0.92, pad=0.02)
        try:
            cbar.ax.yaxis.set_tick_params(color=txt)
            for t in cbar.ax.get_yticklabels():
                t.set_color(txt)
        except Exception:
            pass

        try:
            ax.set_box_aspect(1)
        except Exception:
            try:
                ax.set_aspect("equal", adjustable="box")
            except Exception:
                pass
        try:
            ax.set_anchor("C")
        except Exception:
            pass

        ax.set_title(tr("training_cm_title"), color=txt)
        ax.set_xlabel(tr("training_cm_pred"), color=txt)
        ax.set_ylabel(tr("training_cm_true"), color=txt)

        class_labels = self._cm_labels or [str(i) for i in range(n)]
        if len(class_labels) != n:
            class_labels = [str(i) for i in range(n)]

        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(class_labels, rotation=45, ha="right", rotation_mode="anchor", color=txt)
        ax.set_yticklabels(class_labels, color=txt)

        if n <= 15:
            vmax = float(np.max(mat)) if np.max(mat) > 0 else 0.0
            thresh = vmax * 0.55
            for i in range(n):
                for j in range(n):
                    v = int(mat[i, j])
                    ax.text(
                        j,
                        i,
                        str(v),
                        ha="center",
                        va="center",
                        color="#ffffff" if v > thresh else txt,
                        fontsize=9,
                    )

        canvas = FigureCanvasTkAgg(fig, master=self._cm_host)
        canvas.draw()
        widget = canvas.get_tk_widget()
        widget.grid(row=0, column=0, sticky="nsew")
        self._cm_canvas = canvas
        self._cm_widget = widget
        self._cm_fig = fig

        try:
            self._cm_placeholder.grid_remove()
        except Exception:
            pass
        try:
            self.copy_cm_btn.configure(state="normal")
        except Exception:
            pass

    def _copy_cm_to_clipboard(self) -> None:
        fig = self._cm_fig
        if fig is None:
            return
        try:
            if sys.platform.startswith("win"):
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=150)
                buf.seek(0)
                img = Image.open(buf)
                _copy_pil_image_to_clipboard_windows(img)
                self.app.log(tr("log_cm_copied"))
            else:
                self.app.log(tr("err_cm_copy", err="unsupported_os"))
        except Exception as exc:
            self.app.log(tr("err_cm_copy", err=str(exc)))

    def _add_kv(self, master, row: int, label: str, var: tk.StringVar) -> None:
        ctk.CTkLabel(master, text=label).grid(row=row, column=0, sticky="w", padx=12, pady=8)
        ctk.CTkEntry(master, textvariable=var).grid(row=row, column=1, sticky="ew", padx=12, pady=8)

    def _toggle_advanced(self) -> None:
        if self.advanced_switch.get() == 1:
            self.advanced_frame.grid()
            self.app.log(tr("log_advanced_shown"))
        else:
            self.advanced_frame.grid_remove()
            self.app.log(tr("log_advanced_hidden"))

    def _on_start_training(self) -> None:
        try:
            epochs = int(self.epochs_var.get())
            batch = int(self.batch_var.get())
            split = float(self.split_var.get())
            patience = int(self.patience_var.get())
            val_split = float(self.val_split_var.get())
            random_state = int(self.random_state_var.get())
        except ValueError:
            self.app.log(tr("err_invalid_numbers"))
            return

        if split <= 0 or split >= 1 or val_split <= 0 or val_split >= 1:
            self.app.log(tr("err_test_size_range"))
            return

        try:
            self.app.training_params.epochs = int(epochs)
            self.app.training_params.batch_size = int(batch)
            self.app.training_params.test_split = float(split)
        except Exception:
            self.app.training_params = type(self.app.training_params)(epochs=int(epochs), batch_size=int(batch), test_split=float(split))
        self.app.epochs_var.set(epochs)
        self.app.batch_size_var.set(batch)
        self.app.test_size_var.set(split)
        self.app.patience_var.set(patience)
        self.app.val_split_var.set(val_split)
        self.app.monitor_var.set(self.monitor_var.get().strip() or "val_loss")
        self.app.random_state_var.set(random_state)
        self.app.progress_var.set(0.0)
        self.app.log(tr("log_training_started", epochs=epochs, batch=batch, split=split))
        self.app.start_training()

    def show_plots(self, history) -> None:
        self._history = history

        hist = getattr(history, "history", None) or {}
        try:
            keys = list(hist.keys())
            self.app.log(tr("log_history_keys", keys=keys))
        except Exception:
            pass

        try:
            if self._plot_widget is not None:
                self._plot_widget.destroy()
        except Exception:
            pass
        self._plot_widget = None
        self._plot_canvas = None
        self._plot_fig = None

        try:
            self._chart_placeholder.configure(text=tr("training_chart_placeholder"), fg=_theme_text_hex(), bg=self._chart_bg)
            self._chart_placeholder.grid()
        except Exception:
            pass
        try:
            self.copy_chart_btn.configure(state="disabled")
        except Exception:
            pass

        try:
            self._chart_host.update_idletasks()
            w_px = int(self._chart_host.winfo_width())
            h_px = int(self._chart_host.winfo_height())
        except Exception:
            w_px, h_px = 900, 560
        w_px = max(640, w_px)
        h_px = max(520, h_px)
        self._last_plot_px = (w_px, h_px)
        dpi = 100

        bg = _to_hex_color(_theme_frame_bg_hex(), fallback="#111111", tk_widget=self._chart_host)
        txt = _to_hex_color(_theme_text_hex(), fallback="#ffffff", tk_widget=self._chart_host)
        bg = _mpl_safe_color(bg, "#111111")
        txt = _mpl_safe_color(txt, "#ffffff")
        fig = Figure(figsize=(w_px / dpi, h_px / dpi), dpi=dpi)
        fig.patch.set_facecolor(bg)
        loss = list(hist.get("loss", []))
        vloss = list(hist.get("val_loss", []))
        acc = list(hist.get("accuracy", []))
        vacc = list(hist.get("val_accuracy", []))
        if not acc and "acc" in hist:
            acc = list(hist.get("acc", []))
        if not vacc and "val_acc" in hist:
            vacc = list(hist.get("val_acc", []))

        n_epochs = max(len(loss), len(acc), len(vloss), len(vacc), 0)
        if n_epochs <= 0:
            return

        epochs = list(range(1, n_epochs + 1))

        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)
        for ax in (ax1, ax2):
            ax.set_facecolor(bg)
            ax.tick_params(colors=txt)
            for spine in ax.spines.values():
                spine.set_color(txt)

        if loss:
            ax1.plot(epochs[: len(loss)], loss, label=tr("plot_train"))
        if vloss:
            ax1.plot(epochs[: len(vloss)], vloss, label=tr("plot_val"))
        ax1.set_title(tr("plot_loss"), color=txt)
        ax1.set_xlabel(tr("plot_epoch"), color=txt)
        ax1.grid(True, alpha=0.25)
        if loss or vloss:
            try:
                leg1 = ax1.legend(facecolor=bg, edgecolor=txt, framealpha=0.35)
                for t in leg1.get_texts():
                    t.set_color(txt)
            except Exception:
                pass

        if acc:
            ax2.plot(epochs[: len(acc)], acc, label=tr("plot_train"))
        if vacc:
            ax2.plot(epochs[: len(vacc)], vacc, label=tr("plot_val"))
        ax2.set_title(tr("plot_accuracy"), color=txt)
        ax2.set_xlabel(tr("plot_epoch"), color=txt)
        ax2.grid(True, alpha=0.25)
        if acc or vacc:
            try:
                leg2 = ax2.legend(facecolor=bg, edgecolor=txt, framealpha=0.35)
                for t in leg2.get_texts():
                    t.set_color(txt)
            except Exception:
                pass

        try:
            fig.tight_layout(pad=1.6)
        except Exception:
            pass

        try:
            canvas = FigureCanvasTkAgg(fig, master=self._chart_host)
            canvas.draw()
            widget = canvas.get_tk_widget()
            widget.grid(row=0, column=0, sticky="nsew")
            self._plot_canvas = canvas
            self._plot_widget = widget
            self._plot_fig = fig

            try:
                self._chart_placeholder.grid_remove()
            except Exception:
                pass

            try:
                self.copy_chart_btn.configure(state="normal")
            except Exception:
                pass
        except Exception as exc:
            try:
                self._chart_placeholder.configure(text=tr("err_chart_copy", err=str(exc)))
                self._chart_placeholder.grid()
            except Exception:
                pass
            raise

    def _copy_chart_to_clipboard(self) -> None:
        fig = self._plot_fig
        if fig is None:
            return
        try:
            if sys.platform.startswith("win"):
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=150)
                buf.seek(0)
                img = Image.open(buf)
                _copy_pil_image_to_clipboard_windows(img)
                self.app.log(tr("log_chart_copied"))
            else:
                self.app.log(tr("err_chart_copy", err="unsupported_os"))
        except Exception as exc:
            self.app.log(tr("err_chart_copy", err=str(exc)))


