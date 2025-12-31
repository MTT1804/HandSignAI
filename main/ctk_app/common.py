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

try:
    from locales import tr, load
except Exception:
    def tr(key: str, **kwargs):
        return key

    def load(lang: str):
        return None


mp = None
tf = None
training = None


def _appearance_index() -> int:
    try:
        return 1 if ctk.get_appearance_mode().lower() == "dark" else 0
    except Exception:
        return 1


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    c = (hex_color or "").strip()
    if c.startswith("#"):
        c = c[1:]
    if len(c) == 3:
        c = "".join(ch * 2 for ch in c)
    try:
        return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
    except Exception:
        return (0, 0, 0)


def _theme_frame_bg_hex() -> str:
    try:
        fg = ctk.ThemeManager.theme["CTkFrame"]["fg_color"]
        if isinstance(fg, (list, tuple)):
            return str(fg[_appearance_index()])
        return str(fg)
    except Exception:
        return "#000000"


def _theme_text_hex() -> str:
    try:
        v = ctk.ThemeManager.theme["CTkLabel"]["text_color"]
        if isinstance(v, (list, tuple)):
            return str(v[_appearance_index()])
        return str(v)
    except Exception:
        return "#FFFFFF" if _appearance_index() == 1 else "#000000"


def _to_hex_color(color: str, *, fallback: str = "#000000", tk_widget: tk.Misc | None = None) -> str:
    c = (color or "").strip()
    if not c:
        return fallback

    if c.startswith("#"):
        if len(c) == 4:
            return "#" + "".join(ch * 2 for ch in c[1:])
        if len(c) == 7:
            return c
        return fallback

    try:
        w = tk_widget
        if w is None:
            w = tk._default_root
        if w is None:
            return fallback
        r16, g16, b16 = w.winfo_rgb(c)
        r = int(r16 / 65535 * 255)
        g = int(g16 / 65535 * 255)
        b = int(b16 / 65535 * 255)
        return f"#{r:02x}{g:02x}{b:02x}"
    except Exception:
        return fallback


def _copy_pil_image_to_clipboard_windows(img: Image.Image) -> None:
    with io.BytesIO() as output:
        img.convert("RGB").save(output, "BMP")
        data = output.getvalue()[14:]

    CF_DIB = 8
    GMEM_MOVEABLE = 0x0002
    GMEM_ZEROINIT = 0x0040

    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
    user32.OpenClipboard.argtypes = [ctypes.c_void_p]
    user32.OpenClipboard.restype = ctypes.c_bool
    user32.EmptyClipboard.argtypes = []
    user32.EmptyClipboard.restype = ctypes.c_bool
    user32.SetClipboardData.argtypes = [ctypes.c_uint, ctypes.c_void_p]
    user32.SetClipboardData.restype = ctypes.c_void_p
    user32.CloseClipboard.argtypes = []
    user32.CloseClipboard.restype = ctypes.c_bool

    kernel32.GlobalAlloc.argtypes = [ctypes.c_uint, ctypes.c_size_t]
    kernel32.GlobalAlloc.restype = ctypes.c_void_p
    kernel32.GlobalLock.argtypes = [ctypes.c_void_p]
    kernel32.GlobalLock.restype = ctypes.c_void_p
    kernel32.GlobalUnlock.argtypes = [ctypes.c_void_p]
    kernel32.GlobalUnlock.restype = ctypes.c_bool
    kernel32.GlobalFree.argtypes = [ctypes.c_void_p]
    kernel32.GlobalFree.restype = ctypes.c_void_p

    if not user32.OpenClipboard(None):
        raise RuntimeError("OpenClipboard failed")
    try:
        if not user32.EmptyClipboard():
            raise RuntimeError("EmptyClipboard failed")

        h_global = kernel32.GlobalAlloc(GMEM_MOVEABLE | GMEM_ZEROINIT, ctypes.c_size_t(len(data)))
        if not h_global:
            raise MemoryError("GlobalAlloc failed")

        p_global = kernel32.GlobalLock(h_global)
        if not p_global:
            kernel32.GlobalFree(h_global)
            raise MemoryError("GlobalLock failed")
        try:
            ctypes.memmove(ctypes.c_void_p(p_global), data, len(data))
        finally:
            kernel32.GlobalUnlock(h_global)

        if not user32.SetClipboardData(CF_DIB, h_global):
            kernel32.GlobalFree(h_global)
            raise RuntimeError("SetClipboardData failed")
        
    finally:
        user32.CloseClipboard()


def _open_path_cross_platform(path: Path, *, select_in_file_manager: bool = False) -> None:
    try:
        p = Path(path).expanduser().resolve()
    except Exception:
        p = Path(path)

    if sys.platform.startswith("win"):
        try:
            if select_in_file_manager:
                subprocess.Popen(["explorer", f"/select,{str(p)}"], shell=False)
                return
            os.startfile(str(p))
            return
        except Exception:
            try:
                subprocess.Popen(["explorer", str(p)], shell=False)
                return
            except Exception as exc:
                raise RuntimeError(str(exc))

    if sys.platform == "darwin":
        try:
            if select_in_file_manager and p.exists():
                subprocess.Popen(["open", "-R", str(p)], shell=False)
            else:
                subprocess.Popen(["open", str(p)], shell=False)
            return
        except Exception as exc:
            raise RuntimeError(str(exc))

    try:
        target = p
        if select_in_file_manager and p.exists() and p.is_file():
            target = p.parent
        subprocess.Popen(["xdg-open", str(target)], shell=False)
    except Exception as exc:
        raise RuntimeError(str(exc))



def _mpl_safe_color(color: object, fallback: str) -> str:
    if isinstance(color, str) and color.strip():
        return color.strip()
    return fallback


def _theme_overlay_bg_hex() -> str:
    v = _theme_value("CTkTextbox", "fg_color", None)
    if isinstance(v, (list, tuple)):
        v = v[_appearance_index()]
    return str(v) if v else _theme_frame_bg_hex()


def _theme_textbox_bg_hex() -> str:
    v = _theme_value("CTkTextbox", "fg_color", None)
    if isinstance(v, (list, tuple)):
        v = v[_appearance_index()]
    return str(v) if v else _theme_frame_bg_hex()


def _theme_textbox_fg_hex() -> str:
    v = _theme_value("CTkTextbox", "text_color", None)
    if isinstance(v, (list, tuple)):
        v = v[_appearance_index()]
    return str(v) if v else _theme_text_hex()


def _theme_value(widget: str, key: str, default=None):
    try:
        v = ctk.ThemeManager.theme[widget][key]
        if isinstance(v, (list, tuple)):
            return v[_appearance_index()]
        return v
    except Exception:
        return default


def _parse_camera_index(value: str) -> int:
    s = (value or "").strip()
    digits = "".join(ch for ch in s if ch.isdigit())
    try:
        return int(digits) if digits else int(s)
    except Exception:
        return 0


def _letterbox_to(pil: Image.Image, target_w: int, target_h: int, bg_hex: str) -> Image.Image:
    pil = pil.convert("RGB")
    img = pil.copy()
    img.thumbnail((target_w, target_h), Image.Resampling.LANCZOS)
    bg = Image.new("RGB", (target_w, target_h), _hex_to_rgb(bg_hex))
    x = (target_w - img.size[0]) // 2
    y = (target_h - img.size[1]) // 2
    bg.paste(img, (x, y))
    return bg


LANDMARKS_PER_HAND = 21
FEATURES_PER_HAND = LANDMARKS_PER_HAND * 2
MAX_HANDS_SUPPORTED = 4


def _multi_hand_header(max_hands: int = MAX_HANDS_SUPPORTED) -> list[str]:
    header: list[str] = []
    for h in range(1, max_hands + 1):
        for i in range(LANDMARKS_PER_HAND):
            header += [f"h{h}_x{i}", f"h{h}_y{i}"]
    header += ["label", "index"]
    return header


def _extract_multi_hand_features(results, max_hands: int = MAX_HANDS_SUPPORTED) -> list[float]:
    if not results or not getattr(results, "multi_hand_landmarks", None):
        return [0.0] * (max_hands * FEATURES_PER_HAND)

    hands = list(results.multi_hand_landmarks)

    def _centroid_x(hand) -> float:
        try:
            xs = [lm.x for lm in hand.landmark]
            return float(sum(xs) / max(1, len(xs)))
        except Exception:
            return 0.0

    hands.sort(key=_centroid_x)
    hands = hands[:max_hands]

    feats: list[float] = []
    for hand in hands:
        try:
            feats.extend([coord for lm in hand.landmark for coord in (float(lm.x), float(lm.y))])
        except Exception:
            feats.extend([0.0] * FEATURES_PER_HAND)

    missing = max_hands - len(hands)
    if missing > 0:
        feats.extend([0.0] * (missing * FEATURES_PER_HAND))
    return feats


ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")


