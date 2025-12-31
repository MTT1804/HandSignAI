from __future__ import annotations

import importlib
import os

import cv2
import joblib
import numpy as np
import pandas as pd

from ctk_app.common import MAX_HANDS_SUPPORTED, _extract_multi_hand_features

class HandSignDetector:
    def __init__(self, app: App):
        self._app = app
        self._log = app.log

        self._classes: list[str] | None = None
        self._model = None
        self._scaler = None

        self._mp_hands = app._mp_hands
        self._mp_drawing = app._mp_drawing

    def _ensure_loaded(self) -> None:
        if globals().get("tf") is None:
            globals()["tf"] = importlib.import_module("tensorflow")

        if self._classes is None:
            csv_path = self._app.csv_file_var.get()
            if not csv_path or not os.path.exists(csv_path):
                raise FileNotFoundError(f"CSV not found: {csv_path}")
            df = pd.read_csv(csv_path)
            if "label" not in df.columns:
                raise ValueError("CSV missing 'label' column")
            self._classes = sorted(df["label"].unique().tolist())
            self._log(f"Classes loaded: {len(self._classes)}")

        if self._model is None:
            model_path = self._app.model_file_var.get()
            if not model_path or not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found: {model_path}")
            self._model = globals()["tf"].keras.models.load_model(model_path)
            self._log("Model loaded.")

        if self._scaler is None:
            scaler_path = self._app.scaler_file_var.get()
            if not scaler_path or not os.path.exists(scaler_path):
                raise FileNotFoundError(f"Scaler not found: {scaler_path}")
            self._scaler = joblib.load(scaler_path)
            self._log("Scaler loaded.")

    def get_classes(self) -> list[str]:
        self._ensure_loaded()
        return list(self._classes or [])

    def process_frame(self, frame_bgr, threshold: float = 0.7):
        self._ensure_loaded()

        frame = frame_bgr
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._app._hands.process(frame_rgb)

        text = "—"
        conf = 0.0
        pred_prob = None
        hand_count = 0

        if results.multi_hand_landmarks:
            hand_count = len(results.multi_hand_landmarks)
            row = _extract_multi_hand_features(results, max_hands=MAX_HANDS_SUPPORTED)

            expected = None
            try:
                expected = int(getattr(self._scaler, "n_features_in_", None) or 0)
            except Exception:
                expected = None
            if expected and expected != len(row):
                raise ValueError(
                    f"Feature size mismatch: scaler expects {expected}, got {len(row)}. "
                    f"Reset CSV and retrain model for 1–4 hands."
                )

            X_scaled = self._scaler.transform(np.array(row, dtype=np.float32).reshape(1, -1))
            pred_prob = self._model.predict(X_scaled, verbose=0)[0]
            pred_class = int(np.argmax(pred_prob))
            max_prob = float(pred_prob[pred_class])

            if self._app.show_overlays:
                for hand_landmarks in results.multi_hand_landmarks:
                    self._mp_drawing.draw_landmarks(frame, hand_landmarks, self._mp_hands.HAND_CONNECTIONS)

            conf = max_prob
            if max_prob >= float(threshold):
                cls = self._classes[pred_class]
                text = " " if cls == "#" else str(cls)

        return text, conf, frame, pred_prob, hand_count
