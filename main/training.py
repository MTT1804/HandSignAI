import os
import pandas as pd
import joblib
import numpy as np
import psutil
import platform
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.config import list_physical_devices
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import shuffle
from locales import tr


class EpochProgressCallback(keras.callbacks.Callback):
    def __init__(self, total_epochs, progress_var, app_log_func, root):
        super().__init__()
        self.total_epochs = total_epochs
        self.progress_var = progress_var
        self.app_log_func = app_log_func
        self.root = root

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current_epoch = epoch + 1
        percent = 100.0 * current_epoch / self.total_epochs
        self.root.after(0, lambda: self.progress_var.set(percent))

        msg = tr(
            "log_epoch_progress",
            curr=current_epoch,
            total=self.total_epochs,
            loss=logs.get("loss", 0),
            acc=logs.get("accuracy", 0),
            vloss=logs.get("val_loss", 0),
            vacc=logs.get("val_accuracy", 0),
        )
        self.root.after(0, lambda: self.app_log_func(msg))


def run_training_in_thread(app):
    gpus = list_physical_devices('GPU')
    if gpus:
        gpu_names = [gpu.name for gpu in gpus]
        app.log(f"Training on GPU(s): {', '.join(gpu_names)}")
    else:
        app.log("Training on CPU")

    sys = platform.system()
    rel = platform.release()
    proc = platform.processor() or "unknown"
    cores = os.cpu_count()
    app.log(f"System: {sys} {rel} ({platform.machine()})")
    app.log(f"CPU: {proc}, cores: {cores}")

    if psutil:
        mem = psutil.virtual_memory()
        total_gb = mem.total / (1024**3)
        app.log(f"RAM total: {total_gb:.1f} GB")
    else:
        app.log("psutil not installed; skipping RAM info")
    csv_path = app.csv_file_var.get()
    model_path = app.model_file_var.get()
    scaler_path = app.scaler_file_var.get()
    test_size = app.test_size_var.get()
    random_state = app.random_state_var.get()
    epochs = app.epochs_var.get()
    batch_size = app.batch_size_var.get()
    patience = app.patience_var.get()

    if not os.path.exists(csv_path):
        app.log(tr("log_csv_missing_train", path=csv_path))
        return

    df = pd.read_csv(csv_path)
    if "index" in df.columns:
        df.drop(columns=["index"], inplace=True)
    if "label" not in df.columns:
        app.log(tr("log_label_missing_csv"))
        return

    X = df.drop("label", axis=1)
    y = df["label"]
    X, y = shuffle(X, y, random_state=random_state)

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )
    except ValueError as exc:
        app.log(tr("log_split_error", err=str(exc)))
        return

    y_train_encoded = pd.get_dummies(y_train)
    y_test_encoded = pd.get_dummies(y_test)

    label_map = {i: lab for i, lab in enumerate(y_train_encoded.columns)}
    print("Class map:", label_map)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)
    app.log(tr("log_scaler_saved", path=scaler_path))

    num_features = X_train_scaled.shape[1]
    num_classes = y_train_encoded.shape[1]

    model = keras.Sequential(
        [
            layers.Input(shape=(num_features,)),
            layers.Dense(128, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    app.log(tr("log_model_summary", summary=model.summary()))

    val_split = getattr(app, "val_split_var", None)
    val_split = val_split.get() if val_split else 0.1

    monitor_choice = getattr(app, "monitor_var", None)
    monitor_choice = monitor_choice.get() if monitor_choice else "val_loss"

    early_stop = EarlyStopping(
        monitor=monitor_choice,
        patience=patience,
        restore_best_weights=True
    )
    progress_callback = EpochProgressCallback(
        total_epochs=epochs,
        progress_var=app.progress_var,
        app_log_func=app.log,
        root=app.root,
    )

    history = model.fit(
        X_train_scaled,
        y_train_encoded,
        validation_split=val_split,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop, progress_callback],
        verbose=0,
    )


    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    app.log(tr("log_model_saved", path=model_path))

    test_loss, test_acc = model.evaluate(X_test_scaled, y_test_encoded, verbose=0)
    app.log(tr("log_test_accuracy", acc=f"{test_acc:.4f}"))

    y_pred_prob = model.predict(X_test_scaled)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(y_test_encoded.values, axis=1)

    cm = confusion_matrix(y_true, y_pred)
    app.log(tr("log_confusion_matrix", cm=cm))
    app.log(classification_report(y_true, y_pred))
    app.log(tr("log_training_finished"))
    app.root.after(0, lambda: app.progress_var.set(100.0))
    app.root.after(0, lambda hist=history: app.show_training_plots(hist))