import tkinter as tk
from tkinter import ttk, messagebox
from utils import disable_space_activation
from locales import tr

def create_train_tab(app):
    app.train_main_frame = ttk.Frame(app.tab_train)
    app.train_main_frame.pack(fill=tk.BOTH, expand=True)

    app.train_cfg_frame = ttk.LabelFrame(app.train_main_frame, text=tr("frame_train_config"))
    app.train_cfg_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

    app.train_lbl_test_size = ttk.Label(app.train_cfg_frame, text=tr("lbl_test_size"))
    app.train_lbl_test_size.grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
    app.test_size_var = tk.DoubleVar(value=0.2)
    ttk.Entry(app.train_cfg_frame, textvariable=app.test_size_var, width=8) \
        .grid(row=0, column=1, sticky=tk.W)

    app.train_lbl_random_state = ttk.Label(app.train_cfg_frame, text=tr("lbl_random_state"))
    app.train_lbl_random_state.grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
    app.random_state_var = tk.IntVar(value=42)
    ttk.Entry(app.train_cfg_frame, textvariable=app.random_state_var, width=8) \
        .grid(row=1, column=1, sticky=tk.W)

    app.train_lbl_epochs = ttk.Label(app.train_cfg_frame, text=tr("lbl_epochs"))
    app.train_lbl_epochs.grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
    app.epochs_var = tk.IntVar(value=50)
    ttk.Entry(app.train_cfg_frame, textvariable=app.epochs_var, width=8) \
        .grid(row=2, column=1, sticky=tk.W)

    app.train_lbl_batch_size = ttk.Label(app.train_cfg_frame, text=tr("lbl_batch_size"))
    app.train_lbl_batch_size.grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
    app.batch_size_var = tk.IntVar(value=32)
    ttk.Entry(app.train_cfg_frame, textvariable=app.batch_size_var, width=8) \
        .grid(row=3, column=1, sticky=tk.W)

    app.train_lbl_patience = ttk.Label(app.train_cfg_frame, text=tr("lbl_patience"))
    app.train_lbl_patience.grid(row=4, column=0, sticky=tk.W, padx=5, pady=5)
    app.patience_var = tk.IntVar(value=5)
    ttk.Entry(app.train_cfg_frame, textvariable=app.patience_var, width=8) \
        .grid(row=4, column=1, sticky=tk.W)

    app.val_split_label = ttk.Label(app.train_cfg_frame, text=tr("lbl_validation_split"))
    app.val_split_label.grid(row=5, column=0, sticky=tk.W, padx=5, pady=5)
    app.val_split_var = tk.DoubleVar(value=0.1)
    ttk.Entry(app.train_cfg_frame, textvariable=app.val_split_var, width=8) \
        .grid(row=5, column=1, sticky=tk.W)

    app.monitor_label = ttk.Label(app.train_cfg_frame, text=tr("lbl_monitor"))
    app.monitor_label.grid(row=6, column=0, sticky=tk.W, padx=5, pady=5)
    app.monitor_var = tk.StringVar(value="val_loss")
    monitor_combo = ttk.Combobox(
        app.train_cfg_frame,
        textvariable=app.monitor_var,
        values=["val_loss", "val_accuracy"],
        state="readonly",
        width=12
    )
    monitor_combo.grid(row=6, column=1, sticky=tk.W)

    btn_frame = ttk.Frame(app.train_main_frame)
    btn_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

    original_start_training_cmd = app.start_training
    def validated_start_training():
        test_size = app.test_size_var.get()
        random_state = app.random_state_var.get()
        epochs = app.epochs_var.get()
        batch_size = app.batch_size_var.get()
        patience = app.patience_var.get()
        val_split = app.val_split_var.get()

        if not (0 < test_size < 1):
            messagebox.showerror(tr("dlg_error"), tr("err_test_size_range"))
            return
        if epochs <= 0:
            messagebox.showerror(tr("dlg_error"), tr("err_epochs_positive"))
            return
        if batch_size <= 0:
            messagebox.showerror(tr("dlg_error"), tr("err_batch_positive"))
            return
        if patience < 0:
            messagebox.showerror(tr("dlg_error"), tr("err_patience_nonnegative"))
            return
        if not (0 < val_split < 1):
            messagebox.showerror(tr("dlg_error"), tr("err_validation_split_range"))
            return

        original_start_training_cmd()

    app.train_start_btn = ttk.Button(
        btn_frame,
        text=tr("btn_start_training"),
        command=validated_start_training
    )
    app.train_start_btn.pack(side=tk.LEFT, padx=5)
    disable_space_activation(app.train_start_btn)

    app.progress_var = tk.DoubleVar(value=0.0)
    app.progress_bar = ttk.Progressbar(
        app.train_main_frame,
        variable=app.progress_var,
        orient="horizontal",
        length=300,
        mode="determinate",
    )
    app.progress_bar.pack(side=tk.TOP, padx=10, pady=5)
    app.plot_frame = ttk.LabelFrame(app.train_main_frame, text=tr("frame_train_plots"))
    app.plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
