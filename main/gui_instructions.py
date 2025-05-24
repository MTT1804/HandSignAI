import tkinter as tk
from tkinter import ttk, scrolledtext
from locales import tr  # i18n helper


def create_instructions_tab(app):
    app.instructions_box = scrolledtext.ScrolledText(
        app.tab_instructions, wrap=tk.WORD, font=("Roboto", 12)
    )
    app.instructions_box.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    _refresh_instructions(app)
    app.refresh_instructions = lambda: _refresh_instructions(app)


def _refresh_instructions(app):
    """(Re)fills the scrolled‑text box with translated manual."""
    box: scrolledtext.ScrolledText = app.instructions_box
    box.config(state=tk.NORMAL)
    box.delete("1.0", tk.END)
    box.insert(tk.END, tr("instructions_text"))
    box.config(state=tk.DISABLED)
