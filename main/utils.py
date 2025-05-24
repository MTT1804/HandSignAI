import tkinter as tk

def disable_space_activation(widget):
    widget.bind("<space>", lambda e: "break")

class TextRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget
    def write(self, string):
        self.text_widget.insert(tk.END, string)
        self.text_widget.see(tk.END)
    def flush(self):
        pass
