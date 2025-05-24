import tkinter as tk
from app_main import HandDataCollectorApp

def main():
    root = tk.Tk()
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    root.geometry(f"{screen_w}x{screen_h}+0+0")

    app = HandDataCollectorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
