from ctk_app import App


def main() -> None:
    app = App()
    app.protocol("WM_DELETE_WINDOW", app.on_app_close)
    app.mainloop()


if __name__ == "__main__":
    main()
