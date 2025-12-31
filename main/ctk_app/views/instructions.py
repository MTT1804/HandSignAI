from __future__ import annotations

import re
import customtkinter as ctk
import tkinter as tk
import tkinter.font as tkfont

from locales import tr

class InstructionsView(ctk.CTkFrame):
    def __init__(self, master, app: App):
        super().__init__(master)
        self.app = app

        self._tags_ready = False

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        root = ctk.CTkFrame(self)
        root.grid(row=0, column=0, sticky="nsew", padx=18, pady=18)
        root.grid_rowconfigure(1, weight=1)
        root.grid_columnconfigure(0, weight=1)

        self.title_label = ctk.CTkLabel(root, text=tr("tab_instr"), font=ctk.CTkFont(size=18, weight="bold"))
        self.title_label.grid(
            row=0, column=0, sticky="w", padx=14, pady=(14, 8)
        )

        self.textbox = ctk.CTkTextbox(root, wrap="word")
        self.textbox.grid(row=1, column=0, sticky="nsew", padx=14, pady=(0, 14))

        self._refresh()

    def on_show(self) -> None:
        self._refresh()

    def _refresh(self) -> None:
        try:
            self.title_label.configure(text=tr("tab_instr"))
        except Exception:
            pass
        self._render_markdown(tr("instructions_text"))

    def _ensure_tags(self, txt: tk.Text) -> None:
        if self._tags_ready:
            return

        base = tkfont.nametofont("TkDefaultFont")
        base_size = int(base.cget("size"))

        h1 = tkfont.Font(txt, family=base.cget("family"), size=base_size + 8, weight="bold")
        h2 = tkfont.Font(txt, family=base.cget("family"), size=base_size + 5, weight="bold")
        h3 = tkfont.Font(txt, family=base.cget("family"), size=base_size + 2, weight="bold")
        bold = tkfont.Font(txt, family=base.cget("family"), size=base_size, weight="bold")
        mono = tkfont.nametofont("TkFixedFont")
        mono_small = tkfont.Font(txt, family=mono.cget("family"), size=base_size)
        quote = tkfont.Font(txt, family=base.cget("family"), size=base_size, slant="italic")

        txt.tag_configure("md_h1", font=h1, spacing1=12, spacing3=8)
        txt.tag_configure("md_h2", font=h2, spacing1=10, spacing3=6)
        txt.tag_configure("md_h3", font=h3, spacing1=8, spacing3=4)
        txt.tag_configure("md_bold", font=bold)
        txt.tag_configure("md_code", font=mono_small)
        txt.tag_configure("md_codeblock", font=mono_small, lmargin1=18, lmargin2=18, spacing1=4, spacing3=6)
        txt.tag_configure("md_quote", font=quote, lmargin1=18, lmargin2=18, spacing1=2, spacing3=6)
        txt.tag_configure("md_list", lmargin1=18, lmargin2=30)
        txt.tag_configure("md_table", font=mono_small, lmargin1=8, lmargin2=8, spacing1=2, spacing3=6)
        txt.tag_configure("md_table_header", font=bold)

        self._tags_ready = True

    def _insert_inline(self, txt: tk.Text, s: str, base_tags: tuple[str, ...] = ()) -> None:
        parts = s.split("`")
        for i, part in enumerate(parts):
            if i % 2 == 1:
                if part:
                    txt.insert("end", part, (*base_tags, "md_code"))
                continue

            cursor = 0
            for m in re.finditer(r"\*\*(.+?)\*\*", part):
                if m.start() > cursor:
                    txt.insert("end", part[cursor:m.start()], base_tags)
                txt.insert("end", m.group(1), (*base_tags, "md_bold"))
                cursor = m.end()
            if cursor < len(part):
                txt.insert("end", part[cursor:], base_tags)

    def _render_table_block(self, lines: list[str]) -> list[tuple[str, tuple[str, ...]]]:
        rows: list[list[str]] = []
        for ln in lines:
            raw = ln.strip()
            if not raw.startswith("|"):
                continue
            raw = raw.strip("|")
            cols = [c.strip() for c in raw.split("|")]
            rows.append(cols)
        if not rows:
            return []

        col_count = max(len(r) for r in rows)
        for r in rows:
            while len(r) < col_count:
                r.append("")

        widths = [0] * col_count
        for r in rows:
            for ci, cell in enumerate(r):
                widths[ci] = max(widths[ci], len(cell))

        def fmt_row(r: list[str]) -> str:
            cells = [r[ci].ljust(widths[ci]) for ci in range(col_count)]
            return " | ".join(cells).rstrip() + "\n"

        out: list[tuple[str, tuple[str, ...]]] = []

        header = rows[0]
        out.append((fmt_row(header), ("md_table", "md_table_header")))
        sep = "-+-".join("-" * w for w in widths).rstrip() + "\n"
        out.append((sep, ("md_table",)))

        for r in rows[2:] if len(rows) >= 2 else rows[1:]:
            out.append((fmt_row(r), ("md_table",)))

        out.append(("\n", ()))
        return out

    def _render_markdown(self, md: str) -> None:
        txt: tk.Text = self.textbox._textbox
        txt.configure(state="normal")
        txt.delete("1.0", "end")
        self._ensure_tags(txt)

        lines = (md or "").splitlines()
        i = 0
        in_code = False
        while i < len(lines):
            line = lines[i]
            stripped = line.rstrip("\n")
            s = stripped.strip()

            if s.startswith("```"):
                in_code = not in_code
                i += 1
                continue

            if in_code:
                txt.insert("end", stripped + "\n", ("md_codeblock",))
                i += 1
                continue

            if s.startswith("|") and i + 1 < len(lines) and "|" in lines[i + 1] and set(lines[i + 1].replace("|", "").strip()) <= {"-", ":", " "}:
                table_lines = []
                j = i
                while j < len(lines) and lines[j].strip().startswith("|"):
                    table_lines.append(lines[j])
                    j += 1
                for text, tags in self._render_table_block(table_lines):
                    txt.insert("end", text, tags)
                i = j
                continue

            if s in {"---", "***", "___"}:
                txt.insert("end", "―" * 50 + "\n\n", ("md_codeblock",))
                i += 1
                continue

            if s.startswith("#"):
                level = len(s) - len(s.lstrip("#"))
                title = s[level:].strip()
                tag = "md_h1" if level == 1 else "md_h2" if level == 2 else "md_h3"
                txt.insert("end", title + "\n", (tag,))
                txt.insert("end", "\n")
                i += 1
                continue

            if s.startswith(">"):
                body = s[1:].lstrip()
                self._insert_inline(txt, body, ("md_quote",))
                txt.insert("end", "\n\n", ("md_quote",))
                i += 1
                continue

            if s.startswith(("- ", "* ", "+ ")):
                body = s[2:]
                txt.insert("end", "• ", ("md_list",))
                self._insert_inline(txt, body, ("md_list",))
                txt.insert("end", "\n", ("md_list",))
                i += 1
                while i < len(lines):
                    nxt = lines[i].strip()
                    if not nxt.startswith(("- ", "* ", "+ ")):
                        break
                    body = nxt[2:]
                    txt.insert("end", "• ", ("md_list",))
                    self._insert_inline(txt, body, ("md_list",))
                    txt.insert("end", "\n", ("md_list",))
                    i += 1
                txt.insert("end", "\n")
                continue

            if s == "":
                txt.insert("end", "\n")
                i += 1
                continue

            self._insert_inline(txt, stripped)
            txt.insert("end", "\n")
            i += 1

        txt.configure(state="disabled")


