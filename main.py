from __future__ import annotations

import queue
import threading
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from analysis_layers import (
    AnalysisBundle,
    build_analysis_bundle,
    export_analysis_json,
    export_graph_dot,
    format_evidence_report,
)
from corpus_cleaner import CleaningResult, clean_corpus_text, format_cleaning_report
from i18n import LANGUAGES, ui_text
from text_core import (
    SUPPORTED_EXTENSIONS,
    FileAnalysis,
    analyze_text,
    combine_analyses,
    extract_text,
    format_metrics,
    is_supported,
)


APP_NAME = "STAP"
APP_VERSION = "v0.4.0"
APP_FULL_NAME = "Scientific Text Analysis Platform"

THEME = {
    "bg": "#050806",
    "panel": "#0b120d",
    "panel_alt": "#101813",
    "fg": "#d7ffe6",
    "muted": "#75a889",
    "accent": "#39ff88",
    "accent_2": "#00d9ff",
    "select": "#123d25",
    "border": "#1d3a28",
}


class STAPApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(f"{APP_NAME} - {APP_FULL_NAME} {APP_VERSION}")
        self.root.geometry("1180x760")
        self.root.minsize(980, 640)

        self.files: list[Path] = []
        self.analyses: list[FileAnalysis] = []
        self.cleaning_results: dict[Path, CleaningResult] = {}
        self.analysis_bundle: AnalysisBundle | None = None
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.worker: threading.Thread | None = None

        self.language = "en"
        self.language_var = tk.StringVar(value=LANGUAGES[self.language])
        self.status_var = tk.StringVar(value=self.t("ready"))
        self.summary_var = tk.StringVar(value=self.t("no_files"))

        self._setup_theme()
        self._build_ui()
        self._poll_events()

    def t(self, key: str, **kwargs: object) -> str:
        return ui_text(self.language, key, **kwargs)

    def _setup_theme(self) -> None:
        self.root.configure(bg=THEME["bg"])
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(".", background=THEME["bg"], foreground=THEME["fg"], fieldbackground=THEME["panel"])
        style.configure("TFrame", background=THEME["bg"])
        style.configure("TLabelframe", background=THEME["bg"], foreground=THEME["accent"])
        style.configure("TLabel", background=THEME["bg"], foreground=THEME["fg"])
        style.configure("TButton", background=THEME["panel_alt"], foreground=THEME["accent"], bordercolor=THEME["border"], focusthickness=0, padding=(9, 5))
        style.map("TButton", background=[("active", THEME["select"])], foreground=[("active", THEME["fg"])])
        style.configure("TCombobox", fieldbackground=THEME["panel"], background=THEME["panel_alt"], foreground=THEME["fg"], arrowcolor=THEME["accent"])
        style.configure("Vertical.TScrollbar", background=THEME["panel_alt"], troughcolor=THEME["bg"], bordercolor=THEME["border"], arrowcolor=THEME["accent"])
        style.configure("Horizontal.TScrollbar", background=THEME["panel_alt"], troughcolor=THEME["bg"], bordercolor=THEME["border"], arrowcolor=THEME["accent"])
        style.configure("TPanedwindow", background=THEME["bg"])

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        toolbar = ttk.Frame(self.root, padding=(10, 10, 10, 6))
        toolbar.grid(row=0, column=0, sticky="ew")
        toolbar.columnconfigure(8, weight=1)

        self.btn_select = ttk.Button(toolbar, command=self.select_files)
        self.btn_analyze = ttk.Button(toolbar, command=self.start_analysis)
        self.btn_report = ttk.Button(toolbar, command=self.export_report)
        self.btn_clean = ttk.Button(toolbar, command=self.export_clean_text)
        self.btn_json = ttk.Button(toolbar, command=self.export_analysis_json)
        self.btn_graph = ttk.Button(toolbar, command=self.export_graph_dot)
        self.btn_clear = ttk.Button(toolbar, command=self.clear)

        self.btn_select.grid(row=0, column=0, padx=(0, 6))
        self.btn_analyze.grid(row=0, column=1, padx=6)
        self.btn_report.grid(row=0, column=2, padx=6)
        self.btn_clean.grid(row=0, column=3, padx=6)
        self.btn_json.grid(row=0, column=4, padx=6)
        self.btn_graph.grid(row=0, column=5, padx=6)
        self.btn_clear.grid(row=0, column=6, padx=6)

        self.language_label = ttk.Label(toolbar)
        self.language_label.grid(row=0, column=7, padx=(12, 4), sticky="e")
        self.language_combo = ttk.Combobox(toolbar, textvariable=self.language_var, values=list(LANGUAGES.values()), width=10, state="readonly")
        self.language_combo.grid(row=0, column=8, sticky="e")
        self.language_combo.bind("<<ComboboxSelected>>", self.change_language)
        ttk.Label(toolbar, textvariable=self.summary_var).grid(row=0, column=9, padx=(12, 0), sticky="e")

        body = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        body.grid(row=1, column=0, sticky="nsew", padx=10, pady=6)

        left = ttk.Frame(body)
        right = ttk.Frame(body)
        body.add(left, weight=1)
        body.add(right, weight=3)

        left.rowconfigure(1, weight=1)
        left.columnconfigure(0, weight=1)
        self.files_label = ttk.Label(left)
        self.files_label.grid(row=0, column=0, sticky="w")
        file_frame = ttk.Frame(left)
        file_frame.grid(row=1, column=0, sticky="nsew", pady=(4, 0))
        file_frame.rowconfigure(0, weight=1)
        file_frame.columnconfigure(0, weight=1)
        self.file_list = tk.Listbox(
            file_frame,
            height=12,
            bg=THEME["panel"],
            fg=THEME["fg"],
            selectbackground=THEME["select"],
            selectforeground=THEME["accent"],
            highlightbackground=THEME["border"],
            highlightcolor=THEME["accent"],
            borderwidth=1,
            font=("Consolas", 10),
        )
        file_scrollbar = ttk.Scrollbar(file_frame, orient=tk.VERTICAL, command=self.file_list.yview)
        self.file_list.configure(yscrollcommand=file_scrollbar.set)
        self.file_list.grid(row=0, column=0, sticky="nsew")
        file_scrollbar.grid(row=0, column=1, sticky="ns")
        self.file_list.bind("<<ListboxSelect>>", lambda _event: self.show_selected_file())

        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)
        self.analysis_label = ttk.Label(right)
        self.analysis_label.grid(row=0, column=0, sticky="w")
        output_frame = ttk.Frame(right)
        output_frame.grid(row=1, column=0, sticky="nsew", pady=(4, 0))
        output_frame.rowconfigure(0, weight=1)
        output_frame.columnconfigure(0, weight=1)
        self.output = tk.Text(
            output_frame,
            wrap="word",
            undo=False,
            bg=THEME["panel"],
            fg=THEME["fg"],
            insertbackground=THEME["accent"],
            selectbackground=THEME["select"],
            selectforeground=THEME["fg"],
            highlightbackground=THEME["border"],
            highlightcolor=THEME["accent"],
            borderwidth=1,
            font=("Consolas", 10),
        )
        output_y = ttk.Scrollbar(output_frame, orient=tk.VERTICAL, command=self.output.yview)
        output_x = ttk.Scrollbar(output_frame, orient=tk.HORIZONTAL, command=self.output.xview)
        self.output.configure(yscrollcommand=output_y.set, xscrollcommand=output_x.set)
        self.output.grid(row=0, column=0, sticky="nsew")
        output_y.grid(row=0, column=1, sticky="ns")
        output_x.grid(row=1, column=0, sticky="ew")

        status = ttk.Frame(self.root, padding=(10, 4, 10, 10))
        status.grid(row=2, column=0, sticky="ew")
        status.columnconfigure(0, weight=1)
        ttk.Label(status, textvariable=self.status_var).grid(row=0, column=0, sticky="w")
        self.apply_language()

    def apply_language(self) -> None:
        self.btn_select.configure(text=self.t("select_files"))
        self.btn_analyze.configure(text=self.t("clean_analyze"))
        self.btn_report.configure(text=self.t("export_report"))
        self.btn_clean.configure(text=self.t("export_clean"))
        self.btn_json.configure(text=self.t("export_json"))
        self.btn_graph.configure(text=self.t("export_graph"))
        self.btn_clear.configure(text=self.t("clear"))
        self.language_label.configure(text=self.t("language"))
        self.files_label.configure(text=self.t("corpus_files"))
        self.analysis_label.configure(text=self.t("analysis_evidence"))

    def change_language(self, _event: object | None = None) -> None:
        selected = self.language_var.get()
        self.language = next((code for code, name in LANGUAGES.items() if name == selected), "en")
        self.apply_language()
        if self.analyses:
            self.render_summary()
        else:
            self.summary_var.set(self.t("no_files"))
            self.status_var.set(self.t("ready"))

    def select_files(self) -> None:
        patterns = " ".join(f"*{ext}" for ext in sorted(SUPPORTED_EXTENSIONS))
        selected = filedialog.askopenfilenames(
            title=self.t("select_title"),
            filetypes=[
                (self.t("supported_documents"), patterns),
                (self.t("all_files"), "*.*"),
            ],
        )
        if not selected:
            return

        self.files = [Path(path) for path in selected if is_supported(Path(path))]
        skipped = len(selected) - len(self.files)
        self.analyses = []
        self.cleaning_results = {}
        self.analysis_bundle = None
        self._refresh_file_list()
        self.output.delete("1.0", tk.END)
        self.summary_var.set(self.t("selected_files", count=len(self.files)))
        self.status_var.set(self.t("skipped_files", count=skipped) if skipped else self.t("files_selected"))

    def start_analysis(self) -> None:
        if not self.files:
            messagebox.showinfo(APP_NAME, self.t("select_first"))
            return
        if self.worker and self.worker.is_alive():
            return

        self.status_var.set(self.t("cleaning"))
        self.output.delete("1.0", tk.END)
        self.worker = threading.Thread(target=self._analyze_worker, daemon=True)
        self.worker.start()

    def _analyze_worker(self) -> None:
        analyses: list[FileAnalysis] = []
        cleaning_results: dict[Path, CleaningResult] = {}

        for index, path in enumerate(self.files, start=1):
            self.events.put(("status", self.t("processing", index=index, total=len(self.files), name=path.name)))
            try:
                raw_text = extract_text(path)
                cleaning_result = clean_corpus_text(raw_text)
                cleaning_results[path] = cleaning_result
                metrics = analyze_text(cleaning_result.clean_text)
                analyses.append(FileAnalysis(path=path, text=cleaning_result.clean_text, metrics=metrics))
            except Exception as exc:  # noqa: BLE001 - GUI reports per-file failures.
                self.events.put(("error", f"{path.name}: {exc}"))

        bundle = build_analysis_bundle(analyses) if analyses else None
        self.events.put(("done", (analyses, cleaning_results, bundle)))

    def _poll_events(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "status":
                    self.status_var.set(str(payload))
                elif kind == "error":
                    self.output.insert(tk.END, f"ERROR: {payload}\n\n")
                elif kind == "done":
                    analyses, cleaning_results, bundle = payload  # type: ignore[misc]
                    self.analyses = list(analyses)
                    self.cleaning_results = dict(cleaning_results)
                    self.analysis_bundle = bundle
                    self.render_summary()
                    self.status_var.set(self.t("corpus_ready", count=len(self.analyses)))
        except queue.Empty:
            pass
        self.root.after(100, self._poll_events)

    def render_summary(self) -> None:
        self.output.delete("1.0", tk.END)
        if not self.analyses:
            self.output.insert(tk.END, self.t("no_text"))
            return

        combined = combine_analyses(self.analyses)
        removed_total = sum(len(result.removed) for result in self.cleaning_results.values())
        self.output.insert(tk.END, f"{APP_FULL_NAME}\n")
        self.output.insert(tk.END, "=" * 60 + "\n")
        self.output.insert(tk.END, f"{self.t('files')}: {len(self.analyses)}\n")
        self.output.insert(tk.END, f"{self.t('removed_lines')}: {removed_total}\n\n")

        self.output.insert(tk.END, f"{self.t('total_clean')}\n")
        self.output.insert(tk.END, "-" * 60 + "\n")
        self.output.insert(tk.END, format_metrics(combined))
        self.output.insert(tk.END, "\n\n")

        if self.analysis_bundle:
            self.output.insert(tk.END, format_evidence_report(self.analysis_bundle, self.language))
            self.output.insert(tk.END, "\n\n")

        for analysis in self.analyses:
            self.output.insert(tk.END, analysis.path.name + "\n")
            self.output.insert(tk.END, "-" * 60 + "\n")
            self.output.insert(tk.END, format_metrics(analysis.metrics))
            self.output.insert(tk.END, "\n\n")

        self.summary_var.set(self.t("clean_files", count=len(self.analyses)))

    def show_selected_file(self) -> None:
        selection = self.file_list.curselection()
        if not selection or not self.analyses:
            return
        index = selection[0]
        if index >= len(self.analyses):
            return

        analysis = self.analyses[index]
        cleaning_result = self.cleaning_results.get(analysis.path)
        self.output.delete("1.0", tk.END)
        self.output.insert(tk.END, analysis.path.name + "\n")
        self.output.insert(tk.END, "=" * 60 + "\n")
        self.output.insert(tk.END, format_metrics(analysis.metrics))

        if cleaning_result:
            self.output.insert(tk.END, "\n\n")
            self.output.insert(tk.END, format_cleaning_report(cleaning_result))

        if self.analysis_bundle:
            related = [
                concept
                for concept in self.analysis_bundle.concepts
                if any(item.source == analysis.path.name for item in concept.evidence)
            ][:12]
            self.output.insert(tk.END, f"\n\n{self.t('file_evidence')}\n")
            self.output.insert(tk.END, "-" * 60 + "\n")
            for concept in related:
                self.output.insert(tk.END, f"{concept.term} ({concept.count})\n")
                for item in concept.evidence:
                    if item.source == analysis.path.name:
                        self.output.insert(tk.END, f"- {self.t('sentence')} {item.sentence_index}: {item.text}\n")

        self.output.insert(tk.END, f"\n\n{self.t('clean_preview')}\n")
        self.output.insert(tk.END, "-" * 60 + "\n")
        self.output.insert(tk.END, analysis.text[:7000])

    def export_report(self) -> None:
        if not self.analyses:
            messagebox.showinfo(APP_NAME, self.t("run_first"))
            return

        default_name = f"stap_report_{datetime.now():%Y%m%d_%H%M%S}.txt"
        path = filedialog.asksaveasfilename(
            title=self.t("export_report_title"),
            defaultextension=".txt",
            initialfile=default_name,
            filetypes=[("Text report", "*.txt"), ("All files", "*.*")],
        )
        if not path:
            return

        Path(path).write_text(self.output.get("1.0", tk.END).strip() + "\n", encoding="utf-8")
        self.status_var.set(self.t("report_saved", path=path))

    def export_clean_text(self) -> None:
        if not self.analyses:
            messagebox.showinfo(APP_NAME, self.t("run_first"))
            return

        default_name = f"clean_corpus_{datetime.now():%Y%m%d_%H%M%S}.txt"
        path = filedialog.asksaveasfilename(
            title=self.t("export_clean_title"),
            defaultextension=".txt",
            initialfile=default_name,
            filetypes=[("Clean corpus text", "*.txt"), ("All files", "*.*")],
        )
        if not path:
            return

        parts = []
        for analysis in self.analyses:
            parts.append(f"===== {analysis.path.name} =====")
            parts.append(analysis.text)
        Path(path).write_text("\n\n".join(parts).strip() + "\n", encoding="utf-8")
        self.status_var.set(self.t("clean_saved", path=path))

    def export_analysis_json(self) -> None:
        if not self.analysis_bundle:
            messagebox.showinfo(APP_NAME, self.t("run_first"))
            return

        default_name = f"stap_analysis_{datetime.now():%Y%m%d_%H%M%S}.json"
        path = filedialog.asksaveasfilename(
            title=self.t("export_json_title"),
            defaultextension=".json",
            initialfile=default_name,
            filetypes=[("Analysis JSON", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return

        export_analysis_json(self.analysis_bundle, Path(path))
        self.status_var.set(self.t("json_saved", path=path))

    def export_graph_dot(self) -> None:
        if not self.analysis_bundle:
            messagebox.showinfo(APP_NAME, self.t("run_first"))
            return

        default_name = f"stap_knowledge_graph_{datetime.now():%Y%m%d_%H%M%S}.dot"
        path = filedialog.asksaveasfilename(
            title=self.t("export_graph_title"),
            defaultextension=".dot",
            initialfile=default_name,
            filetypes=[("Graphviz DOT", "*.dot"), ("All files", "*.*")],
        )
        if not path:
            return

        export_graph_dot(self.analysis_bundle.graph, Path(path))
        self.status_var.set(self.t("graph_saved", path=path))

    def clear(self) -> None:
        self.files = []
        self.analyses = []
        self.cleaning_results = {}
        self.analysis_bundle = None
        self.file_list.delete(0, tk.END)
        self.output.delete("1.0", tk.END)
        self.summary_var.set(self.t("no_files"))
        self.status_var.set(self.t("ready"))

    def _refresh_file_list(self) -> None:
        self.file_list.delete(0, tk.END)
        for path in self.files:
            self.file_list.insert(tk.END, path.name)


def main() -> None:
    root = tk.Tk()
    STAPApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
