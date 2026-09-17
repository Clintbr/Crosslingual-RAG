"""
    This main app entry point plays a monitor role in this evaluation system.
    It allows running every step of the protocol.
    If the defined step is not respected, running a step shows a warning and ask for a confirmation:
        "Some problems could occur running that step now, are you sure to continue?"
    If critical steps are to be run (steps 6 and 7), it shows a warning and ask for a confirmation:
        "This step take very long to be completed(~4 days), are you sure to continue?"
"""

import sys
import os
import io
import time
import queue
import threading
import argparse
from typing import Callable, Optional

import tkinter as tk
from tkinter import ttk, messagebox, filedialog


# Step 1: data fetching from xquad
def run_load_data_xquad():
    from src.ingestion.loading.load_xquad_en_de import run_loader_xquad
    run_loader_xquad()

# Step 2: data fetching from squad_fr
def run_load_data_squad_fr():
    from src.ingestion.loading.load_squad_fr import run_loader_squad
    run_loader_squad()

# Step 3: preparing documents for dbs (mongo and qdrant)
def run_merger_and_converter():
    from src.ingestion.converting.converter import merge_and_convert_docs
    merge_and_convert_docs()


# Step 4: data ingestion
def run_ingestion():
    from src.ingestion.ingestor import start_ingestion
    start_ingestion()


# Step 5: prepare questions
def run_prepare_questions():
    from src.evaluating.prepared_questions.prepare_questions import prepare_questions_for_evaluation
    prepare_questions_for_evaluation()


# Step 6: run rag methods with prepared questions
def run_rag_methods():
    from src.evaluating.ragas.run_rag_pipeline import run_pipeline
    from src.evaluating.run_eval_pipeline import pipelines
    for pipeline in pipelines:
        run_pipeline(
            method_name=pipeline["method"],
            input_json=pipeline["input"],
            output_json=pipeline["store"],
        )


# Step 7: run ragas evaluation
def run_ragas_evaluation():
    from src.evaluating.ragas.ragas_eval import evaluate_ragas_pipeline
    from src.evaluating.run_eval_pipeline import pipelines
    for pipeline in pipelines:
        evaluate_ragas_pipeline(
            input_json=pipeline["store"],
            output_csv=pipeline["output"],
        )


# Step 8: save hardware metrics
def run_save_hardware_metrics():
    from src.evaluating.hardware.save_hardware_results import save_hardware_metrics
    from src.evaluating.run_eval_pipeline import pipelines
    for pipeline in pipelines:
        save_hardware_metrics(
            input_json=pipeline["store"],
            output_csv=pipeline["hardware"],
        )


# Step 9: analyse results
def run_analyse_rag_tests_results():
    from src.evaluating.analysis.analyser import trigger_analyse_and_visualise
    from src.evaluating.run_eval_pipeline import dataset
    trigger_analyse_and_visualise(dataset=dataset)


steps = [
    {
        "step": 1,
        "name": "Fetch XQuAD",
        "description": "data fetching from xquad",
        "phase": "Ingestion",
        "enabled": True,
        "is_critical": False,
        "func": run_load_data_xquad,
    },
    {
        "step": 2,
        "name": "Fetch SQuAD FR",
        "description": "data fetching from squad_fr",
        "phase": "Ingestion",
        "enabled": True,
        "is_critical": False,
        "func": run_load_data_squad_fr,
    },
    {
        "step": 3,
        "name": "Prepare Documents",
        "description": "preparing documents for dbs (mongo and qdrant)",
        "phase": "Ingestion",
        "enabled": True,
        "is_critical": False,
        "func": run_merger_and_converter,
    },
    {
        "step": 4,
        "name": "Data Ingestion",
        "description": "data ingestion",
        "phase": "Ingestion",
        "enabled": True,
        "is_critical": False,
        "func": run_ingestion,
    },
    {
        "step": 5,
        "name": "Prepare Questions",
        "description": "prepare questions",
        "phase": "Evaluation",
        "enabled": True,
        "is_critical": False,
        "func": run_prepare_questions,
    },
    {
        "step": 6,
        "name": "Run RAG Methods",
        "description": "run rag methods with prepared questions",
        "phase": "Evaluation",
        "enabled": True,
        "is_critical": True,
        "func": run_rag_methods,
    },
    {
        "step": 7,
        "name": "Ragas Evaluation",
        "description": "run ragas evaluation",
        "phase": "Evaluation",
        "enabled": True,
        "is_critical": True,
        "func": run_ragas_evaluation,
    },
    {
        "step": 8,
        "name": "Save Hardware Metrics",
        "description": "save hardware metrics",
        "phase": "Evaluation",
        "enabled": True,
        "is_critical": False,
        "func": run_save_hardware_metrics,
    },
    {
        "step": 9,
        "name": "Analyse Results",
        "description": "analyse results",
        "phase": "Evaluation",
        "enabled": True,
        "is_critical": False,
        "func": run_analyse_rag_tests_results,
    },
]

class QueueStream(io.TextIOBase):
    """Custom stream that writes to both the original stream and a thread-safe Queue."""

    def __init__(self, target_queue: queue.Queue, original_stream, stream_tag="INFO"):
        super().__init__()
        self.queue = target_queue
        self.original_stream = original_stream
        self.stream_tag = stream_tag

    def write(self, text: str) -> int:
        if self.original_stream:
            try:
                self.original_stream.write(text)
                self.original_stream.flush()
            except Exception:
                pass
        if text:
            self.queue.put((self.stream_tag, text))
        return len(text)

    def flush(self):
        if self.original_stream:
            try:
                self.original_stream.flush()
            except Exception:
                pass

class MonitorGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Crosslingual-RAG Protocol Monitor")
        self.root.geometry("1180x780")
        self.root.minsize(980, 640)

        # Execution state
        self.log_queue = queue.Queue()
        self.completed_steps = set()
        self.is_running = False
        self.stop_requested = False
        self.current_running_step: Optional[int] = None
        self.start_time: Optional[float] = None

        # Data model
        self.step_vars = {}        # step_num -> tk.BooleanVar
        self.step_status_labels = {} # step_num -> tk.Label
        self.step_run_buttons = {}   # step_num -> ttk.Button
        self.step_cards = {}         # step_num -> tk.Frame

        # Stream redirection
        self.orig_stdout = sys.stdout
        self.orig_stderr = sys.stderr
        sys.stdout = QueueStream(self.log_queue, self.orig_stdout, "INFO")
        sys.stderr = QueueStream(self.log_queue, self.orig_stderr, "ERROR")

        # Configure GUI layout & styling
        self._setup_styles()
        self._build_ui()

        # Start background polling for logs and timer
        self.root.after(40, self._process_log_queue)
        self.root.after(500, self._update_timer)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Welcome message
        print("=" * 60)
        print("  Crosslingual-RAG Protocol Monitor Ready")
        print("  Select steps and click 'Run All Enabled' or 'Run' on a step.")
        print("=" * 60)

    def _setup_styles(self):
        # Color palette: Clean modern light UI with dark terminal
        self.colors = {
            "bg_main": "#f3f4f6",
            "bg_card": "#ffffff",
            "bg_card_alt": "#f9fafb",
            "border": "#e5e7eb",
            "text_primary": "#111827",
            "text_secondary": "#4b5563",
            "text_muted": "#9ca3af",
            "primary": "#2563eb",
            "primary_hover": "#1d4ed8",
            "success": "#10b981",
            "success_bg": "#ecfdf5",
            "warning": "#f59e0b",
            "warning_bg": "#fffbeb",
            "danger": "#ef4444",
            "danger_bg": "#fef2f2",
            "info_badge": "#e0f2fe",
            "info_text": "#0369a1",
            "term_bg": "#0f172a",
            "term_fg": "#f8fafc",
            "term_cursor": "#38bdf8",
        }

        self.root.configure(bg=self.colors["bg_main"])
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

        style.configure(".", background=self.colors["bg_main"], font=("Segoe UI", 9))
        style.configure("TProgressbar", thickness=10, troughcolor="#e5e7eb", background="#2563eb")
        style.configure("Accent.TButton", font=("Segoe UI", 9, "bold"), padding=6)
        style.configure("Step.TButton", font=("Segoe UI", 8), padding=3)

    def _build_ui(self):
        # Main vertical container
        main_frame = tk.Frame(self.root, bg=self.colors["bg_main"])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=14, pady=12)

        # 1. Header Bar
        header = tk.Frame(main_frame, bg=self.colors["bg_card"], bd=1, relief=tk.SOLID, highlightthickness=0)
        header.configure(highlightbackground=self.colors["border"])
        header.pack(fill=tk.X, pady=(0, 10))

        header_inner = tk.Frame(header, bg=self.colors["bg_card"], padx=16, pady=10)
        header_inner.pack(fill=tk.X)

        title_frame = tk.Frame(header_inner, bg=self.colors["bg_card"])
        title_frame.pack(side=tk.LEFT)

        title_lbl = tk.Label(
            title_frame,
            text="Crosslingual-RAG Evaluation Monitor",
            font=("Segoe UI", 15, "bold"),
            fg=self.colors["text_primary"],
            bg=self.colors["bg_card"],
        )
        title_lbl.pack(anchor="w")

        subtitle_lbl = tk.Label(
            title_frame,
            text="Interactive protocol pipeline executor, step dependency validation & live trace monitor",
            font=("Segoe UI", 9),
            fg=self.colors["text_secondary"],
            bg=self.colors["bg_card"],
        )
        subtitle_lbl.pack(anchor="w")

        # Header status pills
        stats_frame = tk.Frame(header_inner, bg=self.colors["bg_card"])
        stats_frame.pack(side=tk.RIGHT)

        self.status_pill = tk.Label(
            stats_frame,
            text="  READY  ",
            font=("Segoe UI", 9, "bold"),
            bg="#dcfce7",
            fg="#15803d",
            padx=8,
            pady=4,
            relief=tk.FLAT,
        )
        self.status_pill.pack(side=tk.LEFT, padx=4)

        self.progress_pill = tk.Label(
            stats_frame,
            text="Completed: 0 / 9",
            font=("Segoe UI", 9),
            bg="#f1f5f9",
            fg="#334155",
            padx=8,
            pady=4,
        )
        self.progress_pill.pack(side=tk.LEFT, padx=4)

        # 2. Main 2-Column Split Area
        content_panes = tk.Frame(main_frame, bg=self.colors["bg_main"])
        content_panes.pack(fill=tk.BOTH, expand=True)

        # Left Column: Steps panel
        left_panel = tk.Frame(content_panes, bg=self.colors["bg_card"], bd=1, relief=tk.SOLID)
        left_panel.configure(highlightbackground=self.colors["border"])
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 6))
        left_panel.config(width=490)
        left_panel.pack_propagate(False)

        self._build_steps_panel(left_panel)

        # Right Column: Terminal Trace Panel
        right_panel = tk.Frame(content_panes, bg=self.colors["term_bg"], bd=1, relief=tk.SOLID)
        right_panel.configure(highlightbackground=self.colors["border"])
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(6, 0))

        self._build_terminal_panel(right_panel)

        # 3. Bottom Status Bar
        bottom_bar = tk.Frame(main_frame, bg=self.colors["bg_card"], bd=1, relief=tk.SOLID, height=36)
        bottom_bar.pack(fill=tk.X, pady=(10, 0))

        bottom_inner = tk.Frame(bottom_bar, bg=self.colors["bg_card"], padx=10, pady=4)
        bottom_inner.pack(fill=tk.X)

        self.bottom_status_lbl = tk.Label(
            bottom_inner,
            text="System Idle. Click a step or 'Run All Enabled' to begin.",
            font=("Segoe UI", 9),
            fg=self.colors["text_secondary"],
            bg=self.colors["bg_card"],
        )
        self.bottom_status_lbl.pack(side=tk.LEFT)

        self.timer_lbl = tk.Label(
            bottom_inner,
            text="Elapsed: 00:00:00",
            font=("Consolas", 9),
            fg=self.colors["text_muted"],
            bg=self.colors["bg_card"],
        )
        self.timer_lbl.pack(side=tk.RIGHT, padx=8)

        self.global_progress = ttk.Progressbar(bottom_inner, mode="determinate", length=180)
        self.global_progress.pack(side=tk.RIGHT, padx=8)
        self.global_progress["value"] = 0

    def _build_steps_panel(self, parent: tk.Frame):
        # Steps Toolbar
        toolbar = tk.Frame(parent, bg="#f8fafc", padx=10, pady=8)
        toolbar.pack(fill=tk.X)

        tb_title = tk.Label(
            toolbar,
            text="PROTOCOL STEPS",
            font=("Segoe UI", 9, "bold"),
            fg=self.colors["text_primary"],
            bg="#f8fafc",
        )
        tb_title.pack(side=tk.LEFT)

        self.btn_run_all = tk.Button(
            toolbar,
            text="▶ Run All Enabled",
            font=("Segoe UI", 9, "bold"),
            bg="#2563eb",
            fg="white",
            activebackground="#1d4ed8",
            activeforeground="white",
            relief=tk.FLAT,
            padx=10,
            pady=3,
            cursor="hand2",
            command=self.run_all_enabled,
        )
        self.btn_run_all.pack(side=tk.RIGHT, padx=2)

        self.btn_stop = tk.Button(
            toolbar,
            text="⏹ Stop",
            font=("Segoe UI", 9, "bold"),
            bg="#ef4444",
            fg="white",
            activebackground="#dc2626",
            activeforeground="white",
            relief=tk.FLAT,
            padx=8,
            pady=3,
            state=tk.DISABLED,
            cursor="hand2",
            command=self.request_stop,
        )
        self.btn_stop.pack(side=tk.RIGHT, padx=2)

        # Sub-toolbar: Selection shortcuts
        sub_tb = tk.Frame(parent, bg=self.colors["bg_card"], padx=10, pady=4)
        sub_tb.pack(fill=tk.X)

        btn_select_all = tk.Button(
            sub_tb,
            text="Select All",
            font=("Segoe UI", 8),
            relief=tk.FLAT,
            bg="#f1f5f9",
            fg="#475569",
            padx=6,
            pady=1,
            command=self.select_all_steps,
        )
        btn_select_all.pack(side=tk.LEFT, padx=(0, 4))

        btn_deselect_all = tk.Button(
            sub_tb,
            text="Deselect All",
            font=("Segoe UI", 8),
            relief=tk.FLAT,
            bg="#f1f5f9",
            fg="#475569",
            padx=6,
            pady=1,
            command=self.deselect_all_steps,
        )
        btn_deselect_all.pack(side=tk.LEFT, padx=4)

        btn_reset_status = tk.Button(
            sub_tb,
            text="↺ Reset Status",
            font=("Segoe UI", 8),
            relief=tk.FLAT,
            bg="#f1f5f9",
            fg="#475569",
            padx=6,
            pady=1,
            command=self.reset_step_statuses,
        )
        btn_reset_status.pack(side=tk.RIGHT)

        ttk.Separator(parent, orient="horizontal").pack(fill=tk.X, pady=(2, 4))

        # Scrollable container for step list
        canvas_container = tk.Frame(parent, bg=self.colors["bg_card"])
        canvas_container.pack(fill=tk.BOTH, expand=True, padx=6, pady=2)

        canvas = tk.Canvas(canvas_container, bg=self.colors["bg_card"], highlightthickness=0)
        scrollbar = ttk.Scrollbar(canvas_container, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=self.colors["bg_card"])

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )

        canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

        def _on_canvas_configure(event):
            canvas.itemconfig(canvas_window, width=event.width)

        canvas.bind("<Configure>", _on_canvas_configure)
        canvas.configure(xscrollcommand=None, yscrollcommand=scrollbar.set)

        # Enable mouse wheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Group steps by phase
        last_phase = None
        for step_info in steps:
            phase = step_info["phase"]
            if phase != last_phase:
                last_phase = phase
                phase_header = tk.Frame(scrollable_frame, bg="#e2e8f0", padx=8, pady=3)
                phase_header.pack(fill=tk.X, pady=(6, 2))
                phase_lbl = tk.Label(
                    phase_header,
                    text=f"Phase: {phase.upper()}",
                    font=("Segoe UI", 8, "bold"),
                    fg="#334155",
                    bg="#e2e8f0",
                )
                phase_lbl.pack(anchor="w")

            self._create_step_card(scrollable_frame, step_info)

    def _create_step_card(self, parent: tk.Frame, step_info: dict):
        step_num = step_info["step"]
        var = tk.BooleanVar(value=step_info.get("enabled", True))
        self.step_vars[step_num] = var

        card = tk.Frame(
            parent,
            bg=self.colors["bg_card"],
            bd=1,
            relief=tk.SOLID,
            padx=8,
            pady=6,
        )
        card.configure(highlightbackground=self.colors["border"])
        card.pack(fill=tk.X, pady=3, padx=2)
        self.step_cards[step_num] = card

        # Top row of card: Checkbox + Step Badge + Step Name + Run Button
        top_row = tk.Frame(card, bg=self.colors["bg_card"])
        top_row.pack(fill=tk.X)

        cb = tk.Checkbutton(
            top_row,
            variable=var,
            bg=self.colors["bg_card"],
            activebackground=self.colors["bg_card"],
            command=self._update_progress_summary,
        )
        cb.pack(side=tk.LEFT, padx=(0, 4))

        badge_bg = "#fee2e2" if step_info.get("is_critical") else "#e0f2fe"
        badge_fg = "#b91c1c" if step_info.get("is_critical") else "#0369a1"

        badge = tk.Label(
            top_row,
            text=f"Step {step_num}",
            font=("Segoe UI", 8, "bold"),
            bg=badge_bg,
            fg=badge_fg,
            padx=5,
            pady=1,
        )
        badge.pack(side=tk.LEFT, padx=(0, 6))

        title_lbl = tk.Label(
            top_row,
            text=step_info["name"],
            font=("Segoe UI", 9, "bold"),
            fg=self.colors["text_primary"],
            bg=self.colors["bg_card"],
        )
        title_lbl.pack(side=tk.LEFT)

        if step_info.get("is_critical"):
            crit_badge = tk.Label(
                top_row,
                text="⚠️ ~4 days",
                font=("Segoe UI", 7, "bold"),
                bg="#fef3c7",
                fg="#92400e",
                padx=4,
                pady=1,
            )
            crit_badge.pack(side=tk.LEFT, padx=4)

        btn_run = tk.Button(
            top_row,
            text="▶ Run",
            font=("Segoe UI", 8),
            bg="#f1f5f9",
            fg="#1e293b",
            activebackground="#e2e8f0",
            relief=tk.FLAT,
            padx=8,
            pady=1,
            cursor="hand2",
            command=lambda s=step_num: self.run_single_step(s),
        )
        btn_run.pack(side=tk.RIGHT)
        self.step_run_buttons[step_num] = btn_run

        # Bottom row of card: Description and status badge
        bottom_row = tk.Frame(card, bg=self.colors["bg_card"])
        bottom_row.pack(fill=tk.X, pady=(4, 0))

        desc_lbl = tk.Label(
            bottom_row,
            text=step_info["description"],
            font=("Segoe UI", 8),
            fg=self.colors["text_secondary"],
            bg=self.colors["bg_card"],
            anchor="w",
        )
        desc_lbl.pack(side=tk.LEFT)

        status_lbl = tk.Label(
            bottom_row,
            text="Pending",
            font=("Segoe UI", 7, "bold"),
            bg="#f1f5f9",
            fg="#64748b",
            padx=6,
            pady=1,
        )
        status_lbl.pack(side=tk.RIGHT)
        self.step_status_labels[step_num] = status_lbl

    def _build_terminal_panel(self, parent: tk.Frame):
        # Header for terminal trace
        term_header = tk.Frame(parent, bg="#1e293b", padx=10, pady=6)
        term_header.pack(fill=tk.X)

        term_title = tk.Label(
            term_header,
            text="⚡ TERMINAL EXECUTION TRACE",
            font=("Consolas", 10, "bold"),
            fg="#38bdf8",
            bg="#1e293b",
        )
        term_title.pack(side=tk.LEFT)

        # Terminal action buttons
        self.autoscroll_var = tk.BooleanVar(value=True)
        cb_auto = tk.Checkbutton(
            term_header,
            text="Auto-scroll",
            variable=self.autoscroll_var,
            font=("Segoe UI", 8),
            fg="#cbd5e1",
            bg="#1e293b",
            activebackground="#1e293b",
            activeforeground="#ffffff",
            selectcolor="#0f172a",
        )
        cb_auto.pack(side=tk.RIGHT, padx=4)

        btn_clear = tk.Button(
            term_header,
            text="Clear",
            font=("Segoe UI", 8),
            bg="#334155",
            fg="#f8fafc",
            activebackground="#475569",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=6,
            pady=1,
            cursor="hand2",
            command=self.clear_terminal,
        )
        btn_clear.pack(side=tk.RIGHT, padx=4)

        btn_save = tk.Button(
            term_header,
            text="Save Log",
            font=("Segoe UI", 8),
            bg="#334155",
            fg="#f8fafc",
            activebackground="#475569",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=6,
            pady=1,
            cursor="hand2",
            command=self.save_log_to_file,
        )
        btn_save.pack(side=tk.RIGHT, padx=4)

        # Scrolled Text Terminal area
        text_frame = tk.Frame(parent, bg=self.colors["term_bg"])
        text_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        self.term_text = tk.Text(
            text_frame,
            wrap=tk.WORD,
            bg=self.colors["term_bg"],
            fg=self.colors["term_fg"],
            insertbackground=self.colors["term_cursor"],
            font=("Consolas", 9),
            padx=8,
            pady=8,
            bd=0,
            highlightthickness=0,
        )
        term_scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=self.term_text.yview)
        self.term_text.configure(yscrollcommand=term_scrollbar.set)

        self.term_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        term_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Syntax / Color tags
        self.term_text.tag_config("INFO", foreground="#f8fafc")
        self.term_text.tag_config("ERROR", foreground="#f87171")
        self.term_text.tag_config("WARNING", foreground="#fbbf24")
        self.term_text.tag_config("SUCCESS", foreground="#34d399")
        self.term_text.tag_config("STEP", foreground="#38bdf8")
        self.term_text.tag_config("SYSTEM", foreground="#c084fc")

    def _process_log_queue(self):
        """Drains the log queue and updates the terminal trace widget."""
        max_lines_per_batch = 100
        count = 0
        while not self.log_queue.empty() and count < max_lines_per_batch:
            try:
                tag, text = self.log_queue.get_nowait()
                count += 1
                self._append_to_terminal(text, tag)
            except queue.Empty:
                break

        self.root.after(40, self._process_log_queue)

    def _append_to_terminal(self, text: str, default_tag="INFO"):
        # Handle carriage returns \r for progress bars (e.g. tqdm)
        if "\r" in text and not text.endswith("\n"):
            lines = text.split("\r")
            text = lines[-1]

        # Determine tag highlight based on content if not explicitly error
        tag = default_tag
        text_lower = text.lower()
        if "error" in text_lower or "exception" in text_lower or "traceback" in text_lower or "failed" in text_lower:
            tag = "ERROR"
        elif "warning" in text_lower or "warn" in text_lower:
            tag = "WARNING"
        elif "success" in text_lower or "finished" in text_lower or "done" in text_lower or "saved" in text_lower:
            tag = "SUCCESS"
        elif text.startswith("===") or text.startswith("[STEP"):
            tag = "STEP"

        self.term_text.insert(tk.END, text, tag)

        # Keep buffer limited to 10,000 lines
        line_count = int(self.term_text.index("end-1c").split(".")[0])
        if line_count > 10000:
            self.term_text.delete("1.0", f"{line_count - 8000}.0")

        if self.autoscroll_var.get():
            self.term_text.see(tk.END)

    def clear_terminal(self):
        self.term_text.delete("1.0", tk.END)

    def save_log_to_file(self):
        content = self.term_text.get("1.0", tk.END)
        file_path = filedialog.asksaveasfilename(
            defaultextension=".log",
            filetypes=[("Log files", "*.log"), ("Text files", "*.txt"), ("All files", "*.*")],
            title="Save Terminal Trace",
        )
        if file_path:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            messagebox.showinfo("Log Saved", f"Terminal trace saved to:\n{file_path}")

    def set_step_status(self, step_num: int, status: str):
        lbl = self.step_status_labels.get(step_num)
        card = self.step_cards.get(step_num)
        if not lbl or not card:
            return

        status = status.lower()
        if status == "running":
            lbl.config(text="Running...", bg="#fef3c7", fg="#b45309")
            card.config(bg="#fefce8")
        elif status == "completed":
            lbl.config(text="Completed ✓", bg="#dcfce7", fg="#15803d")
            card.config(bg="#f0fdf4")
            self.completed_steps.add(step_num)
        elif status == "failed":
            lbl.config(text="Failed ✗", bg="#fee2e2", fg="#b91c1c")
            card.config(bg="#fef2f2")
        elif status == "skipped":
            lbl.config(text="Skipped", bg="#f1f5f9", fg="#94a3b8")
            card.config(bg=self.colors["bg_card"])
        else: # pending / ready
            lbl.config(text="Pending", bg="#f1f5f9", fg="#64748b")
            card.config(bg=self.colors["bg_card"])

        self._update_progress_summary()

    def reset_step_statuses(self):
        if self.is_running:
            return
        self.completed_steps.clear()
        for s in steps:
            self.set_step_status(s["step"], "pending")
        self._update_progress_summary()
        self.bottom_status_lbl.config(text="Step statuses reset.")

    def select_all_steps(self):
        for var in self.step_vars.values():
            var.set(True)
        self._update_progress_summary()

    def deselect_all_steps(self):
        for var in self.step_vars.values():
            var.set(False)
        self._update_progress_summary()

    def _update_progress_summary(self):
        total_enabled = sum(1 for s in steps if self.step_vars[s["step"]].get())
        completed_count = len(self.completed_steps)
        self.progress_pill.config(text=f"Completed: {completed_count} / {len(steps)}")

        if len(steps) > 0:
            percent = (completed_count / len(steps)) * 100
            self.global_progress["value"] = percent

    def _update_timer(self):
        if self.is_running and self.start_time:
            elapsed = time.time() - self.start_time
            hrs = int(elapsed // 3600)
            mins = int((elapsed % 3600) // 60)
            secs = int(elapsed % 60)
            self.timer_lbl.config(text=f"Elapsed: {hrs:02d}:{mins:02d}:{secs:02d}")
        self.root.after(500, self._update_timer)

    def validate_and_confirm_step(self, step_num: int) -> bool:
        step_entry = next((s for s in steps if s["step"] == step_num), None)
        if not step_entry:
            return False

        # Rule 1: Step sequence verification
        # Check if all enabled previous steps (1 to step_num-1) are completed
        prior_missing = [
            s["step"]
            for s in steps
            if s["step"] < step_num
            and self.step_vars[s["step"]].get()
            and s["step"] not in self.completed_steps
        ]

        if prior_missing:
            msg = (
                f"Step {step_num} ({step_entry['name']}) is being run out of order.\n"
                f"Prior enabled step(s) {prior_missing} have not completed yet.\n\n"
                "Some problems could occur running that step now, are you sure to continue?"
            )
            confirm = messagebox.askyesno(
                "Sequence Warning",
                msg,
                icon=messagebox.WARNING,
                parent=self.root,
            )
            if not confirm:
                return False

        # Rule 2: Critical steps warning (steps 6 & 7)
        if step_info_is_critical := step_entry.get("is_critical", False):
            msg = (
                f"Step {step_num}: {step_entry['name']}\n\n"
                "This step take very long to be completed(~4 days), are you sure to continue?"
            )
            confirm = messagebox.askyesno(
                "Critical Step Warning",
                msg,
                icon=messagebox.WARNING,
                parent=self.root,
            )
            if not confirm:
                return False

        return True

    def _set_ui_running_state(self, running: bool):
        self.is_running = running
        if running:
            self.status_pill.config(text=" RUNNING ", bg="#fef3c7", fg="#b45309")
            self.btn_run_all.config(state=tk.DISABLED)
            self.btn_stop.config(state=tk.NORMAL)
            for btn in self.step_run_buttons.values():
                btn.config(state=tk.DISABLED)
        else:
            self.status_pill.config(text="  READY  ", bg="#dcfce7", fg="#15803d")
            self.btn_run_all.config(state=tk.NORMAL)
            self.btn_stop.config(state=tk.DISABLED)
            for btn in self.step_run_buttons.values():
                btn.config(state=tk.NORMAL)
            self.current_running_step = None

    def request_stop(self):
        if self.is_running:
            self.stop_requested = True
            print("\n[MONITOR] Stop requested. Finishing current operation or halting before next step...")
            self.bottom_status_lbl.config(text="Stopping pipeline...")

    def run_single_step(self, step_num: int):
        if self.is_running:
            return

        if not self.validate_and_confirm_step(step_num):
            print(f"[MONITOR] Step {step_num} canceled by user confirmation.")
            return

        self.stop_requested = False
        self.start_time = time.time()
        self._set_ui_running_state(True)

        thread = threading.Thread(
            target=self._worker_run_single,
            args=(step_num,),
            daemon=True,
        )
        thread.start()

    def _worker_run_single(self, step_num: int):
        step_entry = next((s for s in steps if s["step"] == step_num), None)
        if not step_entry:
            self.root.after(0, self._set_ui_running_state, False)
            return

        self.current_running_step = step_num
        self.root.after(0, self.set_step_status, step_num, "running")
        self.root.after(0, self.bottom_status_lbl.config, {"text": f"Running Step {step_num}: {step_entry['name']}..."})

        print(f"\n{'='*70}")
        print(f"[STEP {step_num}] STARTING: {step_entry['name']} ({step_entry['description']})")
        print(f"{'='*70}\n")

        t0 = time.time()
        success = False
        try:
            step_entry["func"]()
            success = True
            elapsed = time.time() - t0
            print(f"\n{'='*70}")
            print(f"[STEP {step_num}] SUCCESS: Completed in {elapsed:.2f}s")
            print(f"{'='*70}\n")
            self.root.after(0, self.set_step_status, step_num, "completed")
            self.root.after(0, self.bottom_status_lbl.config, {"text": f"Step {step_num} completed in {elapsed:.2f}s."})
        except Exception as e:
            elapsed = time.time() - t0
            print(f"\n{'='*70}")
            print(f"[STEP {step_num}] ERROR: Failed after {elapsed:.2f}s -> {e}")
            print(f"{'='*70}\n")
            import traceback
            traceback.print_exc(file=sys.stderr)
            self.root.after(0, self.set_step_status, step_num, "failed")
            self.root.after(0, self.bottom_status_lbl.config, {"text": f"Step {step_num} failed: {e}"})
        finally:
            self.root.after(0, self._set_ui_running_state, False)

    def run_all_enabled(self):
        if self.is_running:
            return

        enabled_steps = [s for s in steps if self.step_vars[s["step"]].get()]
        if not enabled_steps:
            messagebox.showinfo("No Steps Enabled", "Please enable at least one step to run.")
            return

        # Check all confirmations upfront or before the respective step
        self.stop_requested = False
        self.start_time = time.time()
        self._set_ui_running_state(True)

        thread = threading.Thread(
            target=self._worker_run_all,
            args=(enabled_steps,),
            daemon=True,
        )
        thread.start()

    def _worker_run_all(self, enabled_steps: list):
        print(f"\n{'='*70}")
        print(f"[MONITOR] Starting Execution of {len(enabled_steps)} Enabled Step(s)...")
        print(f"{'='*70}\n")

        pipeline_success = True

        for step_entry in enabled_steps:
            step_num = step_entry["step"]

            if self.stop_requested:
                print("\n[MONITOR] Pipeline execution halted by user request.")
                break

            # If critical step (6 or 7) or sequence out of order, ask confirmation via main thread
            confirm_holder = {"approved": True}
            event = threading.Event()

            def _ask_confirm():
                confirm_holder["approved"] = self.validate_and_confirm_step(step_num)
                event.set()

            self.root.after(0, _ask_confirm)
            event.wait()

            if not confirm_holder["approved"]:
                print(f"\n[MONITOR] Skipping Step {step_num} ({step_entry['name']}) due to user cancellation.")
                self.root.after(0, self.set_step_status, step_num, "skipped")
                continue

            # Execute step
            self.current_running_step = step_num
            self.root.after(0, self.set_step_status, step_num, "running")
            self.root.after(0, self.bottom_status_lbl.config, {"text": f"Running Step {step_num}: {step_entry['name']}..."})

            print(f"\n{'='*70}")
            print(f"[STEP {step_num}] STARTING: {step_entry['name']} ({step_entry['description']})")
            print(f"{'='*70}\n")

            t0 = time.time()
            step_ok = False
            try:
                step_entry["func"]()
                step_ok = True
                elapsed = time.time() - t0
                print(f"\n{'='*70}")
                print(f"[STEP {step_num}] SUCCESS: Completed in {elapsed:.2f}s")
                print(f"{'='*70}\n")
                self.root.after(0, self.set_step_status, step_num, "completed")
            except Exception as e:
                pipeline_success = False
                elapsed = time.time() - t0
                print(f"\n{'='*70}")
                print(f"[STEP {step_num}] ERROR: Failed after {elapsed:.2f}s -> {e}")
                print(f"{'='*70}\n")
                import traceback
                traceback.print_exc(file=sys.stderr)
                self.root.after(0, self.set_step_status, step_num, "failed")

                # Prompt user if they wish to continue next steps after failure
                continue_event = threading.Event()
                user_choice = {"continue": False}

                def _ask_continue_on_error():
                    user_choice["continue"] = messagebox.askyesno(
                        "Step Failed",
                        f"Step {step_num} failed with error:\n{e}\n\nDo you want to continue with the next steps?",
                        icon=messagebox.ERROR,
                        parent=self.root,
                    )
                    continue_event.set()

                self.root.after(0, _ask_continue_on_error)
                continue_event.wait()

                if not user_choice["continue"]:
                    print("\n[MONITOR] Execution stopped due to step error.")
                    break

        print(f"\n{'='*70}")
        status_msg = "COMPLETED" if pipeline_success else "FINISHED WITH ISSUES"
        print(f"[MONITOR] Pipeline Batch Execution {status_msg}")
        print(f"{'='*70}\n")

        self.root.after(0, self._set_ui_running_state, False)
        self.root.after(0, self.bottom_status_lbl.config, {"text": f"Pipeline {status_msg.lower()}."})

    def _on_close(self):
        # Restore original stdout/stderr
        sys.stdout = self.orig_stdout
        sys.stderr = self.orig_stderr
        self.root.destroy()

def run_cli_mode(step_to_run: Optional[int] = None, run_all: bool = False):
    """Fallback CLI mode for headless or script-based execution."""
    print("Crosslingual-RAG Protocol Monitor (CLI Mode)")
    if step_to_run is not None:
        step_entry = next((s for s in steps if s["step"] == step_to_run), None)
        if not step_entry:
            print(f"Error: Step {step_to_run} not found.")
            return
        print(f"Executing Step {step_to_run}: {step_entry['name']}...")
        step_entry["func"]()
        print(f"Step {step_to_run} completed.")
    elif run_all:
        for s in steps:
            if s.get("enabled", True):
                print(f"Executing Step {s['step']}: {s['name']}...")
                s["func"]()
                print(f"Step {s['step']} completed.")
    else:
        print("Available steps:")
        for s in steps:
            print(f"  Step {s['step']}: {s['name']} - {s['description']}")


def main():
    parser = argparse.ArgumentParser(description="Crosslingual-RAG Protocol Monitor & Step Executor")
    parser.add_argument("--step", type=int, help="Run a specific step number (CLI mode)")
    parser.add_argument("--all", action="store_true", help="Run all enabled steps (CLI mode)")
    parser.add_argument("--cli", action="store_true", help="Run in CLI mode without opening GUI")
    args = parser.parse_args()

    if args.cli or args.step is not None or args.all:
        run_cli_mode(step_to_run=args.step, run_all=args.all)
    else:
        # Default mode: Launch the Monitor GUI
        root = tk.Tk()
        app = MonitorGUI(root)
        root.mainloop()


if __name__ == "__main__":
    main()