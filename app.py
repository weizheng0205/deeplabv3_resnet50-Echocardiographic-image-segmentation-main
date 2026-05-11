import argparse
import csv
import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

from PIL import Image, ImageTk

try:
    import cv2
except ImportError:
    cv2 = None


APP_TITLE = "EchoSeg - 超声心动图分割工作台"
DEFAULT_OUTPUT = "output/segmentation/deeplabv3_resnet50_random"
DEFAULT_DATA = "data/EchoNet-Dynamic"
DEFAULT_WEIGHTS = "output/segmentation/deeplabv3_resnet50_random/best.pt"


class EchoSegWorkbench:
    def __init__(self, root):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1360x860")
        self.root.minsize(1120, 720)

        self.process = None
        self.log_queue = queue.Queue()
        self.video_cap = None
        self.video_path = None
        self.video_after_id = None
        self.video_playing = False
        self.updating_seek = False
        self.frame_count = 0
        self.current_frame_index = 0
        self.video_photo = None
        self.video_files = []
        self.screen_video_cap = None
        self.screen_video_path = None
        self.screen_video_after_id = None
        self.screen_video_playing = False
        self.screen_video_photo = None
        self.screen_frame_count = 0
        self.screen_current_frame_index = 0
        self.screen_updating_seek = False

        self.weights_var = tk.StringVar(value=self._default_path(DEFAULT_WEIGHTS))
        self.data_var = tk.StringVar(value=self._default_path(DEFAULT_DATA))
        self.output_var = tk.StringVar(value=self._default_path(DEFAULT_OUTPUT))
        self.batch_var = tk.StringVar(value="4")
        self.workers_var = tk.StringVar(value="0")
        self.device_var = tk.StringVar(value="cuda")
        self.save_video_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="就绪")
        self.video_status_var = tk.StringVar(value="未加载视频")
        self.screen_video_status_var = tk.StringVar(value="选择 EF 异常样本后显示 AVI")
        self.screen_filter_var = tk.StringVar(value="全部")
        self.screen_summary_cards = {}
        self.screening_rows = []

        self._configure_style()
        self._build_layout()
        self.root.after(120, self._drain_log_queue)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _default_path(self, value):
        path = Path(value)
        return str(path) if path.exists() else value

    def _configure_style(self):
        self.root.configure(bg="#f5f7fb")
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(".", font=("Microsoft YaHei UI", 10), background="#f5f7fb")
        style.configure("TFrame", background="#f5f7fb")
        style.configure("Panel.TFrame", background="#ffffff", relief="flat")
        style.configure("Muted.TLabel", background="#ffffff", foreground="#64748b")
        style.configure("Title.TLabel", background="#ffffff", foreground="#0f172a", font=("Microsoft YaHei UI", 16, "bold"))
        style.configure("CardTitle.TLabel", background="#ffffff", foreground="#475569", font=("Microsoft YaHei UI", 9))
        style.configure("CardValue.TLabel", background="#ffffff", foreground="#0f172a", font=("Microsoft YaHei UI", 18, "bold"))
        style.configure("TLabel", background="#f5f7fb", foreground="#1e293b")
        style.configure("TButton", padding=(12, 7))
        style.configure("Primary.TButton", padding=(14, 8), foreground="#ffffff", background="#2563eb")
        style.map("Primary.TButton", background=[("active", "#1d4ed8"), ("disabled", "#94a3b8")])
        style.configure("Danger.TButton", padding=(14, 8), foreground="#ffffff", background="#dc2626")
        style.map("Danger.TButton", background=[("active", "#b91c1c"), ("disabled", "#fca5a5")])
        style.configure("Treeview", rowheight=28, background="#ffffff", fieldbackground="#ffffff")
        style.configure("Treeview.Heading", font=("Microsoft YaHei UI", 10, "bold"))

    def _build_layout(self):
        shell = ttk.Frame(self.root, padding=14)
        shell.pack(fill=tk.BOTH, expand=True)
        shell.columnconfigure(0, minsize=360)
        shell.columnconfigure(1, weight=1)
        shell.rowconfigure(0, weight=1)

        self.sidebar = ttk.Frame(shell, style="Panel.TFrame", padding=18)
        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        self.sidebar.columnconfigure(0, weight=1)

        main = ttk.Frame(shell)
        main.grid(row=0, column=1, sticky="nsew")
        main.rowconfigure(1, weight=1)
        main.columnconfigure(0, weight=1)

        self._build_sidebar()
        self._build_header(main)
        self._build_tabs(main)

    def _build_sidebar(self):
        ttk.Label(self.sidebar, text="EchoSeg", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(self.sidebar, text="DeepLabV3 ResNet50 左心室分割", style="Muted.TLabel").grid(row=1, column=0, sticky="w", pady=(2, 18))

        self._path_picker(self.sidebar, 2, "权重文件", self.weights_var, self.browse_weights, "选择 .pt 权重")
        self._path_picker(self.sidebar, 4, "数据集目录", self.data_var, self.browse_data, "选择 EchoNet-Dynamic")
        self._path_picker(self.sidebar, 6, "输出目录", self.output_var, self.browse_output, "选择结果目录")

        params = ttk.Frame(self.sidebar, style="Panel.TFrame")
        params.grid(row=8, column=0, sticky="ew", pady=(12, 6))
        params.columnconfigure(0, weight=1)
        params.columnconfigure(1, weight=1)
        ttk.Label(params, text="批大小", style="Muted.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Entry(params, textvariable=self.batch_var, width=10).grid(row=1, column=0, sticky="ew", padx=(0, 8), pady=(4, 10))
        ttk.Label(params, text="加载线程", style="Muted.TLabel").grid(row=0, column=1, sticky="w")
        ttk.Entry(params, textvariable=self.workers_var, width=10).grid(row=1, column=1, sticky="ew", pady=(4, 10))
        ttk.Label(params, text="设备", style="Muted.TLabel").grid(row=2, column=0, sticky="w")
        ttk.Combobox(params, textvariable=self.device_var, values=("cuda", "cpu"), state="readonly").grid(row=3, column=0, sticky="ew", padx=(0, 8), pady=(4, 10))
        ttk.Checkbutton(params, text="生成分割视频", variable=self.save_video_var).grid(row=3, column=1, sticky="w", pady=(4, 10))

        actions = ttk.Frame(self.sidebar, style="Panel.TFrame")
        actions.grid(row=9, column=0, sticky="ew", pady=(8, 12))
        actions.columnconfigure(0, weight=1)
        actions.columnconfigure(1, weight=1)
        self.run_button = ttk.Button(actions, text="开始推理", style="Primary.TButton", command=self.start_segmentation)
        self.run_button.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.stop_button = ttk.Button(actions, text="停止", style="Danger.TButton", command=self.stop_segmentation, state=tk.DISABLED)
        self.stop_button.grid(row=0, column=1, sticky="ew")
        ttk.Button(self.sidebar, text="加载已有结果", command=self.load_results).grid(row=10, column=0, sticky="ew", pady=(0, 8))
        ttk.Button(self.sidebar, text="清空日志", command=self.clear_log).grid(row=11, column=0, sticky="ew")

        ttk.Separator(self.sidebar).grid(row=12, column=0, sticky="ew", pady=18)
        ttk.Label(self.sidebar, text="运行日志", style="Muted.TLabel").grid(row=13, column=0, sticky="w")
        self.log_text = scrolledtext.ScrolledText(self.sidebar, height=18, borderwidth=0, relief=tk.FLAT, wrap=tk.WORD)
        self.log_text.grid(row=14, column=0, sticky="nsew", pady=(6, 0))
        self.sidebar.rowconfigure(14, weight=1)

    def _path_picker(self, parent, row, label, variable, command, title):
        ttk.Label(parent, text=label, style="Muted.TLabel").grid(row=row, column=0, sticky="w")
        line = ttk.Frame(parent, style="Panel.TFrame")
        line.grid(row=row + 1, column=0, sticky="ew", pady=(4, 12))
        line.columnconfigure(0, weight=1)
        ttk.Entry(line, textvariable=variable).grid(row=0, column=0, sticky="ew", padx=(0, 8))
        ttk.Button(line, text="浏览", command=command).grid(row=0, column=1)

    def _build_header(self, parent):
        header = ttk.Frame(parent, style="Panel.TFrame", padding=(18, 14))
        header.grid(row=0, column=0, sticky="ew", pady=(0, 14))
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="分割结果分析", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(header, textvariable=self.status_var, style="Muted.TLabel").grid(row=1, column=0, sticky="w", pady=(2, 0))
        self.progress = ttk.Progressbar(header, mode="indeterminate", length=180)
        self.progress.grid(row=0, column=1, rowspan=2, sticky="e")

    def _build_tabs(self, parent):
        self.tabs = ttk.Notebook(parent)
        self.tabs.grid(row=1, column=0, sticky="nsew")

        self.overview_tab = ttk.Frame(self.tabs, padding=14)
        self.video_tab = ttk.Frame(self.tabs, padding=14)
        self.metrics_tab = ttk.Frame(self.tabs, padding=14)
        self.screen_tab = ttk.Frame(self.tabs, padding=14)
        self.curves_tab = ttk.Frame(self.tabs, padding=14)

        self.tabs.add(self.overview_tab, text="总览")
        self.tabs.add(self.video_tab, text="视频预览")
        self.tabs.add(self.metrics_tab, text="指标明细")
        self.tabs.add(self.screen_tab, text="心功能筛查")
        self.tabs.add(self.curves_tab, text="训练曲线")

        self._build_overview()
        self._build_video_tab()
        self._build_metrics_tab()
        self._build_screen_tab()
        self._build_curves_tab()

    def _build_overview(self):
        self.overview_tab.columnconfigure((0, 1, 2, 3), weight=1)
        self.metric_cards = {}
        for col, key, title in [
            (0, "videos", "视频结果"),
            (1, "overall", "Overall Dice"),
            (2, "large", "Large Dice"),
            (3, "small", "Small Dice"),
        ]:
            card = ttk.Frame(self.overview_tab, style="Panel.TFrame", padding=16)
            card.grid(row=0, column=col, sticky="ew", padx=(0 if col == 0 else 8, 0), pady=(0, 12))
            ttk.Label(card, text=title, style="CardTitle.TLabel").pack(anchor="w")
            value = ttk.Label(card, text="-", style="CardValue.TLabel")
            value.pack(anchor="w", pady=(6, 0))
            self.metric_cards[key] = value

        summary_frame = ttk.Frame(self.overview_tab, style="Panel.TFrame", padding=16)
        summary_frame.grid(row=1, column=0, columnspan=4, sticky="nsew")
        self.overview_tab.rowconfigure(1, weight=1)
        summary_frame.rowconfigure(0, weight=1)
        summary_frame.columnconfigure(0, weight=1)
        self.summary_text = scrolledtext.ScrolledText(summary_frame, borderwidth=0, relief=tk.FLAT, wrap=tk.WORD)
        self.summary_text.grid(row=0, column=0, sticky="nsew")
        self.summary_text.insert(tk.END, "点击“加载已有结果”查看当前输出目录中的指标、视频和训练日志。\n")
        self.summary_text.configure(state=tk.DISABLED)

    def _build_video_tab(self):
        self.video_tab.columnconfigure(1, weight=1)
        self.video_tab.rowconfigure(0, weight=1)

        list_panel = ttk.Frame(self.video_tab, style="Panel.TFrame", padding=10)
        list_panel.grid(row=0, column=0, sticky="ns", padx=(0, 12))
        ttk.Label(list_panel, text="视频文件", style="Muted.TLabel").pack(anchor="w")
        self.video_list = tk.Listbox(list_panel, width=34, borderwidth=0, activestyle="none")
        self.video_list.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        self.video_list.bind("<<ListboxSelect>>", self.on_video_selected)

        player = ttk.Frame(self.video_tab, style="Panel.TFrame", padding=12)
        player.grid(row=0, column=1, sticky="nsew")
        player.rowconfigure(0, weight=1)
        player.columnconfigure(0, weight=1)
        self.video_canvas = tk.Label(player, bg="#020617")
        self.video_canvas.grid(row=0, column=0, sticky="nsew")

        controls = ttk.Frame(player, style="Panel.TFrame")
        controls.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        controls.columnconfigure(2, weight=1)
        ttk.Button(controls, text="播放", command=self.play_video).grid(row=0, column=0, padx=(0, 8))
        ttk.Button(controls, text="暂停", command=self.pause_video).grid(row=0, column=1, padx=(0, 8))
        self.seek_var = tk.DoubleVar(value=0)
        self.seek = ttk.Scale(controls, variable=self.seek_var, from_=0, to=1, command=self.on_seek)
        self.seek.grid(row=0, column=2, sticky="ew", padx=(0, 8))
        ttk.Label(controls, textvariable=self.video_status_var, style="Muted.TLabel").grid(row=0, column=3, sticky="e")

    def _build_metrics_tab(self):
        self.metrics_tab.rowconfigure(0, weight=1)
        self.metrics_tab.columnconfigure(0, weight=1)
        columns = ("filename", "overall", "large", "small")
        self.metrics_table = ttk.Treeview(self.metrics_tab, columns=columns, show="headings")
        for col, title, width in [
            ("filename", "文件名", 300),
            ("overall", "Overall", 120),
            ("large", "Large", 120),
            ("small", "Small", 120),
        ]:
            self.metrics_table.heading(col, text=title)
            self.metrics_table.column(col, width=width, anchor=tk.CENTER if col != "filename" else tk.W)
        scroll = ttk.Scrollbar(self.metrics_tab, orient=tk.VERTICAL, command=self.metrics_table.yview)
        self.metrics_table.configure(yscrollcommand=scroll.set)
        self.metrics_table.grid(row=0, column=0, sticky="nsew")
        scroll.grid(row=0, column=1, sticky="ns")

    def _build_screen_tab(self):
        self.screen_tab.columnconfigure((0, 1, 2), weight=2)
        self.screen_tab.columnconfigure(3, weight=1)
        self.screen_tab.rowconfigure(2, weight=1)
        for col, key, title in [
            (0, "total", "筛查样本"),
            (1, "abnormal", "EF < 50%"),
            (2, "severe", "EF < 30%"),
            (3, "avg_ef", "平均 EF"),
        ]:
            card = ttk.Frame(self.screen_tab, style="Panel.TFrame", padding=16)
            card.grid(row=0, column=col, sticky="ew", padx=(0 if col == 0 else 8, 0), pady=(0, 12))
            ttk.Label(card, text=title, style="CardTitle.TLabel").pack(anchor="w")
            value = ttk.Label(card, text="-", style="CardValue.TLabel")
            value.pack(anchor="w", pady=(6, 0))
            self.screen_summary_cards[key] = value

        toolbar = ttk.Frame(self.screen_tab, style="Panel.TFrame", padding=(12, 10))
        toolbar.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(0, 12))
        ttk.Label(toolbar, text="风险筛选", style="Muted.TLabel").pack(side=tk.LEFT, padx=(0, 8))
        filter_box = ttk.Combobox(
            toolbar,
            textvariable=self.screen_filter_var,
            values=("全部", "异常风险", "重度降低", "正常范围"),
            state="readonly",
            width=12,
        )
        filter_box.pack(side=tk.LEFT)
        filter_box.bind("<<ComboboxSelected>>", lambda _event: self._render_screening_rows())
        ttk.Label(toolbar, text="EF 阈值：低于 50% 作为收缩功能异常风险提示，结果仅供科研筛查。", style="Muted.TLabel").pack(side=tk.RIGHT)

        columns = ("filename", "ef", "edv", "esv", "split", "risk")
        table_panel = ttk.Frame(self.screen_tab, style="Panel.TFrame", padding=10)
        table_panel.grid(row=2, column=0, columnspan=3, sticky="nsew", padx=(0, 12))
        table_panel.rowconfigure(0, weight=1)
        table_panel.columnconfigure(0, weight=1)

        self.screen_table = ttk.Treeview(table_panel, columns=columns, show="headings")
        for col, title, width in [
            ("filename", "文件名", 280),
            ("ef", "EF (%)", 90),
            ("edv", "EDV", 90),
            ("esv", "ESV", 90),
            ("split", "划分", 90),
            ("risk", "筛查结果", 180),
        ]:
            self.screen_table.heading(col, text=title)
            self.screen_table.column(col, width=width, anchor=tk.CENTER if col != "filename" else tk.W)
        scroll = ttk.Scrollbar(table_panel, orient=tk.VERTICAL, command=self.screen_table.yview)
        self.screen_table.configure(yscrollcommand=scroll.set)
        self.screen_table.tag_configure("abnormal", background="#fff7ed")
        self.screen_table.tag_configure("severe", background="#fee2e2")
        self.screen_table.grid(row=0, column=0, sticky="nsew")
        scroll.grid(row=0, column=1, sticky="ns")
        self.screen_table.bind("<<TreeviewSelect>>", self.on_screening_selected)

        preview = ttk.Frame(self.screen_tab, style="Panel.TFrame", padding=12)
        preview.grid(row=2, column=3, sticky="nsew")
        preview.rowconfigure(1, weight=1)
        preview.columnconfigure(0, weight=1)
        ttk.Label(preview, text="样本 AVI", style="CardTitle.TLabel").grid(row=0, column=0, sticky="w")
        self.screen_video_canvas = tk.Label(preview, bg="#020617")
        self.screen_video_canvas.grid(row=1, column=0, sticky="nsew", pady=(8, 10))
        controls = ttk.Frame(preview, style="Panel.TFrame")
        controls.grid(row=2, column=0, sticky="ew")
        controls.columnconfigure(2, weight=1)
        ttk.Button(controls, text="播放", command=self.play_screen_video).grid(row=0, column=0, padx=(0, 8))
        ttk.Button(controls, text="暂停", command=self.pause_screen_video).grid(row=0, column=1, padx=(0, 8))
        self.screen_seek_var = tk.DoubleVar(value=0)
        self.screen_seek = ttk.Scale(controls, variable=self.screen_seek_var, from_=0, to=1, command=self.on_screen_seek)
        self.screen_seek.grid(row=0, column=2, sticky="ew")
        ttk.Label(preview, textvariable=self.screen_video_status_var, style="Muted.TLabel", wraplength=280).grid(
            row=3, column=0, sticky="ew", pady=(8, 0)
        )

    def _build_curves_tab(self):
        self.curves_tab.rowconfigure(0, weight=1)
        self.curves_tab.columnconfigure(0, weight=1)
        self.curves_holder = ttk.Frame(self.curves_tab, style="Panel.TFrame", padding=12)
        self.curves_holder.grid(row=0, column=0, sticky="nsew")

    def browse_weights(self):
        path = filedialog.askopenfilename(title="选择模型权重", filetypes=[("PyTorch 权重", "*.pt"), ("所有文件", "*.*")])
        if path:
            self.weights_var.set(path)

    def browse_data(self):
        path = filedialog.askdirectory(title="选择 EchoNet-Dynamic 数据集目录")
        if path:
            self.data_var.set(path)

    def browse_output(self):
        path = filedialog.askdirectory(title="选择输出目录")
        if path:
            self.output_var.set(path)

    def start_segmentation(self):
        if self.process and self.process.poll() is None:
            messagebox.showinfo("任务运行中", "当前已有一个推理任务正在运行。")
            return
        if not Path(self.weights_var.get()).exists():
            messagebox.showerror("缺少权重", "请选择有效的 .pt 权重文件。")
            return
        if not Path(self.data_var.get()).exists():
            messagebox.showerror("缺少数据集", "请选择有效的数据集目录。")
            return
        try:
            int(self.batch_var.get())
            int(self.workers_var.get())
        except ValueError:
            messagebox.showerror("参数错误", "批大小和加载线程必须是整数。")
            return

        Path(self.output_var.get()).mkdir(parents=True, exist_ok=True)
        args = [
            "--data_dir", self.data_var.get(),
            "--output", self.output_var.get(),
            "--weights", self.weights_var.get(),
            "--batch_size", self.batch_var.get(),
            "--num_workers", self.workers_var.get(),
            "--device", self.device_var.get(),
            "--run_test",
        ]
        args.append("--save_video" if self.save_video_var.get() else "--skip_video")
        cmd = self._worker_command(args)

        self.log("启动命令: " + " ".join(cmd))
        self.status_var.set("正在运行分割任务")
        self.progress.start(12)
        self.run_button.configure(state=tk.DISABLED)
        self.stop_button.configure(state=tk.NORMAL)

        thread = threading.Thread(target=self._run_process, args=(cmd,), daemon=True)
        thread.start()

    def _worker_command(self, args):
        if getattr(sys, "frozen", False):
            return [sys.executable, "--worker", *args]
        return [sys.executable, str(Path(__file__).resolve()), "--worker", *args]

    def _run_process(self, cmd):
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
                creationflags=creationflags,
            )
            assert self.process.stdout is not None
            for line in self.process.stdout:
                self.log_queue.put(line.rstrip())
            code = self.process.wait()
            self.log_queue.put(f"__PROCESS_DONE__:{code}")
        except Exception as exc:
            self.log_queue.put(f"运行失败: {exc}")
            self.log_queue.put("__PROCESS_DONE__:-1")

    def stop_segmentation(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            self.log("已发送停止信号。")

    def _drain_log_queue(self):
        try:
            while True:
                line = self.log_queue.get_nowait()
                if line.startswith("__PROCESS_DONE__:"):
                    code = int(line.split(":", 1)[1])
                    self._process_finished(code)
                else:
                    self.log(line)
        except queue.Empty:
            pass
        self.root.after(120, self._drain_log_queue)

    def _process_finished(self, code):
        self.progress.stop()
        self.run_button.configure(state=tk.NORMAL)
        self.stop_button.configure(state=tk.DISABLED)
        if code == 0:
            self.status_var.set("分割任务完成")
            self.log("任务完成，正在刷新结果。")
            self.load_results()
        else:
            self.status_var.set(f"任务结束，返回码 {code}")

    def load_results(self):
        output = Path(self.output_var.get())
        if not output.exists():
            messagebox.showerror("目录不存在", "输出目录不存在。")
            return
        self.status_var.set("正在加载结果")
        self._load_video_files(output)
        metrics = self._load_metrics(output)
        self._load_screening(output)
        self._load_summary(output, metrics)
        self._load_curves(output)
        self.status_var.set("结果已加载")
        self.log(f"已加载结果目录: {output}")

    def _load_video_files(self, output):
        self.video_files = sorted((output / "videos").glob("*.avi")) if (output / "videos").exists() else []
        self.video_list.delete(0, tk.END)
        for item in self.video_files:
            self.video_list.insert(tk.END, item.name)
        self.metric_cards["videos"].configure(text=str(len(self.video_files)))
        if self.video_files:
            self.video_list.selection_set(0)
            if cv2 is None:
                self.video_status_var.set("未安装 opencv-python，无法预览视频")
            else:
                self.open_video(self.video_files[0])
        else:
            self.close_video()
            self.video_status_var.set("未找到视频结果")

    def _load_metrics(self, output):
        for row in self.metrics_table.get_children():
            self.metrics_table.delete(row)
        metrics_file = output / "test_dice.csv"
        if not metrics_file.exists():
            metrics_file = output / "val_dice.csv"
        rows = []
        if metrics_file.exists():
            with metrics_file.open("r", encoding="utf-8", errors="replace", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    cleaned = {key.strip(): value.strip() for key, value in row.items() if key is not None}
                    rows.append(cleaned)
                    self.metrics_table.insert(
                        "",
                        tk.END,
                        values=(
                            cleaned.get("Filename", ""),
                            self._format_float(cleaned.get("Overall", "")),
                            self._format_float(cleaned.get("Large", "")),
                            self._format_float(cleaned.get("Small", "")),
                        ),
                    )
        return rows

    def _load_screening(self, output):
        data_dir = Path(self.data_var.get())
        file_list = data_dir / "FileList.csv"
        if not file_list.exists():
            self.screening_rows = []
            self._render_screening_rows()
            self.log("未找到 FileList.csv，无法生成心功能筛查表。")
            return

        result_names = {path.name for path in self.video_files}
        if not result_names:
            for name in ("test_dice.csv", "val_dice.csv"):
                metrics_path = output / name
                if metrics_path.exists():
                    with metrics_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
                        result_names = {
                            row.get("Filename", "").strip()
                            for row in csv.DictReader(handle)
                            if row.get("Filename", "").strip()
                        }
                    break

        rows = []
        with file_list.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            for row in csv.DictReader(handle):
                filename = row.get("FileName", "").strip()
                if not filename:
                    continue
                video_name = filename if filename.lower().endswith(".avi") else f"{filename}.avi"
                if result_names and video_name not in result_names:
                    continue
                try:
                    ef = float(row.get("EF", ""))
                    edv = float(row.get("EDV", ""))
                    esv = float(row.get("ESV", ""))
                except ValueError:
                    continue
                rows.append(
                    {
                        "filename": video_name,
                        "ef": ef,
                        "edv": edv,
                        "esv": esv,
                        "split": row.get("Split", "").strip(),
                        "risk": self._risk_label(ef),
                    }
                )

        self.screening_rows = sorted(rows, key=lambda item: item["ef"])
        self._render_screening_rows()

    def _risk_label(self, ef):
        if ef < 30:
            return "重度降低"
        if ef < 40:
            return "中度降低"
        if ef < 50:
            return "轻度降低"
        return "正常范围"

    def _render_screening_rows(self):
        for row_id in self.screen_table.get_children():
            self.screen_table.delete(row_id)

        rows = self.screening_rows
        selected = self.screen_filter_var.get()
        if selected == "异常风险":
            rows = [row for row in rows if row["ef"] < 50]
        elif selected == "重度降低":
            rows = [row for row in rows if row["ef"] < 30]
        elif selected == "正常范围":
            rows = [row for row in rows if row["ef"] >= 50]

        for row in rows:
            tags = ()
            if row["ef"] < 30:
                tags = ("severe",)
            elif row["ef"] < 50:
                tags = ("abnormal",)
            self.screen_table.insert(
                "",
                tk.END,
                values=(
                    row["filename"],
                    f"{row['ef']:.2f}",
                    f"{row['edv']:.2f}",
                    f"{row['esv']:.2f}",
                    row["split"],
                    row["risk"],
                ),
                tags=tags,
            )

        total = len(self.screening_rows)
        abnormal = sum(1 for row in self.screening_rows if row["ef"] < 50)
        severe = sum(1 for row in self.screening_rows if row["ef"] < 30)
        avg_ef = sum(row["ef"] for row in self.screening_rows) / total if total else None
        self.screen_summary_cards["total"].configure(text=str(total))
        self.screen_summary_cards["abnormal"].configure(text=str(abnormal))
        self.screen_summary_cards["severe"].configure(text=str(severe))
        self.screen_summary_cards["avg_ef"].configure(text="-" if avg_ef is None else f"{avg_ef:.2f}%")

        abnormal_items = [
            item_id
            for item_id in self.screen_table.get_children()
            if float(self.screen_table.item(item_id, "values")[1]) < 50
        ]
        if abnormal_items:
            self.screen_table.selection_set(abnormal_items[0])
            self.screen_table.focus(abnormal_items[0])
            self.screen_table.see(abnormal_items[0])
            self.on_screening_selected()
        else:
            self.close_screen_video()
            first_item = self.screen_table.get_children()
            if first_item:
                self.screen_table.selection_set(first_item[0])
                self.screen_table.focus(first_item[0])
                self.screen_table.see(first_item[0])
                self.on_screening_selected()
            else:
                self.close_screen_video()
                self.screen_video_status_var.set("当前筛选结果中没有样本")

    def on_screening_selected(self, _event=None):
        selection = self.screen_table.selection()
        if not selection:
            return
        values = self.screen_table.item(selection[0], "values")
        if not values:
            return
        filename = values[0]
        try:
            ef = float(values[1])
        except ValueError:
            ef = 100.0
        video_path = Path(self.output_var.get()) / "videos" / filename
        if not video_path.exists():
            self.close_screen_video()
            self.screen_video_status_var.set(f"未找到分割视频: {video_path}")
            return
        self.open_screen_video(video_path, values[5] if len(values) > 5 else self._risk_label(ef))

    def open_screen_video(self, path, risk_label=""):
        if cv2 is None:
            self.screen_video_status_var.set("未安装 opencv-python，无法预览 AVI")
            return
        self.close_screen_video(keep_canvas=True)
        self.screen_video_path = Path(path)
        self.screen_video_cap = cv2.VideoCapture(str(path))
        if not self.screen_video_cap.isOpened():
            self.screen_video_status_var.set("无法打开该 AVI")
            return
        self.screen_frame_count = int(self.screen_video_cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        self.screen_seek.configure(to=max(self.screen_frame_count - 1, 1))
        self.screen_current_frame_index = 0
        self.screen_video_risk_label = risk_label
        self.screen_video_status_var.set(f"{self.screen_video_path.name}  {risk_label}")
        self.show_screen_frame(0)

    def show_screen_frame(self, index):
        if not self.screen_video_cap:
            return
        index = max(0, min(int(index), self.screen_frame_count - 1))
        self.screen_video_cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self.screen_video_cap.read()
        if not ok:
            return
        self.screen_current_frame_index = index
        self.screen_updating_seek = True
        self.screen_seek_var.set(index)
        self.screen_updating_seek = False
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        width = max(self.screen_video_canvas.winfo_width(), 300)
        height = max(self.screen_video_canvas.winfo_height(), 260)
        image.thumbnail((width, height), Image.Resampling.LANCZOS)
        self.screen_video_photo = ImageTk.PhotoImage(image)
        self.screen_video_canvas.configure(image=self.screen_video_photo)
        risk_label = getattr(self, "screen_video_risk_label", "")
        suffix = f"  {risk_label}" if risk_label else ""
        self.screen_video_status_var.set(f"{self.screen_video_path.name}  {index + 1}/{self.screen_frame_count}{suffix}")

    def play_screen_video(self):
        if not self.screen_video_cap:
            return
        self.screen_video_playing = True
        self._schedule_next_screen_frame()

    def pause_screen_video(self):
        self.screen_video_playing = False
        if self.screen_video_after_id:
            self.root.after_cancel(self.screen_video_after_id)
            self.screen_video_after_id = None

    def _schedule_next_screen_frame(self):
        if not self.screen_video_playing or not self.screen_video_cap:
            return
        next_index = self.screen_current_frame_index + 1
        if next_index >= self.screen_frame_count:
            self.screen_video_playing = False
            return
        self.show_screen_frame(next_index)
        fps = self.screen_video_cap.get(cv2.CAP_PROP_FPS) or 30
        delay = max(10, int(1000 / fps))
        self.screen_video_after_id = self.root.after(delay, self._schedule_next_screen_frame)

    def on_screen_seek(self, value):
        if self.screen_video_playing or self.screen_updating_seek:
            return
        self.show_screen_frame(float(value))

    def close_screen_video(self, keep_canvas=False):
        self.pause_screen_video()
        if self.screen_video_cap:
            self.screen_video_cap.release()
        self.screen_video_cap = None
        self.screen_video_path = None
        self.screen_frame_count = 0
        self.screen_current_frame_index = 0
        if not keep_canvas and hasattr(self, "screen_video_canvas"):
            self.screen_video_canvas.configure(image="")

    def _load_summary(self, output, metrics):
        averages = self._metric_averages(metrics)
        self.metric_cards["overall"].configure(text=self._format_float(averages.get("Overall")))
        self.metric_cards["large"].configure(text=self._format_float(averages.get("Large")))
        self.metric_cards["small"].configure(text=self._format_float(averages.get("Small")))

        log_file = output / "log.csv"
        log_tail = ""
        if log_file.exists():
            lines = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
            log_tail = "\n".join(lines[-18:])

        self.summary_text.configure(state=tk.NORMAL)
        self.summary_text.delete("1.0", tk.END)
        self.summary_text.insert(tk.END, f"输出目录: {output}\n")
        self.summary_text.insert(tk.END, f"指标文件记录数: {len(metrics)}\n")
        self.summary_text.insert(tk.END, f"分割视频数: {len(self.video_files)}\n\n")
        if averages:
            self.summary_text.insert(
                tk.END,
                "平均 Dice: Overall {Overall:.4f}, Large {Large:.4f}, Small {Small:.4f}\n\n".format(**averages),
            )
        if log_tail:
            self.summary_text.insert(tk.END, "最近日志:\n" + log_tail)
        self.summary_text.configure(state=tk.DISABLED)

    def _load_curves(self, output):
        for child in self.curves_holder.winfo_children():
            child.destroy()
        log_file = output / "log.csv"
        points = self._parse_training_log(log_file)
        if not points:
            ttk.Label(self.curves_holder, text="没有找到可绘制的训练曲线数据。", style="Muted.TLabel").pack(anchor="w")
            return
        try:
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        except Exception as exc:
            ttk.Label(self.curves_holder, text=f"无法加载 matplotlib: {exc}", style="Muted.TLabel").pack(anchor="w")
            return

        train = [p for p in points if p["phase"] == "train"]
        val = [p for p in points if p["phase"] == "val"]
        fig, (ax_loss, ax_dice) = plt.subplots(2, 1, figsize=(8, 6), dpi=100)
        if train:
            ax_loss.plot([p["epoch"] for p in train], [p["loss"] for p in train], label="Train loss", color="#2563eb")
        if val:
            ax_loss.plot([p["epoch"] for p in val], [p["loss"] for p in val], label="Val loss", color="#dc2626")
            ax_dice.plot([p["epoch"] for p in val], [p["overall"] for p in val], label="Val Dice", color="#16a34a")
        ax_loss.set_title("Loss")
        ax_loss.grid(alpha=0.25)
        ax_loss.legend()
        ax_dice.set_title("Validation Dice")
        ax_dice.grid(alpha=0.25)
        ax_dice.legend()
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.curves_holder)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _parse_training_log(self, log_file):
        if not log_file.exists():
            return []
        points = []
        with log_file.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                parts = [part.strip() for part in line.split(",")]
                if len(parts) < 6:
                    continue
                try:
                    points.append(
                        {
                            "epoch": int(parts[0]),
                            "phase": parts[1],
                            "loss": float(parts[2]),
                            "overall": float(parts[3]),
                            "large": float(parts[4]),
                            "small": float(parts[5]),
                        }
                    )
                except ValueError:
                    continue
        return points

    def _metric_averages(self, rows):
        totals = {"Overall": 0.0, "Large": 0.0, "Small": 0.0}
        count = 0
        for row in rows:
            try:
                totals["Overall"] += float(row["Overall"])
                totals["Large"] += float(row["Large"])
                totals["Small"] += float(row["Small"])
                count += 1
            except (KeyError, ValueError):
                continue
        if not count:
            return {}
        return {key: value / count for key, value in totals.items()}

    def _format_float(self, value):
        if value in (None, ""):
            return "-"
        try:
            return f"{float(value):.4f}"
        except (TypeError, ValueError):
            return str(value)

    def on_video_selected(self, _event=None):
        selection = self.video_list.curselection()
        if selection:
            self.open_video(self.video_files[selection[0]])

    def open_video(self, path):
        if cv2 is None:
            self.video_status_var.set("未安装 opencv-python，无法预览视频")
            messagebox.showerror("缺少依赖", "当前 Python 环境未安装 opencv-python，无法预览 AVI 视频。")
            return
        self.close_video(keep_canvas=True)
        self.video_path = Path(path)
        self.video_cap = cv2.VideoCapture(str(path))
        if not self.video_cap.isOpened():
            self.video_status_var.set("无法打开视频")
            return
        self.frame_count = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        self.seek.configure(to=max(self.frame_count - 1, 1))
        self.current_frame_index = 0
        self.video_status_var.set(self.video_path.name)
        self.show_frame(0)

    def show_frame(self, index):
        if not self.video_cap:
            return
        index = max(0, min(int(index), self.frame_count - 1))
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self.video_cap.read()
        if not ok:
            return
        self.current_frame_index = index
        self.updating_seek = True
        self.seek_var.set(index)
        self.updating_seek = False
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        width = max(self.video_canvas.winfo_width(), 640)
        height = max(self.video_canvas.winfo_height(), 360)
        image.thumbnail((width, height), Image.Resampling.LANCZOS)
        self.video_photo = ImageTk.PhotoImage(image)
        self.video_canvas.configure(image=self.video_photo)
        self.video_status_var.set(f"{self.video_path.name}  {index + 1}/{self.frame_count}")

    def play_video(self):
        if not self.video_cap:
            return
        self.video_playing = True
        self._schedule_next_frame()

    def pause_video(self):
        self.video_playing = False
        if self.video_after_id:
            self.root.after_cancel(self.video_after_id)
            self.video_after_id = None

    def _schedule_next_frame(self):
        if not self.video_playing or not self.video_cap:
            return
        next_index = self.current_frame_index + 1
        if next_index >= self.frame_count:
            self.video_playing = False
            return
        self.show_frame(next_index)
        fps = self.video_cap.get(cv2.CAP_PROP_FPS) or 30
        delay = max(10, int(1000 / fps))
        self.video_after_id = self.root.after(delay, self._schedule_next_frame)

    def on_seek(self, value):
        if self.video_playing or self.updating_seek:
            return
        self.show_frame(float(value))

    def close_video(self, keep_canvas=False):
        self.pause_video()
        if self.video_cap:
            self.video_cap.release()
        self.video_cap = None
        self.video_path = None
        self.frame_count = 0
        self.current_frame_index = 0
        if not keep_canvas:
            self.video_canvas.configure(image="")

    def log(self, message):
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)

    def clear_log(self):
        self.log_text.delete("1.0", tk.END)

    def on_close(self):
        self.stop_segmentation()
        self.close_video()
        self.close_screen_video()
        self.root.destroy()


def run_worker(argv):
    from main import run as segmentation_command

    segmentation_command.main(args=argv, prog_name="segmentation", standalone_mode=True)


def parse_args(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--worker", action="store_true")
    args, rest = parser.parse_known_args(argv)
    return args, rest


def main(argv=None):
    args, rest = parse_args(sys.argv[1:] if argv is None else argv)
    if args.worker:
        run_worker(rest)
        return
    root = tk.Tk()
    EchoSegWorkbench(root)
    root.mainloop()


if __name__ == "__main__":
    main()
