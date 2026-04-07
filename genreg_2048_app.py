#!/usr/bin/env python3
# ================================================================
# GENREG 2048 — Unified Training + Inference App
# ================================================================
# Pygame window: renders the 2048 board
# Tkinter panel: tabs for Training, Inference, and Config
#
# Usage:
#   python genreg_2048_app.py             # full GUI
#   python genreg_2048_app.py --headless  # no pygame, fast training
# ================================================================

import sys
import os
import copy
import json
import math
import random
import time
import threading
import argparse
from pathlib import Path

HEADLESS = "--headless" in sys.argv or "--headless_v2" in sys.argv or "--headless_v3" in sys.argv or "--headless_v4" in sys.argv or "--headless_v5" in sys.argv
HEADLESS_V2 = "--headless_v2" in sys.argv
HEADLESS_V3 = "--headless_v3" in sys.argv or "--headless_v4" in sys.argv or "--headless_v5" in sys.argv  # V4/V5 extend V3
HEADLESS_V4 = "--headless_v4" in sys.argv or "--headless_v5" in sys.argv  # V5 extends V4
HEADLESS_V5 = "--headless_v5" in sys.argv

if not HEADLESS:
    import pygame

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

from genreg_2048_env import Game2048Env
from genreg_controller import GENREGController
from genreg_genome import GENREGPopulation, GENREGGenome
from genreg_proteins import (
    SensorProtein,
    TrendProtein,
    ComparatorProtein,
    IntegratorProtein,
    GateProtein,
    TrustModifierProtein,
    run_protein_cascade,
)
if HEADLESS_V2:
    import torch
    from genreg_gpu import GPUEvolver

if HEADLESS_V3:
    import torch
    from genreg_gpu_v3 import GPUEvolverV3

if HEADLESS_V5:
    from genreg_gpu_v5 import GPUEvolverV5
elif HEADLESS_V4:
    from genreg_gpu_v4 import GPUEvolverV4

from genreg_checkpoint import (
    save_checkpoint,
    load_checkpoint,
    get_latest_checkpoint,
    list_checkpoints,
)


# ================================================================
# CONSTANTS
# ================================================================
CELL_SIZE = 100
CELL_PAD = 8
BOARD_PX = CELL_SIZE * 4 + CELL_PAD * 5
HEADER_H = 70
WIN_W = BOARD_PX
WIN_H = BOARD_PX + HEADER_H

BG_COLOR = (187, 173, 160)
EMPTY_COLOR = (205, 193, 180)
HEADER_BG = (250, 248, 239)

TILE_COLORS = {
    0:    (205, 193, 180),
    2:    (238, 228, 218),
    4:    (237, 224, 200),
    8:    (242, 177, 121),
    16:   (245, 149, 99),
    32:   (246, 124, 95),
    64:   (246, 94, 59),
    128:  (237, 207, 114),
    256:  (237, 204, 97),
    512:  (237, 200, 80),
    1024: (237, 197, 63),
    2048: (237, 194, 46),
    4096: (60, 58, 50),
    8192: (60, 58, 50),
}

TILE_TEXT_DARK = (119, 110, 101)
TILE_TEXT_LIGHT = (249, 246, 242)

DEFAULT_FPS = 30
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
CONFIGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")


# ================================================================
# PROTEIN TEMPLATE — Domain-agnostic reward shaping for 2048
# ================================================================
def build_protein_template():
    """
    Define the GENREG Regulatory Genome for 2048.

    Principles (same as snake):
    1. Increasing max_tile is GOOD (progress toward goal)
    2. Keeping empty space is GOOD (survival / flexibility)
    3. Increasing score is GOOD (merging efficiently)
    4. Staying alive is GOOD (baseline survival reward)

    We do NOT tell it HOW to play — only what signals matter.
    """
    proteins = []

    # ---- SENSORS ----
    proteins.append(SensorProtein("max_tile"))
    proteins.append(SensorProtein("empty_count"))
    proteins.append(SensorProtein("score"))
    proteins.append(SensorProtein("moves_made"))
    proteins.append(SensorProtein("last_merge_count"))

    # ---- TRENDS (derivatives) ----

    # Track rate of change of max tile
    tp_tile = TrendProtein("max_tile_velocity")
    tp_tile.bind_inputs(["max_tile"])
    tp_tile.params["momentum"] = 0.5  # Fast reaction
    proteins.append(tp_tile)

    # Track rate of change of score
    tp_score = TrendProtein("score_velocity")
    tp_score.bind_inputs(["score"])
    tp_score.params["momentum"] = 0.5
    proteins.append(tp_score)

    # Track rate of change of empty cells
    tp_empty = TrendProtein("empty_velocity")
    tp_empty.bind_inputs(["empty_count"])
    tp_empty.params["momentum"] = 0.7
    proteins.append(tp_empty)

    # ---- TRUST MODIFIERS (reward signals) ----

    # REWARD: Max tile increasing (progress toward 2048)
    trust_progress = TrustModifierProtein("trust_progress")
    trust_progress.bind_inputs(["max_tile_velocity"])
    trust_progress.params["scale"] = 5.0   # Positive velocity = positive trust
    trust_progress.params["gain"] = 2.0
    proteins.append(trust_progress)

    # REWARD: Score increasing (efficient merging)
    trust_merging = TrustModifierProtein("trust_merging")
    trust_merging.bind_inputs(["score_velocity"])
    trust_merging.params["scale"] = 2.0
    trust_merging.params["gain"] = 1.0
    proteins.append(trust_merging)

    # REWARD: Keeping empty space (board health / survival)
    trust_space = TrustModifierProtein("trust_space")
    trust_space.bind_inputs(["empty_velocity"])
    trust_space.params["scale"] = 1.0     # Gaining empty space is rewarded
    trust_space.params["gain"] = 0.5
    proteins.append(trust_space)

    # REWARD: Survival baseline (every step alive gets small drip)
    trust_survival = TrustModifierProtein("trust_survival")
    trust_survival.bind_inputs(["moves_made"])
    trust_survival.params["scale"] = 0.05
    trust_survival.params["decay"] = 0.0   # Constant, no smoothing
    proteins.append(trust_survival)

    return proteins


# ================================================================
# PYGAME RENDERER
# ================================================================
class GameRenderer:
    """Renders the 2048 board in a pygame window."""

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        pygame.display.set_caption("GENREG 2048")
        self.clock = pygame.time.Clock()

        # Fonts
        self.font_big = pygame.font.SysFont("Arial", 36, bold=True)
        self.font_med = pygame.font.SysFont("Arial", 24, bold=True)
        self.font_sm = pygame.font.SysFont("Arial", 16, bold=True)
        self.font_hdr = pygame.font.SysFont("Arial", 18)

    def draw(self, board, score=0, max_tile=0, generation=0, genome_id=0, fps_target=30, extra_info=""):
        """Draw the full game state."""
        self.screen.fill(HEADER_BG)

        # ---- Header ----
        score_surf = self.font_hdr.render(f"Score: {score}", True, (119, 110, 101))
        self.screen.blit(score_surf, (CELL_PAD, 8))

        tile_surf = self.font_hdr.render(f"Max: {max_tile}", True, (119, 110, 101))
        self.screen.blit(tile_surf, (CELL_PAD + 150, 8))

        gen_surf = self.font_hdr.render(f"Gen: {generation}  ID: {genome_id}", True, (119, 110, 101))
        self.screen.blit(gen_surf, (CELL_PAD, 32))

        if extra_info:
            info_surf = self.font_sm.render(extra_info, True, (180, 120, 60))
            self.screen.blit(info_surf, (CELL_PAD, 54))

        # ---- Board background ----
        board_rect = pygame.Rect(0, HEADER_H, BOARD_PX, BOARD_PX)
        pygame.draw.rect(self.screen, BG_COLOR, board_rect, border_radius=6)

        # ---- Tiles ----
        for r in range(4):
            for c in range(4):
                x = CELL_PAD + c * (CELL_SIZE + CELL_PAD)
                y = HEADER_H + CELL_PAD + r * (CELL_SIZE + CELL_PAD)
                val = board[r][c]

                color = TILE_COLORS.get(val, (60, 58, 50))
                tile_rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(self.screen, color, tile_rect, border_radius=4)

                if val > 0:
                    text_color = TILE_TEXT_DARK if val <= 4 else TILE_TEXT_LIGHT
                    if val >= 1024:
                        font = self.font_med
                    elif val >= 100:
                        font = self.font_med
                    else:
                        font = self.font_big
                    txt = font.render(str(val), True, text_color)
                    tx = x + (CELL_SIZE - txt.get_width()) // 2
                    ty = y + (CELL_SIZE - txt.get_height()) // 2
                    self.screen.blit(txt, (tx, ty))

        pygame.display.flip()

    def tick(self, fps):
        self.clock.tick(fps)

    def quit(self):
        pygame.quit()


# ================================================================
# TKINTER CONTROL PANEL
# ================================================================
class ControlPanel:
    """Tkinter window with Training / Inference / Config tabs."""

    def __init__(self, app):
        self.app = app
        self.root = tk.Tk()
        self.root.title("GENREG 2048 — Control Panel")
        self.root.geometry("520x780")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=5, pady=5)

        self._build_config_tab()
        self._build_training_tab()
        self._build_charts_tab()
        self._build_inference_tab()

    # --------------------------------------------------------
    # CONFIG TAB
    # --------------------------------------------------------
    def _build_config_tab(self):
        frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(frame, text="  Config  ")

        # Scrollable canvas so everything fits
        canvas = tk.Canvas(frame, highlightthickness=0)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas, padding=5)
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        # Mousewheel scroll
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(-1 * (e.delta // 120), "units"))
        canvas.bind_all("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))
        canvas.bind_all("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))

        ttk.Label(inner, text="GENREG 2048 Configuration", font=("Arial", 14, "bold")).pack(anchor="w", pady=(0, 10))

        # --- SAVED CONFIGS SECTION ---
        cfg_mgr = ttk.LabelFrame(inner, text="Saved Configs", padding=8)
        cfg_mgr.pack(fill="x", pady=(0, 10))

        # Config selector row
        sel_row = ttk.Frame(cfg_mgr)
        sel_row.pack(fill="x", pady=2)
        ttk.Label(sel_row, text="Config:", width=8, anchor="w").pack(side="left")
        self._config_name_var = tk.StringVar(value="")
        self._config_combo = ttk.Combobox(sel_row, textvariable=self._config_name_var, width=24, state="readonly")
        self._config_combo.pack(side="left", padx=5)
        self._config_combo.bind("<<ComboboxSelected>>", self._on_config_selected)
        ttk.Button(sel_row, text="Load", command=self._load_named_config).pack(side="left", padx=2)

        # Save row
        save_row = ttk.Frame(cfg_mgr)
        save_row.pack(fill="x", pady=2)
        ttk.Label(save_row, text="Name:", width=8, anchor="w").pack(side="left")
        self._save_name_var = tk.StringVar(value="")
        self._save_name_entry = ttk.Entry(save_row, textvariable=self._save_name_var, width=24)
        self._save_name_entry.pack(side="left", padx=5)
        ttk.Button(save_row, text="Save", command=self._save_named_config).pack(side="left", padx=2)
        ttk.Button(save_row, text="Delete", command=self._delete_named_config).pack(side="left", padx=2)

        # Info label
        self._config_info_var = tk.StringVar(value="")
        ttk.Label(cfg_mgr, textvariable=self._config_info_var, foreground="gray").pack(anchor="w", pady=(2, 0))

        # Load the config list
        self._loaded_config_name = None  # tracks which config is actively loaded
        self._refresh_config_list()

        ttk.Separator(inner, orient="horizontal").pack(fill="x", pady=5)

        # --- CONFIG ENTRIES ---
        self.cfg = {}
        configs = [
            ("population_size", "Population Size", "50"),
            ("hidden_size",     "Hidden Layer Size", "32"),
            ("mutation_rate",   "Mutation Rate", "0.1"),
            ("max_generations", "Max Generations", "5000"),
            ("target_tile",     "Target Tile", "2048"),
            ("starting_energy", "Starting Energy", "50"),
            ("checkpoint_freq", "Checkpoint Frequency", "10"),
            ("survival_pct",    "Survival % (top N%)", "20"),
            ("trust_inherit",   "Trust Inheritance %", "10"),
            ("child_mutation",  "Child Mutation Rate", "0.05"),
            ("n_games",         "Games per Genome (V2)", "3"),
            ("ratchet_strength","Ratchet Strength (0-10)", "2.0"),
            ("proximity_strength","Proximity Bonus (0-5)", "1.0"),
        ]

        for key, label, default in configs:
            row = ttk.Frame(inner)
            row.pack(fill="x", pady=2)
            ttk.Label(row, text=label, width=24, anchor="w").pack(side="left")
            var = tk.StringVar(value=default)
            ttk.Entry(row, textvariable=var, width=12).pack(side="left", padx=5)
            self.cfg[key] = var

        ttk.Separator(inner, orient="horizontal").pack(fill="x", pady=10)

        ttk.Label(inner, text="Checkpoint Directory:").pack(anchor="w")
        dir_frame = ttk.Frame(inner)
        dir_frame.pack(fill="x", pady=2)
        self.cfg_ckpt_dir = tk.StringVar(value=CHECKPOINT_DIR)
        ttk.Entry(dir_frame, textvariable=self.cfg_ckpt_dir, width=35).pack(side="left", padx=(0, 5))
        ttk.Button(dir_frame, text="Browse", command=self._browse_ckpt_dir).pack(side="left")

    def _browse_ckpt_dir(self):
        d = filedialog.askdirectory(initialdir=self.cfg_ckpt_dir.get())
        if d:
            self.cfg_ckpt_dir.set(d)

    # --- Named config persistence (JSON in configs/) ---

    def _configs_dir(self):
        os.makedirs(CONFIGS_DIR, exist_ok=True)
        return CONFIGS_DIR

    def _refresh_config_list(self):
        """Rescan the configs directory and update the combo box."""
        d = self._configs_dir()
        names = sorted(
            os.path.splitext(f)[0]
            for f in os.listdir(d)
            if f.endswith(".json")
        )
        self._config_combo["values"] = names
        if self._loaded_config_name and self._loaded_config_name in names:
            self._config_name_var.set(self._loaded_config_name)

    def _on_config_selected(self, _event=None):
        """When user picks a config from the dropdown, populate the save-name
        field so hitting Save will update that same config."""
        name = self._config_name_var.get()
        if name:
            self._save_name_var.set(name)

    def _load_named_config(self):
        """Load the selected config from disk into the panel fields."""
        name = self._config_name_var.get()
        if not name:
            self._config_info_var.set("Select a config first.")
            return
        path = os.path.join(self._configs_dir(), f"{name}.json")
        if not os.path.isfile(path):
            self._config_info_var.set(f"Not found: {name}.json")
            return
        with open(path, "r") as f:
            data = json.load(f)
        for k, v in data.items():
            if k == "checkpoint_dir":
                self.cfg_ckpt_dir.set(str(v))
            elif k in self.cfg:
                self.cfg[k].set(str(v))
        self._loaded_config_name = name
        self._save_name_var.set(name)
        self._config_info_var.set(f"Loaded: {name}")

    def _save_named_config(self):
        """Save the current panel values as a named config JSON."""
        name = self._save_name_var.get().strip()
        if not name:
            self._config_info_var.set("Enter a name to save.")
            return
        # Sanitize filename
        safe = "".join(c for c in name if c.isalnum() or c in "._- ")
        if not safe:
            self._config_info_var.set("Invalid name.")
            return
        data = {}
        for k, var in self.cfg.items():
            data[k] = var.get()
        data["checkpoint_dir"] = self.cfg_ckpt_dir.get()
        path = os.path.join(self._configs_dir(), f"{safe}.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        self._loaded_config_name = safe
        self._config_name_var.set(safe)
        self._refresh_config_list()
        self._config_info_var.set(f"Saved: {safe}")

    def _delete_named_config(self):
        """Delete the selected named config."""
        name = self._config_name_var.get()
        if not name:
            name = self._save_name_var.get().strip()
        if not name:
            self._config_info_var.set("Select a config to delete.")
            return
        path = os.path.join(self._configs_dir(), f"{name}.json")
        if not os.path.isfile(path):
            self._config_info_var.set(f"Not found: {name}.json")
            return
        if not messagebox.askyesno("Delete Config", f"Delete config '{name}'?"):
            return
        os.remove(path)
        if self._loaded_config_name == name:
            self._loaded_config_name = None
        self._config_name_var.set("")
        self._save_name_var.set("")
        self._refresh_config_list()
        self._config_info_var.set(f"Deleted: {name}")

    def get_config(self):
        """Read all config values as a dict."""
        return {
            "population_size": int(self.cfg["population_size"].get()),
            "hidden_size":     int(self.cfg["hidden_size"].get()),
            "mutation_rate":   float(self.cfg["mutation_rate"].get()),
            "max_generations": int(self.cfg["max_generations"].get()),
            "target_tile":     int(self.cfg["target_tile"].get()),
            "starting_energy": int(self.cfg["starting_energy"].get()),
            "checkpoint_freq": int(self.cfg["checkpoint_freq"].get()),
            "survival_pct":    int(self.cfg["survival_pct"].get()),
            "trust_inherit":   float(self.cfg["trust_inherit"].get()) / 100.0,
            "child_mutation":  float(self.cfg["child_mutation"].get()),
            "n_games":         int(self.cfg["n_games"].get()),
            "ratchet_strength": float(self.cfg["ratchet_strength"].get()),
            "proximity_strength": float(self.cfg["proximity_strength"].get()),
            "checkpoint_dir":  self.cfg_ckpt_dir.get(),
        }

    # --------------------------------------------------------
    # TRAINING TAB
    # --------------------------------------------------------
    def _build_training_tab(self):
        frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(frame, text="  Training  ")

        ttk.Label(frame, text="Training Controls", font=("Arial", 14, "bold")).pack(anchor="w", pady=(0, 10))

        # Buttons
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill="x", pady=5)

        self.btn_start = ttk.Button(btn_frame, text="Start Training", command=self._start_training)
        self.btn_start.pack(side="left", padx=2)

        self.btn_stop = ttk.Button(btn_frame, text="Stop", command=self._stop_training, state="disabled")
        self.btn_stop.pack(side="left", padx=2)

        self.btn_resume = ttk.Button(btn_frame, text="Resume from Checkpoint", command=self._resume_training)
        self.btn_resume.pack(side="left", padx=2)

        ttk.Separator(frame, orient="horizontal").pack(fill="x", pady=10)

        # Extend controls
        ext_frame = ttk.Frame(frame)
        ext_frame.pack(fill="x", pady=5)
        ttk.Label(ext_frame, text="Extend:").pack(side="left")
        ttk.Button(ext_frame, text="+1000", command=lambda: self._extend(1000)).pack(side="left", padx=2)
        ttk.Button(ext_frame, text="+5000", command=lambda: self._extend(5000)).pack(side="left", padx=2)
        ttk.Button(ext_frame, text="+10000", command=lambda: self._extend(10000)).pack(side="left", padx=2)
        self.extend_var = tk.StringVar(value="")
        ext_entry = ttk.Entry(ext_frame, textvariable=self.extend_var, width=8)
        ext_entry.pack(side="left", padx=2)
        ttk.Button(ext_frame, text="+Custom", command=self._extend_custom).pack(side="left", padx=2)

        ttk.Separator(frame, orient="horizontal").pack(fill="x", pady=10)

        # FPS slider
        fps_frame = ttk.Frame(frame)
        fps_frame.pack(fill="x", pady=2)
        ttk.Label(fps_frame, text="Viz FPS:").pack(side="left")
        self.train_fps_var = tk.IntVar(value=60)
        self.train_fps_slider = ttk.Scale(fps_frame, from_=1, to=240, variable=self.train_fps_var, orient="horizontal")
        self.train_fps_slider.pack(side="left", fill="x", expand=True, padx=5)
        self.train_fps_label = ttk.Label(fps_frame, text="60")
        self.train_fps_label.pack(side="left")

        ttk.Separator(frame, orient="horizontal").pack(fill="x", pady=10)

        # Live stats
        ttk.Label(frame, text="Live Statistics", font=("Arial", 12, "bold")).pack(anchor="w", pady=(0, 5))

        self.stats_text = tk.Text(frame, height=30, width=55, font=("Courier", 10), state="disabled", bg="#f8f8f0")
        self.stats_text.pack(fill="both", expand=True)

    def _extend(self, n):
        """Extend max_generations by n. Works while training."""
        old = int(self.cfg["max_generations"].get())
        new_val = old + n
        self.cfg["max_generations"].set(str(new_val))
        # If app has a running config, update it too
        if self.app.mode == "training":
            self.app._update_max_generations(new_val)

    def _extend_custom(self):
        try:
            n = int(self.extend_var.get())
            if n > 0:
                self._extend(n)
        except ValueError:
            pass

    def _start_training(self):
        self.app.start_training(resume_checkpoint=None)
        self.btn_start.config(state="disabled")
        self.btn_resume.config(state="disabled")
        self.btn_stop.config(state="normal")

    def _stop_training(self):
        self.app.stop_training()
        # Re-enable Start so user can continue from where they left off
        self.btn_start.config(state="normal")
        self.btn_resume.config(state="normal")
        self.btn_stop.config(state="disabled")

    def _resume_training(self):
        ckpt_dir = self.cfg_ckpt_dir.get()
        path = get_latest_checkpoint(ckpt_dir)
        if not path:
            path = filedialog.askopenfilename(
                title="Select Checkpoint",
                filetypes=[("Pickle", "*.pkl")],
                initialdir=ckpt_dir,
            )
        if path:
            self.app.start_training(resume_checkpoint=path)
            self.btn_start.config(state="disabled")
            self.btn_resume.config(state="disabled")
            self.btn_stop.config(state="normal")

    def update_stats(self, text):
        self.stats_text.config(state="normal")
        self.stats_text.delete("1.0", "end")
        self.stats_text.insert("1.0", text)
        self.stats_text.config(state="disabled")

    # --------------------------------------------------------
    # CHARTS TAB
    # --------------------------------------------------------
    _CHART_COLORS = ["#4fc3f7", "#81c784", "#ffb74d", "#ce93d8", None, "#ef5350"]
    _CHART_TITLES = [
        "Avg Score", "Best Tile", "Avg Tile", "Best Trust",
        "Tile Distribution", "Gen Speed (s)",
    ]

    def _build_charts_tab(self):
        frame = ttk.Frame(self.notebook, padding=2)
        self.notebook.add(frame, text="  Charts  ")

        # Toggle controls
        ctrl = ttk.Frame(frame)
        ctrl.pack(fill="x", pady=(2, 4))
        self.charts_enabled_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(ctrl, text="Enable Charts (reduces training speed)",
                        variable=self.charts_enabled_var).pack(side="left", padx=4)

        self.chart_fig = Figure(figsize=(6, 7), dpi=80, facecolor="#2b2b2b")
        self.chart_axes = self.chart_fig.subplots(3, 2)
        self.chart_fig.subplots_adjust(
            left=0.10, right=0.97, top=0.96, bottom=0.06, hspace=0.45, wspace=0.35)

        # Style axes once — never call clear() again for line charts
        self._chart_lines_raw = []   # raw data lines (thin, transparent)
        self._chart_lines_smooth = []  # smoothed overlay lines

        for idx, ax in enumerate(self.chart_axes.flat):
            ax.set_facecolor("#1e1e1e")
            ax.tick_params(colors="#999", labelsize=7)
            for spine in ax.spines.values():
                spine.set_color("#555")
            ax.set_title(self._CHART_TITLES[idx], color="#ddd", fontsize=9)

            if idx == 4:
                # Bar chart — no persistent lines
                self._chart_lines_raw.append(None)
                self._chart_lines_smooth.append(None)
            else:
                c = self._CHART_COLORS[idx]
                raw_line, = ax.plot([], [], color=c, linewidth=1, alpha=0.35)
                smooth_line, = ax.plot([], [], color=c, linewidth=1.5)
                self._chart_lines_raw.append(raw_line)
                self._chart_lines_smooth.append(smooth_line)

        # Best Tile axis: log scale set once
        self.chart_axes[0, 1].set_yscale("log", base=2)
        self.chart_axes[0, 1].set_ylim(bottom=2, top=4)

        self.chart_canvas = FigureCanvasTkAgg(self.chart_fig, master=frame)
        self.chart_canvas.get_tk_widget().pack(fill="both", expand=True)
        self.chart_canvas.draw()

        # Throttle state
        self._chart_last_len = 0
        self._chart_render_pending = False

    def _smooth(self, data):
        """Fast rolling average — returns (offset, smoothed) or None."""
        n = len(data)
        if n < 6:
            return None
        k = min(20, max(3, n // 5))
        arr = np.array(data, dtype=np.float64)
        cs = np.cumsum(arr)
        cs = np.insert(cs, 0, 0.0)
        smoothed = (cs[k:] - cs[:-k]) / k
        return k - 1, smoothed

    def update_charts(self, app):
        """Update line data on persistent artists — no clear/replot."""
        if not self.charts_enabled_var.get():
            return
        n = len(app.history_gen)
        if n == 0 or n == self._chart_last_len:
            return
        self._chart_last_len = n

        # Window last 500 points
        W = 500
        g = np.array(app.history_gen[-W:], dtype=np.float64)

        # Data arrays for the 5 line charts (idx 0-3, 5)
        datasets = [
            app.history_avg_score,
            app.history_best_tile,
            app.history_avg_tile,
            app.history_best_trust,
            None,  # bar chart placeholder
            app.history_gen_time,
        ]

        for idx in (0, 1, 2, 3, 5):
            raw_data = datasets[idx][-W:]
            raw_line = self._chart_lines_raw[idx]
            smooth_line = self._chart_lines_smooth[idx]

            # Update raw line
            raw_line.set_data(g, raw_data)

            # Update smoothed line
            sm = self._smooth(raw_data)
            if sm is not None:
                offset, sv = sm
                smooth_line.set_data(g[offset:], sv)
            else:
                smooth_line.set_data([], [])

            # Rescale axis
            ax = self.chart_axes.flat[idx]
            ax.set_xlim(g[0], g[-1])
            if idx == 1:
                # Best Tile — log2 scale, keep bottom=2
                mx = max(raw_data) if raw_data else 4
                ax.set_ylim(bottom=2, top=max(4, mx * 1.5))
            else:
                vals = np.array(raw_data, dtype=np.float64)
                lo, hi = vals.min(), vals.max()
                margin = max((hi - lo) * 0.08, 1e-6)
                ax.set_ylim(lo - margin, hi + margin)

        # Tile distribution bar chart — must rebuild bars (fast: few bars)
        ax_bar = self.chart_axes[2, 0]
        ax_bar.clear()
        ax_bar.set_facecolor("#1e1e1e")
        ax_bar.set_title("Tile Distribution", color="#ddd", fontsize=9)
        ax_bar.tick_params(colors="#999", labelsize=7)
        for spine in ax_bar.spines.values():
            spine.set_color("#555")
        dist = app.history_tile_dist
        if dist:
            tiles_sorted = sorted(dist.keys())
            labels = [str(t) for t in tiles_sorted]
            counts = [dist[t] for t in tiles_sorted]
            colors = []
            for t in tiles_sorted:
                if t >= 2048: colors.append("#e040fb")
                elif t >= 512: colors.append("#81c784")
                elif t >= 128: colors.append("#ffb74d")
                elif t >= 32: colors.append("#4fc3f7")
                else: colors.append("#78909c")
            ax_bar.bar(labels, counts, color=colors, edgecolor="#333", linewidth=0.5)
            ax_bar.tick_params(axis="x", rotation=45)

        # Schedule a single canvas blit — non-blocking
        if not self._chart_render_pending:
            self._chart_render_pending = True
            self.root.after_idle(self._chart_flush)

    def _chart_flush(self):
        """Deferred canvas draw — runs when tkinter is idle."""
        self._chart_render_pending = False
        try:
            self.chart_canvas.draw_idle()
        except Exception:
            pass

    # --------------------------------------------------------
    # INFERENCE TAB
    # --------------------------------------------------------
    def _build_inference_tab(self):
        frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(frame, text="  Inference  ")

        ttk.Label(frame, text="Inference Mode", font=("Arial", 14, "bold")).pack(anchor="w", pady=(0, 10))

        # Load checkpoint
        load_frame = ttk.Frame(frame)
        load_frame.pack(fill="x", pady=5)
        ttk.Button(load_frame, text="Load Checkpoint", command=self._load_inference).pack(side="left", padx=2)
        ttk.Button(load_frame, text="Load Latest", command=self._load_latest).pack(side="left", padx=2)

        ttk.Separator(frame, orient="horizontal").pack(fill="x", pady=10)

        # Genome selector
        sel_frame = ttk.Frame(frame)
        sel_frame.pack(fill="x", pady=2)
        ttk.Label(sel_frame, text="Genome:").pack(side="left")
        self.genome_var = tk.IntVar(value=0)
        self.genome_spin = ttk.Spinbox(sel_frame, from_=0, to=0, textvariable=self.genome_var, width=6,
                                        command=self._select_genome)
        self.genome_spin.pack(side="left", padx=5)
        self.genome_info_label = ttk.Label(sel_frame, text="No checkpoint loaded")
        self.genome_info_label.pack(side="left", padx=5)

        # FPS slider
        fps_frame = ttk.Frame(frame)
        fps_frame.pack(fill="x", pady=5)
        ttk.Label(fps_frame, text="Playback FPS:").pack(side="left")
        self.inf_fps_var = tk.IntVar(value=10)
        ttk.Scale(fps_frame, from_=1, to=120, variable=self.inf_fps_var, orient="horizontal").pack(
            side="left", fill="x", expand=True, padx=5)
        self.inf_fps_label = ttk.Label(fps_frame, text="10")
        self.inf_fps_label.pack(side="left")

        # Play / Restart
        ctrl_frame = ttk.Frame(frame)
        ctrl_frame.pack(fill="x", pady=5)
        self.btn_play = ttk.Button(ctrl_frame, text="Play", command=self._play_inference, state="disabled")
        self.btn_play.pack(side="left", padx=2)
        self.btn_restart = ttk.Button(ctrl_frame, text="Restart Game", command=self._restart_game, state="disabled")
        self.btn_restart.pack(side="left", padx=2)
        self.btn_stop_inf = ttk.Button(ctrl_frame, text="Stop", command=self._stop_inference, state="disabled")
        self.btn_stop_inf.pack(side="left", padx=2)

        ttk.Separator(frame, orient="horizontal").pack(fill="x", pady=10)

        # Inference stats
        ttk.Label(frame, text="Game Stats", font=("Arial", 12, "bold")).pack(anchor="w", pady=(0, 5))
        self.inf_stats_text = tk.Text(frame, height=15, width=55, font=("Courier", 10), state="disabled", bg="#f8f8f0")
        self.inf_stats_text.pack(fill="both", expand=True)

    def _load_inference(self):
        path = filedialog.askopenfilename(
            title="Select Checkpoint",
            filetypes=[("Pickle", "*.pkl")],
            initialdir=self.cfg_ckpt_dir.get(),
        )
        if path:
            self.app.load_inference_checkpoint(path)

    def _load_latest(self):
        path = get_latest_checkpoint(self.cfg_ckpt_dir.get())
        if path:
            self.app.load_inference_checkpoint(path)
        else:
            messagebox.showinfo("No Checkpoint", "No checkpoints found in the checkpoint directory.")

    def _select_genome(self):
        self.app.select_inference_genome(self.genome_var.get())

    def _play_inference(self):
        self.app.start_inference()
        self.btn_play.config(state="disabled")
        self.btn_stop_inf.config(state="normal")
        self.btn_restart.config(state="normal")

    def _restart_game(self):
        self.app.restart_inference_game()

    def _stop_inference(self):
        self.app.stop_inference()
        self.btn_play.config(state="normal")
        self.btn_stop_inf.config(state="disabled")

    def update_inference_stats(self, text):
        self.inf_stats_text.config(state="normal")
        self.inf_stats_text.delete("1.0", "end")
        self.inf_stats_text.insert("1.0", text)
        self.inf_stats_text.config(state="disabled")

    def update_genome_selector(self, count, genomes):
        self.genome_spin.config(to=max(0, count - 1))
        if count > 0:
            g = genomes[0]
            self.genome_info_label.config(text=f"ID:{g.id} Trust:{g.trust:.1f} Max:{g.max_tile}")
            self.btn_play.config(state="normal")
            self.btn_restart.config(state="normal")
        else:
            self.genome_info_label.config(text="No genomes")

    # --------------------------------------------------------
    def _on_close(self):
        self.app.running = False

    def update(self):
        # Update FPS labels
        self.train_fps_label.config(text=str(self.train_fps_var.get()))
        self.inf_fps_label.config(text=str(self.inf_fps_var.get()))
        try:
            self.root.update_idletasks()
            self.root.update()
        except tk.TclError:
            self.app.running = False


# ================================================================
# MAIN APPLICATION
# ================================================================
class GENREGApp:
    def __init__(self, headless=False):
        self.running = True
        self.headless = headless
        self.mode = "idle"  # idle | training | inference

        # Renderer (None in headless mode)
        self.renderer = None if headless else GameRenderer()

        # Control panel
        self.panel = ControlPanel(self)

        # Training state
        self.population = None
        self.template_proteins = None
        self.generation = 0
        self.env = None
        self.train_stop_flag = False

        # Per-frame training state
        self.train_genome_idx = 0
        self.train_signals = None
        self.train_done = False
        self.train_current_genome = None
        self.train_episode_started = False

        # Training stats tracking
        self.best_tile_ever = 0
        self.best_score_ever = 0
        self.gen_best_tile = 0
        self.gen_best_score = 0
        self.gen_tiles = []
        self.gen_scores = []
        self.games_played = 0

        # Inference state
        self.inf_population = None
        self.inf_genome = None
        self.inf_env = None
        self.inf_signals = None
        self.inf_done = True
        self.inf_generation = 0
        self.inf_games = 0
        self.inf_best_tile = 0
        self.inf_total_score = 0

        # Logger (for headless v2/v3)
        if HEADLESS_V2 or HEADLESS_V3:
            from genreg_logger import TrainingLogger
            self._logger = TrainingLogger(log_dir=os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "logs"))
        else:
            self._logger = None

        # Chart history (per-generation data points)
        self.history_gen = []          # generation numbers
        self.history_avg_score = []    # avg score per gen
        self.history_best_tile = []    # best tile per gen
        self.history_avg_tile = []     # avg tile per gen
        self.history_best_trust = []   # best trust per gen
        self.history_gen_time = []     # seconds per generation
        self.history_tile_dist = {}    # latest gen tile distribution
        self._gen_start_time = None

        # Current board for rendering
        self.current_board = [[0]*4 for _ in range(4)]
        self.current_score = 0
        self.current_max_tile = 0
        self.current_gen = 0
        self.current_genome_id = 0
        self.current_extra = ""

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------
    def _resize_population(self, cfg):
        """Resize population to match config. Trims worst or pads from elite."""
        import random as _rand
        target = cfg["population_size"]
        current = self.population.size
        self.population.genomes.sort(key=lambda g: g.trust, reverse=True)

        if current > target:
            self.population.genomes = self.population.genomes[:target]
            self.population.size = target
        else:
            needed = target - current
            elite = self.population.genomes[:max(1, int(current * 0.2))]
            for _ in range(needed):
                parent = _rand.choice(elite)
                child = parent.clone()
                child.trust = parent.trust * cfg.get("trust_inherit", 0.1)
                child.mutate(rate=cfg.get("child_mutation", 0.05))
                self.population.genomes.append(child)
            self.population.size = target
        self.population.active = 0

    def _update_max_generations(self, new_val):
        """Update max_generations in the live config (called from extend buttons)."""
        # The panel cfg var is already updated; this is for the headless path
        # which reads config each tick via get_config().
        pass  # get_config() reads the panel vars directly, so nothing extra needed

    def start_training(self, resume_checkpoint=None):
        cfg = self.panel.get_config()
        self.train_stop_flag = False

        if resume_checkpoint:
            self.population, self.generation, self.template_proteins, saved_config = load_checkpoint(resume_checkpoint)
            # Fix controller dimensions to match actual weight shapes (V3 encoder changes input_size)
            for g in self.population.genomes:
                if g.controller.w1 is not None and len(g.controller.w1) > 0:
                    g.controller.input_size = len(g.controller.w1[0])
                    g.controller.hidden_size = len(g.controller.b1)
            if saved_config:
                for k, v in saved_config.items():
                    if k in self.panel.cfg and k != "checkpoint_dir":
                        self.panel.cfg[k].set(str(v))
            self.env = Game2048Env(target_tile=cfg["target_tile"])
            # Auto-extend if checkpoint is past the limit
            if self.generation >= cfg["max_generations"]:
                new_max = self.generation + 5000
                self.panel.cfg["max_generations"].set(str(new_max))
        elif self.population is not None:
            # Resume from where we left off (Stop + Start)
            pass
        else:
            # Fresh start
            self.template_proteins = build_protein_template()
            self.env = Game2048Env(target_tile=cfg["target_tile"])
            self.env.max_energy = cfg["starting_energy"]

            example_signals = self.env.reset()
            input_size = len(example_signals)

            self.population = GENREGPopulation(
                template_proteins=self.template_proteins,
                input_size=input_size,
                hidden_size=cfg["hidden_size"],
                output_size=4,
                size=cfg["population_size"],
                mutation_rate=cfg["mutation_rate"],
            )
            self.generation = 0

        # Resize population if config doesn't match
        if self.population.size != cfg["population_size"]:
            self._resize_population(cfg)

        # Reset per-frame training state
        self.train_genome_idx = 0
        self.train_done = False
        self.train_episode_started = False
        self.gen_tiles = []
        self.gen_scores = []
        self.gen_best_tile = 0
        self.gen_best_score = 0
        self._train_start_time = time.time()
        self.mode = "training"

        # GPU evolver for --headless_v2
        if HEADLESS_V2:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            gpu_cfg = {
                "population_size": cfg["population_size"],
                "hidden_size": cfg["hidden_size"],
                "mutation_rate": cfg["mutation_rate"],
                "target_tile": cfg["target_tile"],
                "starting_energy": cfg["starting_energy"],
                "survival_pct": cfg.get("survival_pct", 20),
                "trust_inherit": cfg.get("trust_inherit", 0.1),
                "child_mutation_rate": cfg.get("child_mutation", 0.05),
                "n_games": cfg.get("n_games", 3),
                "ratchet_strength": cfg.get("ratchet_strength", 2.0),
            }
            self._gpu_evolver = GPUEvolver(self.population, gpu_cfg, device=device)
            self._gpu_cfg = gpu_cfg
            print(f"[V2] GPU Evolver initialized: {self.population.size} genomes, "
                  f"{gpu_cfg['n_games']} games/genome, device={device}")

        # GPU evolver for --headless_v3 / --headless_v4 (with evolved encoder)
        if HEADLESS_V3:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            gpu_cfg = {
                "population_size": cfg["population_size"],
                "hidden_size": cfg["hidden_size"],
                "mutation_rate": cfg["mutation_rate"],
                "target_tile": cfg["target_tile"],
                "starting_energy": cfg["starting_energy"],
                "survival_pct": cfg.get("survival_pct", 20),
                "trust_inherit": cfg.get("trust_inherit", 0.1),
                "child_mutation_rate": cfg.get("child_mutation", 0.05),
                "n_games": cfg.get("n_games", 3),
                "invalid_move_penalty": 2.0,
            }
            encoder_dim = cfg.get("hidden_size", 32)
            if HEADLESS_V5:
                self._gpu_evolver = GPUEvolverV5(
                    self.population, gpu_cfg, device=device, encoder_dim=encoder_dim)
                version_tag = "V5"
            elif HEADLESS_V4:
                self._gpu_evolver = GPUEvolverV4(
                    self.population, gpu_cfg, device=device, encoder_dim=encoder_dim)
                version_tag = "V4"
            else:
                self._gpu_evolver = GPUEvolverV3(
                    self.population, gpu_cfg, device=device, encoder_dim=encoder_dim)
                version_tag = "V3"
            self._gpu_cfg = gpu_cfg
            print(f"[{version_tag}] GPU Evolver + Encoder initialized: {self.population.size} genomes, "
                  f"22→{encoder_dim} encoder, {NUM_ACTIVATIONS} activation catalog, "
                  f"{gpu_cfg['n_games']} games/genome, device={device}")

    def stop_training(self):
        self.train_stop_flag = True
        # Close logger
        if self._logger is not None:
            self._logger.close()
        # Sync GPU state back to CPU before saving (v2/v3)
        if (HEADLESS_V2 or HEADLESS_V3) and hasattr(self, '_gpu_evolver'):
            ck_tiles, ck_scores, ck_trust, _ = self._gpu_evolver.run_generation(n_games=1)
            self._gpu_evolver.sync_to_cpu(self.population, tiles=ck_tiles, scores=ck_scores, trust=ck_trust)
        # Save checkpoint on stop
        if self.population and self.template_proteins:
            cfg = self.panel.get_config()
            save_checkpoint(self.population, self.generation, self.template_proteins, cfg["checkpoint_dir"], config=cfg)
        self.mode = "idle"

    def _run_full_game(self, genome, cfg):
        """Run a complete game for one genome. Returns (max_tile, score, moves)."""
        env = Game2048Env(target_tile=cfg["target_tile"])
        env.max_energy = cfg["starting_energy"]
        signals = env.reset()
        done = False

        while not done:
            inp = [float(v) for v in signals.values()]
            action = genome.controller.forward(inp)
            signals, done = env.step(action)
            genome.forward(signals)

        return env.max_tile, env.score, env.moves_made, env

    def _training_step_headless(self):
        """Run an entire generation at once (no rendering). Called per tkinter tick."""
        cfg = self.panel.get_config()

        if self.train_stop_flag:
            return

        if self.generation >= cfg["max_generations"]:
            self.stop_training()
            self.panel._stop_training()
            return

        self._gen_start_time = time.time()

        # Run the full generation
        self.gen_tiles = []
        self.gen_scores = []
        self.gen_best_tile = 0
        self.gen_best_score = 0

        for idx in range(self.population.size):
            genome = self.population.get_active()
            max_tile, score, moves, env = self._run_full_game(genome, cfg)

            genome.max_tile = max_tile
            genome.game_score = score
            genome.step_count = moves

            self.gen_tiles.append(max_tile)
            self.gen_scores.append(score)
            self.gen_best_tile = max(self.gen_best_tile, max_tile)
            self.gen_best_score = max(self.gen_best_score, score)
            self.best_tile_ever = max(self.best_tile_ever, max_tile)
            self.best_score_ever = max(self.best_score_ever, score)
            self.games_played += 1

            if idx < self.population.size - 1:
                self.population.next_genome()

        # Keep last env for stats display
        self.env = env
        self.train_current_genome = genome
        self.train_genome_idx = self.population.size - 1

        # Evolve
        self.population.evolve(
            survival_pct=cfg.get("survival_pct", 20),
            trust_inherit=cfg.get("trust_inherit", 0.1),
            child_mutation_rate=cfg.get("child_mutation", 0.05),
        )
        self.generation += 1
        self._record_generation()

        # Checkpoint
        if self.generation % cfg["checkpoint_freq"] == 0:
            save_checkpoint(self.population, self.generation, self.template_proteins, cfg["checkpoint_dir"], config=cfg)

    def _training_step_gpu(self):
        """GPU-accelerated generation with multi-game ensemble. Used by --headless_v2."""
        cfg = self.panel.get_config()

        if self.train_stop_flag:
            return

        if self.generation >= cfg["max_generations"]:
            self.stop_training()
            self.panel._stop_training()
            return

        self._gen_start_time = time.time()

        n_games = cfg.get("n_games", 3)
        # Update evolver config with live values (hot-settable)
        self._gpu_evolver.cfg["ratchet_strength"] = cfg.get("ratchet_strength", 2.0)
        self._gpu_evolver.cfg["proximity_strength"] = cfg.get("proximity_strength", 1.0)
        tiles, scores, trust, efficiencies = self._gpu_evolver.run_generation(n_games=n_games)

        # Evolve on GPU
        best_tr, med_tr, low_tr = self._gpu_evolver.evolve(
            trust,
            survival_pct=cfg.get("survival_pct", 20),
            trust_inherit=cfg.get("trust_inherit", 0.1),
            child_mutation_rate=cfg.get("child_mutation", 0.05),
            tiles=tiles,
            scores=scores,
            efficiencies=efficiencies,
        )
        self._last_trust = (best_tr, med_tr, low_tr)
        self.generation += 1

        # Update stats
        self.gen_tiles = tiles
        self.gen_scores = scores
        self.gen_best_tile = max(tiles)
        self.gen_best_score = max(scores)
        self.gen_avg_tile = sum(tiles) / len(tiles)
        self.gen_avg_score = sum(scores) / len(scores)
        self.best_tile_ever = max(self.best_tile_ever, self.gen_best_tile)
        self.best_score_ever = max(self.best_score_ever, self.gen_best_score)
        self.games_played += len(tiles) * n_games

        self._record_generation()

        # Log to file
        if self._logger is not None:
            total_elapsed = time.time() - (self._train_start_time or time.time())
            gen_time = time.time() - self._gen_start_time if self._gen_start_time else 0
            bt, mt, lt = getattr(self, '_last_trust', (0, 0, 0))
            genome_ids = list(range(len(tiles)))
            self._logger.log_generation(
                self.generation, tiles, scores,
                trust_best=bt, trust_med=mt, trust_low=lt,
                gen_time=gen_time, elapsed_total=total_elapsed,
                genome_ids=genome_ids,
            )

        # Checkpoint — sync to CPU first
        if self.generation % cfg["checkpoint_freq"] == 0:
            ck_tiles, ck_scores, ck_trust, _ = self._gpu_evolver.run_generation(n_games=1)
            self._gpu_evolver.sync_to_cpu(self.population, tiles=ck_tiles, scores=ck_scores, trust=ck_trust)
            save_checkpoint(self.population, self.generation, self.template_proteins, cfg["checkpoint_dir"], config=cfg)

    def _training_step(self):
        """Advance training by one game-step (called each frame)."""
        cfg = self.panel.get_config()

        if self.train_stop_flag:
            return

        if self.generation >= cfg["max_generations"]:
            self.stop_training()
            self.panel._stop_training()
            return

        # Start new episode for current genome if needed
        if not self.train_episode_started:
            if self.train_genome_idx == 0:
                self._gen_start_time = time.time()
            self.train_current_genome = self.population.get_active()
            self.env = Game2048Env(target_tile=cfg["target_tile"])
            self.env.max_energy = cfg["starting_energy"]
            self.train_signals = self.env.reset()
            self.train_done = False
            self.train_episode_started = True

        if self.train_done:
            # Episode finished for this genome — record stats
            genome = self.train_current_genome
            genome.max_tile = self.env.max_tile
            genome.game_score = self.env.score
            genome.step_count = self.env.moves_made

            self.gen_tiles.append(self.env.max_tile)
            self.gen_scores.append(self.env.score)
            self.gen_best_tile = max(self.gen_best_tile, self.env.max_tile)
            self.gen_best_score = max(self.gen_best_score, self.env.score)
            self.best_tile_ever = max(self.best_tile_ever, self.env.max_tile)
            self.best_score_ever = max(self.best_score_ever, self.env.score)
            self.games_played += 1

            # Move to next genome
            self.train_genome_idx += 1

            if self.train_genome_idx >= self.population.size:
                # Generation complete — evolve
                self.population.evolve(
                    survival_pct=cfg.get("survival_pct", 20),
                    trust_inherit=cfg.get("trust_inherit", 0.1),
                    child_mutation_rate=cfg.get("child_mutation", 0.05),
                )
                self.generation += 1
                self._record_generation()

                # Checkpoint
                if self.generation % cfg["checkpoint_freq"] == 0:
                    save_checkpoint(self.population, self.generation, self.template_proteins, cfg["checkpoint_dir"], config=cfg)

                # Reset generation tracking
                self.train_genome_idx = 0
                self.gen_tiles = []
                self.gen_scores = []
                self.gen_best_tile = 0
                self.gen_best_score = 0
            else:
                self.population.next_genome()

            self.train_episode_started = False
            return

        # --- Play one step ---
        genome = self.train_current_genome
        inp = [float(v) for v in self.train_signals.values()]
        action = genome.controller.forward(inp)
        self.train_signals, self.train_done = self.env.step(action)
        genome.forward(self.train_signals)

        # Update rendering data
        self.current_board = self.env.get_board_copy()
        self.current_score = self.env.score
        self.current_max_tile = self.env.max_tile
        self.current_gen = self.generation
        self.current_genome_id = genome.id

    def _record_generation(self):
        """Record stats for completed generation into chart history."""
        if not self.gen_tiles:
            return
        self.history_gen.append(self.generation)
        self.history_avg_score.append(sum(self.gen_scores) / len(self.gen_scores))
        self.history_best_tile.append(max(self.gen_tiles))
        self.history_avg_tile.append(sum(self.gen_tiles) / len(self.gen_tiles))
        # Best trust from population
        if self.population and self.population.genomes:
            best_trust = max(g.trust for g in self.population.genomes)
            self.history_best_trust.append(best_trust)
        else:
            self.history_best_trust.append(0.0)
        # Generation time
        if self._gen_start_time is not None:
            self.history_gen_time.append(time.time() - self._gen_start_time)
        elif self.history_gen_time:
            self.history_gen_time.append(self.history_gen_time[-1])
        else:
            self.history_gen_time.append(0.0)
        # Tile distribution
        dist = {}
        for t in self.gen_tiles:
            dist[t] = dist.get(t, 0) + 1
        self.history_tile_dist = dist

    def _update_training_stats(self):
        """Build stats string for the training panel."""
        if not self.population:
            return

        genome = self.train_current_genome
        cfg = self.panel.get_config()
        gen_elapsed = time.time() - self._gen_start_time if self._gen_start_time else 0
        total_elapsed = time.time() - self._train_start_time if getattr(self, '_train_start_time', None) else 0

        lines = []
        lines.append(f"Generation: {self.generation} / {cfg['max_generations']}")
        pop_size = len(self.gen_tiles) if self.gen_tiles else (self.population.size if self.population else 0)
        lines.append(f"Population: {pop_size} genomes")
        lines.append(f"Games Played: {self.games_played:,}")

        # Timing
        if total_elapsed > 0:
            hrs = int(total_elapsed // 3600)
            mins = int((total_elapsed % 3600) // 60)
            secs = int(total_elapsed % 60)
            lines.append(f"Elapsed: {hrs}h {mins}m {secs}s")
        if gen_elapsed > 0:
            lines.append(f"Gen Speed: {gen_elapsed:.2f}s")
            if self.generation > 0:
                gens_per_min = 60.0 / gen_elapsed
                lines.append(f"Gens/min: {gens_per_min:.1f}")

        lines.append("")

        # --- Mutation & Evolution ---
        lines.append(f"--- Evolution ---")
        evo = getattr(self, '_gpu_evolver', None)
        eff_mut = getattr(evo, 'effective_mutation_rate', None) if evo else None
        if eff_mut is not None:
            lines.append(f"  Mutation Rate:  {eff_mut:.4f}  (adaptive)")
        else:
            lines.append(f"  Mutation Rate:  {cfg.get('child_mutation', 0.05)}")
        lines.append(f"  Survival:       top {cfg.get('survival_pct', 20)}%")
        lines.append(f"  Ratchet:        {cfg.get('ratchet_strength', 2.0)}")
        lines.append(f"  Proximity:      {cfg.get('proximity_strength', 1.0)}")

        # Trust
        bt, mt, lt = getattr(self, '_last_trust', (0, 0, 0))
        if bt or mt or lt:
            lines.append(f"  Trust Best:     {bt:.2f}")
            lines.append(f"  Trust Median:   {mt:.2f}")
            lines.append(f"  Trust Low:      {lt:.2f}")
            lines.append(f"  Trust Spread:   {bt - lt:.2f}")

        # Activation distribution (V3)
        if evo and hasattr(evo, 'act_ids'):
            act_names = ["tanh_sc", "gated", "soft_th", "reson",
                         "dual_p", "abs_gt", "quad_r", "id_plus"]
            act_counts = {}
            ids = evo.act_ids.cpu().tolist()
            for a in ids:
                act_counts[a] = act_counts.get(a, 0) + 1
            parts = []
            for aid, cnt in sorted(act_counts.items(), key=lambda x: -x[1]):
                name = act_names[aid] if aid < len(act_names) else f"#{aid}"
                parts.append(f"{name}:{cnt}")
            lines.append(f"  Activations:    {' '.join(parts[:4])}")
            if len(parts) > 4:
                lines.append(f"                  {' '.join(parts[4:])}")

        # V4/V5 reproductive traits
        if evo and hasattr(evo, 'genome_mut_rate'):
            mr = evo.genome_mut_rate
            ms = evo.genome_mut_scale
            me = evo.genome_explore
            lines.append(f"  Repro mut_rate:  {mr.mean():.4f} (std {mr.std():.4f})")
            lines.append(f"  Repro mut_scale: {ms.mean():.4f} (std {ms.std():.4f})")
            lines.append(f"  Repro explore:   {me.mean():.4f} (std {me.std():.4f})")
        if evo and hasattr(evo, 'genome_crossover'):
            mc = evo.genome_crossover
            lines.append(f"  Repro crossover: {mc.mean():.4f} (std {mc.std():.4f})")

        lines.append("")

        # --- Current game (non-GPU mode) ---
        if genome and not (HEADLESS_V2 or HEADLESS_V3):
            lines.append(f"--- Current Game ---")
            lines.append(f"  Genome ID:   {genome.id}")
            lines.append(f"  Trust:       {genome.trust:.2f}")
            lines.append(f"  Max Tile:    {self.env.max_tile if self.env else 0}")
            lines.append(f"  Score:       {self.env.score if self.env else 0}")
            lines.append(f"  Moves:       {self.env.moves_made if self.env else 0}")
            lines.append(f"  Energy:      {self.env.energy if self.env else 0}")
            lines.append(f"  Alive:       {self.env.alive if self.env else False}")
            lines.append("")

        # --- Generation Stats ---
        lines.append(f"--- Generation Stats ---")
        lines.append(f"  Best Tile:    {self.gen_best_tile}")
        lines.append(f"  Best Score:   {self.gen_best_score}")
        if self.gen_tiles:
            avg_tile = sum(self.gen_tiles) / len(self.gen_tiles)
            avg_score = sum(self.gen_scores) / len(self.gen_scores)
            sorted_t = sorted(self.gen_tiles)
            sorted_s = sorted(self.gen_scores)
            n = len(self.gen_tiles)
            lines.append(f"  Median Tile:  {sorted_t[n//2]}")
            lines.append(f"  Avg Tile:     {avg_tile:.0f}")
            lines.append(f"  Avg Score:    {avg_score:.0f}")
            lines.append(f"  Min Score:    {sorted_s[0]}")

            # Tile distribution
            dist = {}
            for t in self.gen_tiles:
                dist[t] = dist.get(t, 0) + 1
            dist_str = "  ".join(f"{t}:{c}" for t, c in sorted(dist.items(), reverse=True))
            lines.append(f"  Tiles: {dist_str}")

            # Thresholds
            for thresh in [128, 256, 512, 1024, 2048]:
                cnt = sum(1 for t in self.gen_tiles if t >= thresh)
                if cnt > 0:
                    lines.append(f"  >={thresh}: {cnt}/{n} ({cnt/n*100:.1f}%)")
        lines.append("")
        lines.append(f"--- All-Time Records ---")
        lines.append(f"  Best Tile Ever:    {self.best_tile_ever}")
        lines.append(f"  Best Score Ever:   {self.best_score_ever}")

        self.panel.update_stats("\n".join(lines))

    # --------------------------------------------------------
    # INFERENCE
    # --------------------------------------------------------
    def load_inference_checkpoint(self, path):
        pop, gen, template, _saved_config = load_checkpoint(path)
        self.inf_population = pop
        self.inf_generation = gen
        # Fix controller dimensions to match actual weight shapes (V3 encoder changes input_size)
        for g in pop.genomes:
            if g.controller.w1 is not None and len(g.controller.w1) > 0:
                g.controller.input_size = len(g.controller.w1[0])
                g.controller.hidden_size = len(g.controller.b1)
        # Sort by trust
        self.inf_population.genomes.sort(key=lambda g: g.trust, reverse=True)
        self.panel.update_genome_selector(len(pop.genomes), pop.genomes)
        self.select_inference_genome(0)

    def select_inference_genome(self, idx):
        if not self.inf_population:
            return
        idx = max(0, min(idx, len(self.inf_population.genomes) - 1))
        self.inf_genome = self.inf_population.genomes[idx]
        g = self.inf_genome
        self.panel.genome_info_label.config(text=f"ID:{g.id} Trust:{g.trust:.1f} Max:{g.max_tile}")
        self.restart_inference_game()

    def start_inference(self):
        if self.headless:
            self._run_headless_inference()
            return
        self.mode = "inference"
        self.inf_games = 0
        self.inf_best_tile = 0
        self.inf_total_score = 0
        self.restart_inference_game()

    def stop_inference(self):
        self.mode = "idle"

    def restart_inference_game(self):
        cfg = self.panel.get_config()
        self.inf_env = Game2048Env(target_tile=cfg["target_tile"])
        self.inf_env.max_energy = cfg["starting_energy"]
        self.inf_signals = self.inf_env.reset()
        self.inf_done = False

        # Reset protein states for fresh game
        if self.inf_genome:
            for p in self.inf_genome.proteins:
                for key in p.state:
                    if isinstance(p.state[key], (int, float)):
                        if key == "running_max": p.state[key] = 1.0
                        elif key == "count": p.state[key] = 0
                        else: p.state[key] = 0.0
                    elif isinstance(p.state[key], bool):
                        p.state[key] = False
                    elif p.state[key] is None:
                        p.state[key] = None
            self.inf_genome.trust = 0.0

    def _inference_step(self):
        """Advance inference by one game-step."""
        if not self.inf_genome or self.inf_done:
            if self.inf_done and self.inf_env:
                # Auto-restart after a short pause
                self.inf_games += 1
                self.inf_best_tile = max(self.inf_best_tile, self.inf_env.max_tile)
                self.inf_total_score += self.inf_env.score
                self.restart_inference_game()
            return

        inp = [float(v) for v in self.inf_signals.values()]
        # Route through encoder if genome has one (V3)
        enc = getattr(self.inf_genome, 'encoder', None)
        if enc is not None:
            inp = enc.forward(inp)
        action = self.inf_genome.controller.forward(inp)
        self.inf_signals, self.inf_done = self.inf_env.step(action)
        self.inf_genome.forward(self.inf_signals)

        # Update rendering data
        self.current_board = self.inf_env.get_board_copy()
        self.current_score = self.inf_env.score
        self.current_max_tile = self.inf_env.max_tile
        self.current_gen = self.inf_generation
        self.current_genome_id = self.inf_genome.id
        self.current_extra = f"Inference | Games: {self.inf_games} | Best: {self.inf_best_tile}"

    def _update_inference_stats(self):
        if not self.inf_genome or not self.inf_env:
            return

        g = self.inf_genome
        env = self.inf_env
        lines = []
        lines.append(f"Genome ID:     {g.id}")
        lines.append(f"Trust:         {g.trust:.2f}")
        lines.append(f"Generation:    {self.inf_generation}")
        lines.append("")
        lines.append(f"--- Current Game ---")
        lines.append(f"  Max Tile:    {env.max_tile}")
        lines.append(f"  Score:       {env.score}")
        lines.append(f"  Moves:       {env.moves_made}")
        lines.append(f"  Energy:      {env.energy}")
        lines.append(f"  Empty Cells: {env._count_empty()}")
        lines.append(f"  Alive:       {env.alive}")
        if env.won:
            lines.append(f"  *** REACHED {env.target_tile}! ***")
        lines.append("")
        lines.append(f"--- Session Stats ---")
        lines.append(f"  Games Played:  {self.inf_games}")
        lines.append(f"  Best Tile:     {self.inf_best_tile}")
        if self.inf_games > 0:
            lines.append(f"  Avg Score:     {self.inf_total_score / self.inf_games:.0f}")

        self.panel.update_inference_stats("\n".join(lines))

    # --------------------------------------------------------
    # HEADLESS INFERENCE (opens pygame for one game, then idles)
    # --------------------------------------------------------
    def _run_headless_inference(self):
        """
        Open a pygame window, play one full inference game with rendering,
        then keep the final board on screen until the user closes the window.
        After close, return to headless mode — no crash, no exit.
        """
        if not self.inf_genome:
            return

        # Stop training while we show inference
        was_training = (self.mode == "training")
        prev_mode = self.mode
        self.mode = "inference"

        # Prepare the game
        self.restart_inference_game()

        # Open pygame
        import pygame as _pg
        _pg.init()
        screen = _pg.display.set_mode((WIN_W, WIN_H))
        _pg.display.set_caption("GENREG 2048 — Inference")
        clock = _pg.time.Clock()
        font_big = _pg.font.SysFont("Arial", 36, bold=True)
        font_med = _pg.font.SysFont("Arial", 24, bold=True)
        font_sm = _pg.font.SysFont("Arial", 16, bold=True)
        font_hdr = _pg.font.SysFont("Arial", 18)

        fps = self.panel.inf_fps_var.get() or 10

        def _draw(board, score, max_tile, extra=""):
            screen.fill(HEADER_BG)
            # Header
            screen.blit(font_hdr.render(f"Score: {score}", True, (119,110,101)), (CELL_PAD, 8))
            screen.blit(font_hdr.render(f"Max: {max_tile}", True, (119,110,101)), (CELL_PAD+150, 8))
            screen.blit(font_hdr.render(
                f"Gen: {self.inf_generation}  ID: {self.inf_genome.id}", True, (119,110,101)),
                (CELL_PAD, 32))
            if extra:
                screen.blit(font_sm.render(extra, True, (180,120,60)), (CELL_PAD, 54))
            # Board
            board_rect = _pg.Rect(0, HEADER_H, BOARD_PX, BOARD_PX)
            _pg.draw.rect(screen, BG_COLOR, board_rect, border_radius=6)
            for r in range(4):
                for c in range(4):
                    x = CELL_PAD + c * (CELL_SIZE + CELL_PAD)
                    y = HEADER_H + CELL_PAD + r * (CELL_SIZE + CELL_PAD)
                    val = board[r][c]
                    color = TILE_COLORS.get(val, (60, 58, 50))
                    _pg.draw.rect(screen, color, _pg.Rect(x, y, CELL_SIZE, CELL_SIZE), border_radius=4)
                    if val > 0:
                        tc = TILE_TEXT_DARK if val <= 4 else TILE_TEXT_LIGHT
                        font = font_med if val >= 100 else font_big
                        txt = font.render(str(val), True, tc)
                        screen.blit(txt, (x + (CELL_SIZE - txt.get_width())//2,
                                          y + (CELL_SIZE - txt.get_height())//2))
            _pg.display.flip()

        # --- Play one game ---
        game_over = False
        while not game_over:
            for ev in _pg.event.get():
                if ev.type == _pg.QUIT:
                    # User closed window mid-game — abort
                    _pg.quit()
                    self.mode = prev_mode
                    self.panel.btn_play.config(state="normal")
                    self.panel.btn_stop_inf.config(state="disabled")
                    self._update_inference_stats()
                    return

            # Step
            inp = [float(v) for v in self.inf_signals.values()]
            # Route through encoder if genome has one (V3)
            enc = getattr(self.inf_genome, 'encoder', None)
            if enc is not None:
                inp = enc.forward(inp)
            action = self.inf_genome.controller.forward(inp)
            self.inf_signals, self.inf_done = self.inf_env.step(action)
            self.inf_genome.forward(self.inf_signals)

            board = self.inf_env.get_board_copy()
            extra = f"Inference | Score: {self.inf_env.score} | Moves: {self.inf_env.moves_made}"
            _draw(board, self.inf_env.score, self.inf_env.max_tile, extra)
            clock.tick(fps)

            if self.inf_done:
                game_over = True

        # --- Game over: show final state until window closed ---
        won_text = f"REACHED {self.inf_env.target_tile}!" if self.inf_env.won else ""
        final_extra = (f"GAME OVER | Score: {self.inf_env.score} | "
                       f"Max Tile: {self.inf_env.max_tile} | "
                       f"Moves: {self.inf_env.moves_made}")
        if won_text:
            final_extra += f" | {won_text}"
        final_extra += " | Close window to return"

        _draw(self.inf_env.get_board_copy(), self.inf_env.score,
              self.inf_env.max_tile, final_extra)

        # Idle loop — just pump events until user closes
        waiting = True
        while waiting:
            for ev in _pg.event.get():
                if ev.type == _pg.QUIT:
                    waiting = False
            clock.tick(15)
            # Keep tkinter alive so the control panel doesn't freeze
            try:
                self.panel.root.update_idletasks()
                self.panel.root.update()
            except Exception:
                waiting = False

        _pg.quit()

        # Update stats and restore mode
        self.inf_games += 1
        self.inf_best_tile = max(self.inf_best_tile, self.inf_env.max_tile)
        self.inf_total_score += self.inf_env.score
        self._update_inference_stats()

        self.mode = prev_mode
        self.panel.btn_play.config(state="normal")
        self.panel.btn_stop_inf.config(state="disabled")

    # --------------------------------------------------------
    # MAIN LOOP
    # --------------------------------------------------------
    def run(self):
        if self.headless:
            self._run_headless()
        else:
            self._run_gui()

    def _run_headless(self):
        """Headless loop: tkinter only, full generations per tick."""
        chart_counter = 0
        while self.running:
            if self.mode == "training":
                if HEADLESS_V2 or HEADLESS_V3:
                    self._training_step_gpu()
                else:
                    self._training_step_headless()
                self._update_training_stats()
                chart_counter += 1
                if chart_counter % 5 == 0:
                    self.panel.update_charts(self)
            elif self.mode == "inference":
                self._inference_step()
                self._update_inference_stats()

            self.panel.update()

        # Cleanup
        if self.mode == "training" and self.population and self.template_proteins:
            if (HEADLESS_V2 or HEADLESS_V3) and hasattr(self, '_gpu_evolver'):
                ck_tiles, ck_scores, ck_trust, _ = self._gpu_evolver.run_generation(n_games=1)
                self._gpu_evolver.sync_to_cpu(self.population, tiles=ck_tiles, scores=ck_scores, trust=ck_trust)
            cfg = self.panel.get_config()
            save_checkpoint(self.population, self.generation, self.template_proteins, cfg["checkpoint_dir"], config=cfg)

    def _run_gui(self):
        """Full GUI loop: pygame + tkinter, one step per frame."""
        frame_count = 0

        while self.running:
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            # Determine FPS
            if self.mode == "training":
                fps = self.panel.train_fps_var.get()
            elif self.mode == "inference":
                fps = self.panel.inf_fps_var.get()
            else:
                fps = DEFAULT_FPS

            # Game logic
            if self.mode == "training":
                self._training_step()
                self.current_extra = f"Training | Gen {self.generation} | Genome {self.train_genome_idx+1}/{self.population.size if self.population else 0}"
            elif self.mode == "inference":
                self._inference_step()

            # Render
            self.renderer.draw(
                self.current_board,
                self.current_score,
                self.current_max_tile,
                self.current_gen,
                self.current_genome_id,
                fps,
                self.current_extra,
            )

            # Update tkinter (stats every 5 frames for performance)
            if frame_count % 5 == 0:
                if self.mode == "training":
                    self._update_training_stats()
                elif self.mode == "inference":
                    self._update_inference_stats()

            # Update charts less frequently (~every 2 seconds at 60fps)
            if frame_count % 120 == 0 and self.mode == "training":
                self.panel.update_charts(self)

            self.panel.update()
            self.renderer.tick(fps)
            frame_count += 1

        # Cleanup
        if self.mode == "training" and self.population and self.template_proteins:
            cfg = self.panel.get_config()
            save_checkpoint(self.population, self.generation, self.template_proteins, cfg["checkpoint_dir"], config=cfg)
        self.renderer.quit()


# ================================================================
# ENTRY POINT
# ================================================================
if __name__ == "__main__":
    if HEADLESS_V5:
        from genreg_encoder_gpu import NUM_ACTIVATIONS
        print("=" * 60)
        print("  GENREG 2048 — Headless V5 (Evolved Reproduction + Crossover)")
        print(f"  V4 + neuron-level crossover between elite parents")
        print(f"  Crossover probability is itself an evolvable trait")
        print(f"  Population decides: clone, mutate, or breed")
        print("=" * 60)
    elif HEADLESS_V4:
        from genreg_encoder_gpu import NUM_ACTIVATIONS
        print("=" * 60)
        print("  GENREG 2048 — Headless V4 (Evolved Reproduction)")
        print(f"  V3 encoder + evolved per-genome reproductive strategy")
        print(f"  Each genome evolves: mut_rate, mut_scale, explore_drive")
        print(f"  Population self-organizes r/K selection dynamics")
        print("=" * 60)
    elif HEADLESS_V3:
        from genreg_encoder_gpu import NUM_ACTIVATIONS
        print("=" * 60)
        print("  GENREG 2048 — Headless V3 (Evolved Encoder)")
        print(f"  Perception layer: 22 raw → encoder → evolved activation → controller")
        print(f"  Activation catalog: {NUM_ACTIVATIONS} functions, evolved per-genome")
        print("  Each genome sees the board through a different mathematical lens")
        print("=" * 60)
    elif HEADLESS_V2:
        print("=" * 60)
        print("  GENREG 2048 — Headless V2 (GPU Multi-Game Ensemble)")
        print("  Each genome plays N games per generation (default: 3)")
        print("  Trust averaged across games → reduces board luck")
        print("=" * 60)
    app = GENREGApp(headless=HEADLESS)
    app.run()
