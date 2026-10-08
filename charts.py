#!/usr/bin/env -S uv run
# /// script
# requires-python = "==3.12.*"
# dependencies = ["matplotlib"]
# ///
"""Render benchmark scores by reasoning effort as one interactive HTML page: HTML, CSS and inline SVG, no JavaScript."""

import csv
import html
import io
import itertools
import json
import math
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from string import Template

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.backend_bases import RendererBase
from matplotlib.lines import Line2D
from matplotlib.path import Path as MatplotlibPath
from matplotlib.text import Annotation, Text
from matplotlib.transforms import Bbox

OUTPUT_DIRECTORY = Path(__file__).parent / "out"


@dataclass(frozen=True)
class Color:
    """A themeable color: the SVG is drawn in the light value, then each light value becomes var(--variable)."""
    variable: str
    light: str
    dark: str


SURFACE = Color("surface", "#fcfcfb", "#1a1a19")
PAGE = Color("page", "#f9f9f7", "#0d0d0d")
TEXT_PRIMARY = Color("ink-1", "#0b0b0b", "#ffffff")
TEXT_SECONDARY = Color("ink-2", "#52514e", "#c3c2b7")
TEXT_MUTED = Color("ink-3", "#898781", "#898781")
GRIDLINE = Color("grid", "#e1e0d9", "#2c2c2a")
BASELINE = Color("baseline", "#c3c2b7", "#383835")
BORDER = Color("border", "rgba(11,11,11,0.10)", "rgba(255,255,255,0.10)")

EFFORT_LEVELS = ["off", "low", "medium", "high", "xhigh", "max"]


def slugify(text: str) -> str:
    """
    >>> slugify("GPT-5.6 Sol")
    'gpt-5-6-sol'
    """
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


@dataclass(frozen=True)
class Model:
    name: str
    color: Color

    @property
    def slug(self) -> str:
        return slugify(self.name)


GPT_5_6_SOL = Model("GPT-5.6 Sol", Color("series-1", "#2a78d6", "#3987e5"))
GPT_6_ASTRA = Model("GPT-6 Astra", Color("series-2", "#eb6834", "#d95926"))
GPT_6_SOL = Model("GPT-6 Sol", Color("series-3", "#1baf7a", "#199e70"))
CLAUDE_FABLE_5_1 = Model("Fable 5.1", Color("series-4", "#eda100", "#c98500"))
CLAUDE_OPUS_5_5 = Model("Opus 5.5", Color("series-5", "#e87ba4", "#d55181"))
GPT_6_1_SOL = Model("GPT-6.1 Sol", Color("series-6", "#008300", "#008300"))
CLAUDE_SONNET_5_5 = Model("Sonnet 5.5", Color("series-7", "#4a3aa7", "#9085e9"))
MODELS = [GPT_5_6_SOL, GPT_6_ASTRA, GPT_6_SOL, GPT_6_1_SOL, CLAUDE_FABLE_5_1, CLAUDE_OPUS_5_5, CLAUDE_SONNET_5_5]
COLORS = [SURFACE, PAGE, TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED, GRIDLINE, BASELINE, BORDER,
          *(model.color for model in MODELS)]

Scores = list[float | None]


@dataclass(frozen=True)
class Metric:
    name: str
    unit_and_source: str
    scores: dict[Model, Scores]
    y_range: tuple[float, float]
    tick_step: float
    decimals: int = 1
    estimated: frozenset[tuple[Model, str]] = frozenset()

    @property
    def models(self) -> list[Model]:
        return [model for model in MODELS if model in self.scores]

    def label(self, score: float) -> str:
        return f"{score:.{self.decimals}f}"

    @property
    def costs(self) -> dict[Model, Scores] | None:
        return COSTS.get(self.name)

    @property
    def cost_unit(self) -> str:
        return COST_UNITS.get(self.name, DEFAULT_COST_UNIT)


INTELLIGENCE = [
    Metric("AA Intelligence Index", "v4.3.2 · Artificial Analysis", {
        GPT_5_6_SOL: [28.3, 33.5, 39.2, 42.3, 44.0, 47.0],
        GPT_6_ASTRA: [None, 45.8, 49.6, 50.9, 52.4, 52.7],
        GPT_6_SOL: [28.5, 34.2, 39.8, 42.4, 44.2, 47.6],
        GPT_6_1_SOL: [None, 42.1, 47.8, 50.2, 51.0, 51.8],
        CLAUDE_FABLE_5_1: [None, 46.8, 48.9, 51.2, 53.2, 53.4],
        CLAUDE_OPUS_5_5: [None, 42.3, 51.2, 53.6, 56.0, 57.6],
        CLAUDE_SONNET_5_5: [None, 35.9, 40.8, 46.8, 51.9, 56.0],
    }, (20, 60), 10, estimated=frozenset({(GPT_5_6_SOL, "off")})),
    Metric("Humanity's Last Exam", "% · Artificial Analysis", {
        GPT_5_6_SOL: [16.7, 39.4, 42.2, 46.0, 47.3, 49.5],
        GPT_6_ASTRA: [None, 49.2, 52.7, 53.1, 54.6, 54.7],
        GPT_6_SOL: [18.4, 34.9, 41.0, 44.1, 46.3, 47.9],
        GPT_6_1_SOL: [None, 47.4, 49.9, 51.4, 52.6, 52.9],
        CLAUDE_FABLE_5_1: [None, 48.9, 53.8, 55.9, 58.7, 59.1],
        CLAUDE_OPUS_5_5: [None, 48.3, 54.7, 55.6, 57.5, 61.4],
        CLAUDE_SONNET_5_5: [None, 36.2, 39.8, 45.8, 50.0, 55.0],
    }, (10, 70), 10),
    Metric("CritPt", "% · Artificial Analysis", {
        GPT_5_6_SOL: [5.1, 14.9, 22.9, 25.7, 28.6, 32.3],
        GPT_6_ASTRA: [None, 26.3, 29.1, 28.9, 31.4, 31.7],
        GPT_6_SOL: [4.0, 16.3, 24.6, 25.4, 28.0, 30.9],
        GPT_6_1_SOL: [None, 24.9, 27.7, 30.0, 31.7, 31.7],
        CLAUDE_FABLE_5_1: [None, 27.7, 29.1, 30.3, 31.1, 29.7],
        CLAUDE_OPUS_5_5: [None, 17.7, 27.7, 30.9, 31.7, 31.7],
        CLAUDE_SONNET_5_5: [None, 11.4, 16.9, 24.6, 31.1, 31.4],
    }, (0, 40), 10),
    Metric("ARC-AGI-2", "% semi-private · ARC Prize", {
        GPT_5_6_SOL: [None, 42.5, 67.1, 85.4, 90.0, 92.5],
        GPT_6_ASTRA: [59.6, 85.4, 92.1, 92.1, 93.3, 95.0],
        GPT_6_SOL: [1.7, 31.5, 57.8, 68.9, 78.1, 89.6],
        GPT_6_1_SOL: [None, 76.7, 86.7, 91.7, 91.7, 94.2],
        CLAUDE_FABLE_5_1: [None, 78.3, 86.2, 88.8, 90.0, 90.0],
        CLAUDE_OPUS_5_5: [None, 70.1, 87.5, 93.3, 92.5, 91.7],
    }, (0, 100), 20),
    Metric("AA-Omniscience Index", "−100 to 100 · Artificial Analysis", {
        GPT_5_6_SOL: [1.1, 18.9, 19.4, 20.4, 21.0, 22.0],
        GPT_6_ASTRA: [None, 40.5, 42.2, 43.7, 43.4, 43.4],
        GPT_6_SOL: [-0.8, 26.5, 27.0, 26.8, 26.7, 27.1],
        GPT_6_1_SOL: [None, 37.6, 40.0, 41.5, 40.9, 41.5],
        CLAUDE_FABLE_5_1: [None, 34.1, 37.6, 40.8, 42.4, 43.5],
        CLAUDE_OPUS_5_5: [None, 38.9, 40.3, 40.6, 42.6, 46.4],
        CLAUDE_SONNET_5_5: [None, 19.4, 20.1, 20.9, 23.5, 32.3],
    }, (-10, 50), 10),
    Metric("GDPval-AA", "Elo · Artificial Analysis", {
        GPT_5_6_SOL: [1242, 1305, 1422, 1505, 1572, 1611],
        GPT_6_ASTRA: [None, 1366, 1468, 1485, 1516, 1542],
        GPT_6_SOL: [1252, 1204, 1350, 1396, 1457, 1510],
        GPT_6_1_SOL: [None, 1297, 1433, 1486, 1510, 1575],
        CLAUDE_FABLE_5_1: [None, 1469, 1549, 1635, 1734, 1758],
        CLAUDE_OPUS_5_5: [None, 1235, 1586, 1707, 1837, 1866],
        CLAUDE_SONNET_5_5: [None, 1179, 1324, 1551, 1731, 1839],
    }, (1100, 1900), 200, decimals=0),
    Metric("AA-Briefcase", "Elo · long-horizon knowledge work · Artificial Analysis", {
        GPT_5_6_SOL: [1009, 1043, 1242, 1368, 1438, 1478],
        GPT_6_ASTRA: [None, 1261, 1459, 1507, 1544, 1569],
        GPT_6_SOL: [1104, 886, 1150, 1268, 1359, 1479],
        GPT_6_1_SOL: [None, 1118, 1365, 1471, 1507, 1564],
        CLAUDE_FABLE_5_1: [None, 1482, 1529, 1581, 1656, 1675],
        CLAUDE_OPUS_5_5: [None, 1280, 1628, 1689, 1768, 1807],
        CLAUDE_SONNET_5_5: [None, 1272, 1442, 1639, 1751, 1823],
    }, (800, 1900), 200, decimals=0),
    Metric("AutomationBench-AA", "% · Artificial Analysis", {
        GPT_5_6_SOL: [22.7, 41.0, 51.3, 55.3, 55.3, 60.1],
        GPT_6_ASTRA: [None, 59.1, 64.6, 66.6, 67.2, 68.5],
        GPT_6_SOL: [34.2, 53.9, 58.0, 60.1, 61.7, 61.6],
        GPT_6_1_SOL: [None, 52.6, 62.6, 64.5, 66.6, 64.9],
        CLAUDE_FABLE_5_1: [None, 52.2, 54.7, 55.3, 57.8, 59.4],
        CLAUDE_OPUS_5_5: [None, 52.9, 61.2, 63.2, 65.0, 69.5],
        CLAUDE_SONNET_5_5: [None, 49.4, 54.9, 59.4, 65.5, 71.8],
    }, (20, 80), 20),
    Metric("AA-LCR", "% long-context reasoning · Artificial Analysis", {
        GPT_5_6_SOL: [62.3, 78.0, 80.3, 81.7, 82.3, 84.0],
        GPT_6_ASTRA: [None, 80.0, 79.7, 80.0, 80.0, 80.7],
        GPT_6_SOL: [64.0, 79.3, 82.3, 83.7, 81.3, 83.7],
        GPT_6_1_SOL: [None, 84.0, 83.3, 82.3, 79.7, 83.0],
        CLAUDE_FABLE_5_1: [None, 82.3, 84.7, 83.7, 83.0, 85.3],
        CLAUDE_OPUS_5_5: [None, 80.7, 84.3, 82.7, 84.7, 84.7],
        CLAUDE_SONNET_5_5: [None, 76.0, 76.3, 78.0, 79.7, 82.7],
    }, (60, 90), 10),
    Metric("GDP.pdf", "% all-pass · Q&A over complex PDFs · Artificial Analysis", {
        GPT_5_6_SOL: [15.2, 21.0, 26.2, 27.8, 27.6, 27.2],
        GPT_6_ASTRA: [None, 30.4, 30.4, 31.0, 32.2, 31.0],
        GPT_6_SOL: [17.0, 24.2, 23.8, 24.4, 24.6, 25.2],
        GPT_6_1_SOL: [None, 27.0, 30.0, 32.0, 31.8, 31.0],
        CLAUDE_FABLE_5_1: [None, 28.0, 26.8, 26.8, 26.2, 26.2],
        CLAUDE_OPUS_5_5: [None, 25.6, 25.6, 28.8, 26.6, 26.2],
        CLAUDE_SONNET_5_5: [None, 16.0, 20.2, 25.2, 24.6, 25.8],
    }, (10, 35), 5),
]

CODING = [
    Metric("AA Coding Agent Index", "v1.4 · reported by OpenAI", {
        GPT_5_6_SOL: [43.4, 55.2, 61.6, 64.1, 63.3, 65.1],
        GPT_6_ASTRA: [None, 62.6, 65.3, 65.5, 67.0, 67.0],
    }, (40, 70), 10),
    Metric("AA Index coding category", "mean of TB 4.0 & SciCode · derived from AA", {
        GPT_5_6_SOL: [None, 28.7, 36.0, 39.2, 40.9, 48.5],
        GPT_6_ASTRA: [None, 48.0, 51.9, 54.7, 57.7, 57.8],
        GPT_6_SOL: [30.2, 29.7, 36.2, 40.6, 42.7, 50.8],
        GPT_6_1_SOL: [None, 42.0, 50.6, 53.6, 54.9, 55.2],
        CLAUDE_FABLE_5_1: [None, 48.5, 50.6, 55.4, 58.0, 57.5],
        CLAUDE_OPUS_5_5: [None, 45.0, 55.9, 58.5, 62.3, 63.2],
        CLAUDE_SONNET_5_5: [None, 34.9, 41.4, 48.8, 57.2, 62.3],
    }, (20, 70), 10),
    Metric("Terminal-Bench 4.0", "% · Artificial Analysis", {
        GPT_5_6_SOL: [None, 1.0, 14.6, 20.7, 24.7, 39.9],
        GPT_6_ASTRA: [None, 41.9, 49.5, 54.0, 59.6, 59.1],
        GPT_6_SOL: [13.1, 9.1, 18.7, 26.3, 30.3, 43.9],
        GPT_6_1_SOL: [None, 30.8, 48.0, 51.5, 54.0, 56.1],
        CLAUDE_FABLE_5_1: [None, 40.4, 44.9, 52.0, 55.1, 52.0],
        CLAUDE_OPUS_5_5: [None, 31.3, 52.5, 56.6, 59.6, 59.6],
        CLAUDE_SONNET_5_5: [None, 20.7, 29.8, 43.9, 57.1, 63.6],
    }, (0, 70), 10),
    Metric("Terminal-Bench 2.1", "% · Artificial Analysis", {
        GPT_5_6_SOL: [74.2, 76.8, 86.1, 87.3, 89.5, 88.0],
        GPT_6_ASTRA: [None, 88.0, 89.5, 89.9, 89.1, 88.4],
        CLAUDE_FABLE_5_1: [None, 85.0, 88.0, 89.9, 91.0, 91.4],
    }, (70, 95), 5),
    Metric("DeepSWE v1.1", "% · reported by OpenAI", {
        GPT_5_6_SOL: [None, 45.4, 61.1, 69.4, 70.7, 72.7],
        GPT_6_ASTRA: [None, 67.0, 72.8, 73.2, 74.1, 73.2],
        GPT_6_SOL: [None, 37.2, 56.6, 65.3, 66.6, 68.8],
        GPT_6_1_SOL: [None, 64.4, 73.0, 75.2, 71.9, 71.9],
    }, (30, 80), 10),
    Metric("FrontierCode 1.1 Main", "score · Cognition leaderboard", {
        GPT_5_6_SOL: [None, 35.4, 39.9, 45.1, 46.8, 47.5],
        GPT_6_ASTRA: [None, 45.3, 48.8, 50.9, 50.6, 53.3],
        GPT_6_SOL: [None, 37.3, 45.9, 47.7, 48.4, 49.3],
        GPT_6_1_SOL: [None, 45.5, 50.2, 48.0, 49.3, 47.6],
        CLAUDE_FABLE_5_1: [None, 49.8, 50.9, 50.3, 48.7, 50.3],
        CLAUDE_OPUS_5_5: [None, 47.3, 54.6, 54.0, 51.4, 54.4],
        CLAUDE_SONNET_5_5: [None, 29.3, 36.5, 49.4, 52.1, 46.2],
    }, (20, 60), 10),
    Metric("CursorBench 4.0", "% · Cursor leaderboard", {
        GPT_5_6_SOL: [None, 24.6, 31.1, 35.7, 37.7, 41.7],
        CLAUDE_FABLE_5_1: [None, 45.1, 46.8, 49.2, 51.6, 51.8],
        CLAUDE_OPUS_5_5: [None, 43.7, 52.5, 56.0, 56.0, 57.8],
        CLAUDE_SONNET_5_5: [None, 35.8, 39.2, 47.8, 53.1, 55.5],
    }, (20, 60), 10),
    Metric("SciCode", "% · Artificial Analysis", {
        GPT_5_6_SOL: [47.7, 56.4, 57.4, 57.8, 57.1, 57.1],
        GPT_6_ASTRA: [None, 54.1, 54.2, 55.4, 55.7, 56.5],
        GPT_6_SOL: [47.3, 50.2, 53.8, 54.9, 55.1, 57.6],
        GPT_6_1_SOL: [None, 53.2, 53.2, 55.8, 55.7, 54.2],
        CLAUDE_FABLE_5_1: [None, 56.7, 56.4, 58.7, 60.9, 63.1],
        CLAUDE_OPUS_5_5: [None, 58.6, 59.3, 60.4, 65.0, 66.9],
        CLAUDE_SONNET_5_5: [None, 49.1, 52.9, 53.7, 57.3, 61.0],
    }, (45, 70), 5),
]

SECTIONS = [("Intelligence", INTELLIGENCE), ("Coding", CODING)]


@dataclass(frozen=True)
class ReportedMetric:
    """A benchmark with one reported score per model, at whatever effort its reporter ran; drawn as bars."""
    name: str
    unit_and_source: str
    scores: dict[Model, tuple[float, str]]  # score, and the effort level the reporter ran it at

    @property
    def models(self) -> list[Model]:
        return [model for model in MODELS if model in self.scores]

    @property
    def missing(self) -> list[Model]:
        return [model for model in MODELS if model not in self.scores]


# Collected by a research pass into single_value_benchmarks.json, with sources there.
REPORTED = {"Coding": [
    ReportedMetric("SWE-Bench Pro", "% resolved · each lab’s own run on its own agent harness", {
        GPT_5_6_SOL: (64.6, "effort not stated"),
        CLAUDE_FABLE_5_1: (81.2, "max"),
        CLAUDE_OPUS_5_5: (89.9, "max"),
        CLAUDE_SONNET_5_5: (81.3, "max"),
    }),
    ReportedMetric("LiveCodeBench", "% · v6 · Vals AI", {
        GPT_5_6_SOL: (82.6, "max"),
        CLAUDE_FABLE_5_1: (90.5, "max"),
    }),
]}

# US dollars per task of each benchmark at each effort level, from the same AA records (ARC Prize for ARC-AGI-2).
# Generated by build_costs.py. The coding category averages its two benchmarks' costs.
COSTS: dict[str, dict[Model, Scores]] = {
    'AA Intelligence Index': {
        GPT_5_6_SOL: [None, 637.0, 997.0, 1490.0, 2080.0, 3460.0],
        GPT_6_ASTRA: [None, 1540.0, 2430.0, 2930.0, 3800.0, 5320.0],
        GPT_6_SOL: [448.0, 268.0, 417.0, 610.0, 865.0, 1550.0],
        GPT_6_1_SOL: [None, 250.0, 361.0, 521.0, 662.0, 1080.0],
        CLAUDE_FABLE_5_1: [None, 3160.0, 3980.0, 5240.0, 9060.0, 13100.0],
        CLAUDE_OPUS_5_5: [None, 860.0, 1630.0, 2170.0, 4060.0, 8710.0],
        CLAUDE_SONNET_5_5: [None, 482.0, 622.0, 1030.0, 2180.0, 7260.0],
    },
    "Humanity's Last Exam": {
        GPT_5_6_SOL: [None, 0.0224, 0.043, 0.0828, 0.142, 0.268],
        GPT_6_ASTRA: [None, 0.0389, 0.101, 0.162, 0.253, 0.378],
        GPT_6_SOL: [0.0021, 0.00698, 0.0175, 0.0331, 0.0591, 0.12],
        GPT_6_1_SOL: [None, 0.00731, 0.0126, 0.0266, 0.0428, 0.0776],
        CLAUDE_FABLE_5_1: [None, 0.124, 0.225, 0.397, 1.03, 1.59],
        CLAUDE_OPUS_5_5: [None, 0.0286, 0.0619, 0.104, 0.241, 0.718],
        CLAUDE_SONNET_5_5: [None, 0.0128, 0.0232, 0.0473, 0.118, 0.544],
    },
    'CritPt': {
        GPT_5_6_SOL: [None, 0.0751, 0.126, 0.245, 0.437, 0.859],
        GPT_6_ASTRA: [None, 0.145, 0.285, 0.434, 0.798, 1.17],
        GPT_6_SOL: [0.0134, 0.0219, 0.0461, 0.0866, 0.154, 0.324],
        GPT_6_1_SOL: [None, 0.0272, 0.0395, 0.0757, 0.119, 0.242],
        CLAUDE_FABLE_5_1: [None, 1.26, 1.8, 2.75, 4.53, 5.71],
        CLAUDE_OPUS_5_5: [None, 0.124, 0.345, 0.53, 1.17, 2.19],
        CLAUDE_SONNET_5_5: [None, 0.0862, 0.149, 0.281, 0.65, 1.89],
    },
    'AA-Omniscience Index': {
        GPT_5_6_SOL: [None, 0.00589, 0.0102, 0.018, 0.0331, 0.0854],
        GPT_6_ASTRA: [None, 0.00696, 0.015, 0.0272, 0.0502, 0.097],
        GPT_6_SOL: [0.000361, 0.00156, 0.00335, 0.00635, 0.0117, 0.0269],
        GPT_6_1_SOL: [None, 0.00145, 0.00222, 0.00458, 0.00822, 0.0167],
        CLAUDE_FABLE_5_1: [None, 0.00864, 0.0153, 0.0207, 0.0773, 0.268],
        CLAUDE_OPUS_5_5: [None, 0.00577, 0.00755, 0.00853, 0.0135, 0.161],
        CLAUDE_SONNET_5_5: [None, 0.00274, 0.00294, 0.00536, 0.00849, 0.147],
    },
    'GDPval-AA': {
        GPT_5_6_SOL: [None, 0.268, 0.595, 1.11, 1.7, 2.81],
        GPT_6_ASTRA: [None, 0.855, 1.82, 2.43, 3.04, 4.53],
        GPT_6_SOL: [0.259, 0.0869, 0.238, 0.483, 0.767, 1.32],
        GPT_6_1_SOL: [None, 0.105, 0.231, 0.428, 0.558, 0.894],
        CLAUDE_FABLE_5_1: [None, 1.43, 2.18, 3.48, 7.22, 9.77],
        CLAUDE_OPUS_5_5: [None, 0.214, 0.856, 1.54, 4.21, 8.92],
        CLAUDE_SONNET_5_5: [None, 0.209, 0.318, 0.688, 1.95, 6.84],
    },
    'AutomationBench-AA': {
        GPT_5_6_SOL: [None, 0.47, 0.601, 0.741, 0.811, 0.972],
        GPT_6_ASTRA: [None, 0.999, 1.18, 1.3, 1.41, 1.6],
        GPT_6_SOL: [0.256, 0.191, 0.218, 0.254, 0.292, 0.358],
        GPT_6_1_SOL: [None, 0.159, 0.196, 0.226, 0.251, 0.297],
        CLAUDE_FABLE_5_1: [None, 1.52, 1.64, 1.77, 2.2, 2.61],
        CLAUDE_OPUS_5_5: [None, 0.493, 0.638, 0.699, 0.878, 1.43],
        CLAUDE_SONNET_5_5: [None, 0.241, 0.262, 0.313, 0.422, 1.03],
    },
    'AA-LCR': {
        GPT_5_6_SOL: [None, 0.382, 0.383, 0.386, 0.388, 0.402],
        GPT_6_ASTRA: [None, 0.95, 0.953, 0.96, 0.97, 0.997],
        GPT_6_SOL: [0.189, 0.19, 0.191, 0.191, 0.193, 0.198],
        GPT_6_1_SOL: [None, 0.19, 0.19, 0.191, 0.193, 0.199],
        CLAUDE_FABLE_5_1: [None, 1.47, 1.48, 1.48, 1.51, 1.58],
        CLAUDE_OPUS_5_5: [None, 0.587, 0.59, 0.592, 0.599, 0.677],
        CLAUDE_SONNET_5_5: [None, 0.291, 0.292, 0.294, 0.297, 0.339],
    },
    'Terminal-Bench 4.0': {
        GPT_5_6_SOL: [None, 0.524, 1.64, 2.22, 3.46, 8.09],
        GPT_6_ASTRA: [None, 2.25, 4.43, 4.05, 5.86, 8.5],
        GPT_6_SOL: [1.94, 0.489, 1.12, 1.6, 1.91, 4.0],
        GPT_6_1_SOL: [None, 0.383, 0.614, 0.828, 1.03, 1.82],
        CLAUDE_FABLE_5_1: [None, 7.32, 9.12, 11.6, 15.8, 19.2],
        CLAUDE_OPUS_5_5: [None, 2.08, 4.04, 5.12, 8.78, 13.1],
        CLAUDE_SONNET_5_5: [None, 1.28, 1.58, 2.19, 6.08, 12.6],
    },
    'SciCode': {
        GPT_5_6_SOL: [None, 0.0241, 0.0285, 0.0353, 0.0449, 0.0797],
        GPT_6_ASTRA: [None, 0.0425, 0.0509, 0.0696, 0.122, 0.232],
        GPT_6_SOL: [0.00694, 0.00743, 0.00949, 0.013, 0.0215, 0.0425],
        GPT_6_1_SOL: [None, 0.00842, 0.00931, 0.013, 0.022, 0.0434],
        CLAUDE_FABLE_5_1: [None, 0.069, 0.0756, 0.0891, 0.241, 0.659],
        CLAUDE_OPUS_5_5: [None, 0.0264, 0.0355, 0.0402, 0.0676, 0.469],
        CLAUDE_SONNET_5_5: [None, 0.0129, 0.0135, 0.016, 0.0237, 0.278],
    },
    'AA Index coding category': {
        GPT_5_6_SOL: [None, 0.274, 0.836, 1.13, 1.75, 4.09],
        GPT_6_ASTRA: [None, 1.15, 2.24, 2.06, 2.99, 4.37],
        GPT_6_SOL: [0.973, 0.248, 0.564, 0.806, 0.963, 2.02],
        GPT_6_1_SOL: [None, 0.196, 0.312, 0.421, 0.524, 0.933],
        CLAUDE_FABLE_5_1: [None, 3.7, 4.6, 5.86, 8.01, 9.94],
        CLAUDE_OPUS_5_5: [None, 1.05, 2.04, 2.58, 4.42, 6.79],
        CLAUDE_SONNET_5_5: [None, 0.649, 0.796, 1.1, 3.05, 6.46],
    },
    'ARC-AGI-2': {
        GPT_5_6_SOL: [None, 0.32, 0.47, 0.74, 1.04, 1.44],
        GPT_6_ASTRA: [0.37, 0.416, 0.48, 0.668, 0.829, 1.12],
        GPT_6_SOL: [0.0741, 0.101, 0.148, 0.203, 0.277, 0.439],
        GPT_6_1_SOL: [None, 0.0855, 0.1, 0.135, 0.178, 0.254],
        CLAUDE_FABLE_5_1: [None, 0.952, 1.22, 1.67, 3.12, 4.49],
        CLAUDE_OPUS_5_5: [None, 0.241, 0.337, 0.408, 0.671, 1.85],
    },
    'AA-Briefcase': {
        GPT_5_6_SOL: [None, 0.419, 0.968, 2.06, 3.08, 4.02],
        GPT_6_ASTRA: [None, 1.44, 3.94, 4.76, 6.56, 9.5],
        GPT_6_SOL: [0.374, 0.127, 0.33, 0.639, 1.13, 2.62],
        GPT_6_1_SOL: [None, 0.177, 0.464, 0.837, 1.04, 2.31],
        CLAUDE_FABLE_5_1: [None, 6.72, 8.58, 11.4, 17.8, 22.7],
        CLAUDE_OPUS_5_5: [None, 1.15, 4.4, 6.27, 12.3, 21.0],
        CLAUDE_SONNET_5_5: [None, 0.802, 1.39, 3.27, 6.98, 20.5],
    },
    'GDP.pdf': {
        GPT_5_6_SOL: [None, 0.63, 0.653, 0.7, 0.782, 0.929],
        GPT_6_ASTRA: [None, 1.7, 1.72, 1.79, 1.91, 2.08],
        GPT_6_SOL: [0.329, 0.331, 0.337, 0.348, 0.37, 0.425],
        GPT_6_1_SOL: [None, 0.334, 0.337, 0.349, 0.368, 0.42],
        CLAUDE_FABLE_5_1: [None, 1.92, 1.96, 2.02, 2.32, 2.77],
        CLAUDE_OPUS_5_5: [None, 0.764, 0.795, 0.825, 0.964, 1.55],
        CLAUDE_SONNET_5_5: [None, 0.372, 0.381, 0.409, 0.46, 0.794],
    },
}
COST_UNITS = {"AA Intelligence Index": "US$ to run the whole index, log scale"}
DEFAULT_COST_UNIT = "US$ per task, log scale"

TITLE = "Models by reasoning effort"
SUBTITLE = ("Tap a model to hide or show it. Show exactly two models to see where they come closest: "
            "a dashed line joins their nearest scores at different effort levels. Hover a level to read every score. "
            "With a mouse, drag across a chart to zoom in on the points inside. "
            "Switch the x axis to cost to see what each effort level costs: the best models sit top left.")

FOOTNOTES = {
    "Intelligence": "Hollow marker: estimated by Artificial Analysis, not measured. GPT-6 Astra has no “off” level "
                    "except in ARC Prize’s “none” run, and GPT-6.1 Sol has none at all. The Claude models have no “off” "
                    "level: AA and ARC Prize run them only with adaptive reasoning. ARC Prize has not tested Sonnet 5.5 "
                    "yet. AA re-rates its Elo benchmarks (GDPval-AA, AA-Briefcase) whenever a model joins; "
                    "these are the ratings of 7–8 October 2026. Cost: AA’s measured US-dollar cost "
                    "per task of each benchmark at each effort level, ARC Prize’s for ARC-AGI-2, and AA’s total cost "
                    "to run the whole AA Intelligence Index.",
    "Coding": "AA Coding Agent Index v1.4 as charted on OpenAI’s Astra page, before the newer GPT and Claude models. "
              "AA’s live version, max only: GPT-5.6 Sol 54.6, GPT-6 Astra 61.6, GPT-6 Sol 56.7, Fable 5.1 62.2, "
              "Opus 5.5 66.0. DeepSWE has no per-level scores for the Claude models; Sonnet 5.5’s system card "
              "reports 71.0 at max. AA ran Terminal-Bench 2.1 only on GPT-5.6 Sol, GPT-6 Astra and Fable 5.1. "
              "CursorBench lists no GPT-6 model. Terminal-Bench scores are AA’s own runs; "
              "Anthropic reports higher scores for its models. Cost: AA’s US-dollar cost per task; the coding "
              "category averages the costs of Terminal-Bench 4.0 and SciCode. The other coding charts have no cost data. "
              "SWE-Bench Pro and LiveCodeBench have one reported score per model, so they show as bars. Each lab ran "
              "SWE-Bench Pro itself on its own agent harness, which can move scores a lot, so compare its bars with care.",
}

LINE_WIDTH = 1.8
MARKER_SIZE = 6.5
MARKER_RING = 1.6
LABEL_FONT_SIZE = 9.5
END_LABEL_MIN_GAP_POINTS = 12
NAME_FONT_SIZE = 7.5
NAME_FONT_FAMILY = "Menlo"  # ships with macOS and iOS, so the SVG renders the font it was measured in
NAME_GAP_POINTS = 5



@dataclass(frozen=True)
class Layout:
    """One render of every chart, shown while the viewport is at least min_viewport pixels wide (up to the next layout).

    The SVG scales to its column, so the figure size in inches sets the text size: each layout's figure is about as
    wide as its column divided by 1.4 to 1.9, which keeps chart text at about 12 to 17px. Wider layouts widen the plot.
    """
    name: str
    min_viewport: int
    figure_size: tuple[float, float]


LAYOUTS = [
    Layout("phone", 0, (4.55, 3.9)),  # 3.8 plot inches plus the name column
    Layout("desktop", 700, (5.0, 4.6)),
    Layout("wide", 1600, (7.0, 4.8)),
    Layout("wider", 2100, (9.5, 5.0)),
    Layout("widest", 2800, (12.5, 5.2)),
]
AXES_LEFT = 0.42
# The right margin holds the model-name column beside the end labels, outside the plot, so names never meet plot labels.
AXES_RIGHT = 0.75
AXES_TOP = 0.22
AXES_BOTTOM = 0.32
COST_AXIS_LABEL_HEIGHT = 0.2  # the cost charts name their x-axis unit
COST_AXES_RIGHT = 0.1  # cost charts have no name column; their end labels sit inside the plot
X_LIMITS = (-0.85, len(EFFORT_LEVELS) - 1 + 0.9)


def configure_fonts() -> None:
    font_manager.fontManager.addfont("/System/Library/Fonts/HelveticaNeue.ttc")
    font_manager.fontManager.addfont("/System/Library/Fonts/Menlo.ttc")
    plt.rcParams.update({
        "font.family": "Helvetica Neue",
        "font.size": 9,
        "svg.fonttype": "none",
        "axes.edgecolor": BASELINE.light,
        "xtick.color": TEXT_MUTED.light,
        "ytick.color": TEXT_MUTED.light,
        "figure.facecolor": "none",
        "axes.facecolor": "none",
        "savefig.facecolor": "none",
    })


def present_points(scores: Scores) -> list[tuple[int, float]]:
    """Keep only the effort levels that have a score.

    >>> present_points([None, 1.0, 2.5])
    [(1, 1.0), (2, 2.5)]
    """
    return [(index, score) for index, score in enumerate(scores) if score is not None]


def labeled_levels(scores: Scores) -> set[int]:
    """The levels that always carry a label: the lowest (start label) and the highest (end label).

    >>> sorted(labeled_levels([None, 1.0, 2.0, 3.0]))
    [1, 3]
    """
    points = present_points(scores)
    return {points[0][0], points[-1][0]}


def spread_label_positions(positions: list[float], minimum_gap: float) -> list[float]:
    """Spread sorted positions so neighbors sit at least minimum_gap apart, each crowded group centered on its mean.

    >>> spread_label_positions([10.0, 12.0], 6.0)
    [8.0, 14.0]
    >>> spread_label_positions([0.0, 20.0], 6.0)
    [0.0, 20.0]
    >>> spread_label_positions([10.0, 11.0, 12.0], 6.0)
    [5.0, 11.0, 17.0]
    """
    def layout(group: list[float]) -> list[float]:
        center = sum(group) / len(group)
        return [center + (offset - (len(group) - 1) / 2) * minimum_gap for offset in range(len(group))]

    groups: list[list[float]] = []
    for position in sorted(positions):
        groups.append([position])
        while len(groups) > 1 and layout(groups[-1])[0] - layout(groups[-2])[-1] < minimum_gap:
            groups[-2:] = [groups[-2] + groups[-1]]
    return [placed for group in groups for placed in layout(group)]


def closest_meeting_levels(first: Scores, second: Scores) -> tuple[int, int]:
    """Find the two models' effort levels whose scores are closest; without overlap, that is the facing extremes.

    Ties go to the pair that adds fewer labels beyond each model's lowest and highest levels.

    >>> closest_meeting_levels([10.0, 20.0, 30.0], [None, 21.0, 40.0])
    (1, 1)
    >>> closest_meeting_levels([10.0, 20.0], [None, 50.0, 60.0])
    (1, 1)
    >>> closest_meeting_levels([10.0, 30.0, 25.0], [None, 55.0, 50.0, 60.0])
    (1, 2)
    """
    first_labeled = labeled_levels(first)
    second_labeled = labeled_levels(second)

    def rank(pair: tuple[tuple[int, float], tuple[int, float]]) -> tuple[float, int]:
        (first_index, first_score), (second_index, second_score) = pair
        added_labels = (first_index not in first_labeled) + (second_index not in second_labeled)
        return round(abs(first_score - second_score), 1), added_labels

    (first_index, _), (second_index, _) = min(itertools.product(present_points(first), present_points(second)), key=rank)
    return first_index, second_index


def nearest_score(scores: Scores, index: int) -> float:
    """Score at the given effort level, or at the nearest level that has one.

    >>> nearest_score([None, 5.0, 9.0], 0)
    5.0
    """
    return min(present_points(scores), key=lambda point: abs(point[0] - index))[1]


Meeting = tuple[tuple[Model, int], tuple[Model, int]]


def cross_level_meetings(metric: Metric) -> list[Meeting]:
    """Each model pair's closest points, skipping pairs that meet at the same effort level, where their lines already touch."""
    meetings = []
    for first, second in itertools.combinations(metric.models, 2):
        first_index, second_index = closest_meeting_levels(metric.scores[first], metric.scores[second])
        meetings.append(((first, first_index), (second, second_index)))
    return [meeting for meeting in meetings if meeting[0][1] != meeting[1][1]]


def element_id(role: str, *models: Model) -> str:
    """Tag an artist with its role and models. The page shows it only while all its models are visible.

    >>> element_id("meeting", GPT_6_ASTRA, GPT_6_SOL)
    'meeting__gpt-6-astra__gpt-6-sol'
    """
    return "__".join([role, *(model.slug for model in models)])


def element_models(artist: Artist) -> set[str]:
    return set(artist.get_gid().split("__")[1:])


# Meeting lines are faint references, like gridlines: labels may sit over them.
MEETING_LINE = "meeting-line"

Placement = tuple[float, float, str, str]

ABOVE: Placement = (0, 7, "center", "bottom")
BELOW: Placement = (0, -7, "center", "top")
ABOVE_LEFT: Placement = (-6, 6, "right", "bottom")
ABOVE_RIGHT: Placement = (6, 6, "left", "bottom")
BELOW_LEFT: Placement = (-6, -6, "right", "top")
BELOW_RIGHT: Placement = (6, -6, "left", "top")
LEFT: Placement = (-7, 0, "right", "center")
RIGHT: Placement = (7, 0, "left", "center")

# Keyed by (label role, whether the side away from the nearest other model is above).
PLACEMENT_ORDERS: dict[tuple[str, bool], list[Placement]] = {
    ("meeting", True): [ABOVE, ABOVE_RIGHT, ABOVE_LEFT, BELOW, BELOW_RIGHT, BELOW_LEFT, RIGHT, LEFT],
    ("meeting", False): [BELOW, BELOW_RIGHT, BELOW_LEFT, ABOVE, ABOVE_RIGHT, ABOVE_LEFT, RIGHT, LEFT],
    ("start", True): [LEFT, ABOVE_LEFT, ABOVE, BELOW_LEFT, BELOW, ABOVE_RIGHT, BELOW_RIGHT, RIGHT],
    ("start", False): [LEFT, BELOW_LEFT, BELOW, ABOVE_LEFT, ABOVE, BELOW_RIGHT, ABOVE_RIGHT, RIGHT],
}

LABEL_HALO = {"boxstyle": "round,pad=0.12", "facecolor": SURFACE.light, "edgecolor": "none"}
POINT_LABEL_STYLE = {"fontsize": LABEL_FONT_SIZE, "fontweight": "medium", "color": TEXT_PRIMARY.light, "zorder": 5,
                     "bbox": LABEL_HALO}
DIM_LABEL_STYLE = {"fontsize": NAME_FONT_SIZE, "family": NAME_FONT_FAMILY, "color": TEXT_MUTED.light, "zorder": 5,
                   "bbox": LABEL_HALO}

LabelPart = tuple[str, dict[str, object]]

# (x, y) unit direction and the text alignment that keeps the label on that side of its point.
DIRECTIONS: list[tuple[float, float, str, str]] = [
    (1, 0, "left", "center"), (0.8, 0.8, "left", "bottom"), (0.8, -0.8, "left", "top"), (0, 1, "center", "bottom"),
    (0, -1, "center", "top"), (-0.8, 0.8, "right", "bottom"), (-0.8, -0.8, "right", "top"), (-1, 0, "right", "center"),
]
# Cost charts crowd their line ends unevenly, so their labels search outward ring by ring; far rings get a leader line.
COST_END_PLACEMENTS: list[Placement] = [(x * radius, y * radius, horizontal, vertical)
                                        for radius in (7, 14, 22, 32, 44, 58) for x, y, horizontal, vertical in DIRECTIONS]
LEADER_MIN_DISTANCE_POINTS = 12



def line_hits(axes: Axes, line: Line2D, box: Bbox) -> bool:
    points = axes.transData.transform(line.get_xydata())
    marker_radius = (line.get_markersize() / 2 + line.get_markeredgewidth() / 2) * axes.figure.dpi / 72
    path_hit = len(points) >= 2 and MatplotlibPath(points).intersects_bbox(box, filled=False)
    marker_hit = line.get_marker() != "None" and any(
        Bbox.from_bounds(x - marker_radius, y - marker_radius, 2 * marker_radius, 2 * marker_radius).overlaps(box)
        for x, y in points)
    return path_hit or marker_hit


def text_box(text: Text, renderer: RendererBase) -> Bbox:
    """The text's own box: an annotation's extent would also span its leader line, which is not an obstacle."""
    if isinstance(text, Annotation):
        text.update_positions(renderer)  # applies the offset that Annotation.get_window_extent would apply
    return Text.get_window_extent(text, renderer)


def label_is_clear(axes: Axes, label: Annotation, renderer: RendererBase, scene: set[str], ignored_ids: set[str],
                   own_parts: list[Annotation]) -> bool:
    """A label is clear when it stays off the axes' sides and baseline and touches no line, marker or other label of its scene.

    The scene is the set of model slugs visible together with the label. The top is open. Meeting lines,
    artists whose gid is in ignored_ids, and the label's own other parts do not count.
    """
    def blocks(artist: Artist) -> bool:
        gid = artist.get_gid()
        return artist not in own_parts and gid not in ignored_ids and not gid.startswith(MEETING_LINE) \
            and element_models(artist) <= scene

    box = text_box(label, renderer).padded(2)
    plot = axes.get_window_extent(renderer)
    inside = plot.x0 <= box.x0 and box.x1 <= plot.x1 and plot.y0 <= box.y0
    hits_line = any(line_hits(axes, line, box) for line in axes.lines if blocks(line))
    hits_text = any(text_box(text, renderer).overlaps(box) for text in axes.texts if blocks(text))
    return inside and not hits_line and not hits_text


def annotate_parts(axes: Axes, point: tuple[float, float], parts: list[LabelPart], placement: Placement,
                   gid: str) -> list[Annotation]:
    """Annotate the first part at the placement and chain the others beside it, on the side away from the point."""
    renderer = axes.figure.canvas.get_renderer()
    dx, dy, horizontal, vertical = placement
    (first_text, first_style), *others = parts
    leader = {"arrowprops": {"arrowstyle": "-", "color": BASELINE.light, "linewidth": 0.6, "shrinkA": 1, "shrinkB": 4}} \
        if math.hypot(dx, dy) > LEADER_MIN_DISTANCE_POINTS else {}
    annotations = [axes.annotate(first_text, point, xytext=(dx, dy), textcoords="offset points",
                                 ha=horizontal, va=vertical, gid=gid, **first_style, **leader)]
    if leader:
        annotations[0].arrow_patch.set_gid(gid)  # the leader draws outside the text's SVG group, so tag it too
    point_x, point_y = axes.transData.transform(point)
    points_per_pixel = 72 / axes.figure.dpi
    leftward = horizontal == "right"
    for text, style in others:
        box = text_box(annotations[-1], renderer)
        edge = box.x0 - point_x if leftward else box.x1 - point_x
        offset = (edge * points_per_pixel + (-NAME_GAP_POINTS if leftward else NAME_GAP_POINTS),
                  ((box.y0 + box.y1) / 2 - point_y) * points_per_pixel)
        annotations.append(axes.annotate(text, point, xytext=offset, textcoords="offset points",
                                         ha="right" if leftward else "left", va="center", gid=gid, **style))
    return annotations


def place_label(axes: Axes, metric: Metric, point: tuple[float, float], parts: list[LabelPart],
                placements: list[Placement], gid: str, owner: Model, scene: list[Model], required: bool = True) -> None:
    """Put the label at the first clear placement, relaxing the rules pass by pass in crowded spots.

    The passes: touch no series line; then cover only its owner's line; then cover any line, with its halo.
    Markers and other labels always block. When no pass finds a spot, a required label raises ValueError
    and an optional one is left out; its score stays in the hover tooltip and the table.
    """
    renderer = axes.figure.canvas.get_renderer()
    scene_slugs = {model.slug for model in scene}
    passes = [set(), {element_id("series", owner)}, {element_id("series", model) for model in scene}]
    for ignored_ids, placement in itertools.product(passes, placements):
        annotations = annotate_parts(axes, point, parts, placement, gid)
        if all(label_is_clear(axes, annotation, renderer, scene_slugs, ignored_ids, annotations)
               for annotation in annotations):
            return
        for annotation in annotations:
            annotation.remove()
    if not required:
        return
    raise ValueError(f"{metric.name}: no clear spot for {owner.name}'s label {' '.join(text for text, _ in parts)}")


def value_part(metric: Metric, score: float) -> LabelPart:
    return metric.label(score), POINT_LABEL_STYLE


def away_is_above(metric: Metric, model: Model, index: int, scene: list[Model]) -> bool:
    """Whether the side away from the nearest other model's line in the scene is above this point."""
    score = metric.scores[model][index]
    neighbor = min((nearest_score(metric.scores[other], index) for other in scene if other is not model),
                   key=lambda other_score: abs(other_score - score))
    return score >= neighbor


def draw_start_labels(axes: Axes, metric: Metric) -> None:
    """Label each model's lowest effort level, clear of every model's lines and labels, where crowding allows."""
    for model in metric.models:
        index = present_points(metric.scores[model])[0][0]
        score = metric.scores[model][index]
        placements = PLACEMENT_ORDERS[("start", away_is_above(metric, model, index, metric.models))]
        place_label(axes, metric, (index, score), [value_part(metric, score)], placements, element_id("start", model),
                    model, metric.models, required=False)


def draw_meetings(axes: Axes, metric: Metric) -> None:
    """Draw each cross-level meeting: a faint dashed line at the lower of the two scores, and labels on its points.

    A meeting shows only while its two models are the only visible ones, so its labels need to clear only their lines.
    """
    for meeting in cross_level_meetings(metric):
        (first, first_index), (second, second_index) = meeting
        height = min(metric.scores[first][first_index], metric.scores[second][second_index])
        axes.plot([first_index, second_index], [height, height], color=TEXT_MUTED.light, linewidth=0.9,
                  linestyle=(0, (2.5, 2.5)), alpha=0.8, zorder=2.5, gid=element_id(MEETING_LINE, first, second))
        for model, index in meeting:
            if index in labeled_levels(metric.scores[model]):
                continue
            score = metric.scores[model][index]
            placements = PLACEMENT_ORDERS[("meeting", away_is_above(metric, model, index, [first, second]))]
            place_label(axes, metric, (index, score), [value_part(metric, score)], placements,
                        element_id("meeting", first, second), model, [first, second])


def draw_end_labels(axes: Axes, metric: Metric) -> None:
    """Label each series' last score to the right of its end point, with a leader line when displaced.

    A dim model name follows each end label, in one column past the widest end label, so no one has to decode colors.
    """
    ends = sorted(((model, *present_points(metric.scores[model])[-1]) for model in metric.models), key=lambda end: end[2])
    to_display = axes.transData.transform
    from_display = axes.transData.inverted().transform
    minimum_gap_pixels = END_LABEL_MIN_GAP_POINTS * axes.figure.dpi / 72
    label_heights = spread_label_positions([to_display((index, score))[1] for _, index, score in ends], minimum_gap_pixels)
    label_x = len(EFFORT_LEVELS) - 1 + 0.32
    label_ys = [from_display((0, label_height))[1] for label_height in label_heights]
    for (model, index, score), label_y in zip(ends, label_ys):
        if abs(label_y - score) > 1e-9:
            axes.plot([index + 0.1, label_x - 0.06], [score, label_y], color=BASELINE.light, linewidth=0.6,
                      solid_capstyle="round", zorder=2, gid=element_id("leader", model))
        axes.text(label_x, label_y, metric.label(score), color=TEXT_PRIMARY.light, fontsize=LABEL_FONT_SIZE,
                  fontweight="medium", va="center", ha="left", gid=element_id("end", model))
    renderer = axes.figure.canvas.get_renderer()
    widest_end = max(text.get_window_extent(renderer).x1 for text in axes.texts)
    name_x = from_display((widest_end + NAME_GAP_POINTS * axes.figure.dpi / 72, 0))[0]
    for (model, _, _), label_y in zip(ends, label_ys):
        axes.text(name_x, label_y, model.name, color=TEXT_MUTED.light, fontsize=NAME_FONT_SIZE, family=NAME_FONT_FAMILY,
                  va="center", ha="left", gid=element_id("name", model))


def draw_series(axes: Axes, metric: Metric, model: Model, points: list[tuple[int, float, float]]) -> None:
    """Draw one model's line and markers through (level, x, score) points, hollow where AA estimated the score."""
    axes.plot([x for _, x, _ in points], [score for _, _, score in points], color=model.color.light,
              linewidth=LINE_WIDTH, solid_joinstyle="round", solid_capstyle="round", zorder=3,
              gid=element_id("series", model))
    for index, x, score in points:
        is_estimated = (model, EFFORT_LEVELS[index]) in metric.estimated
        axes.plot(x, score, marker="o", markersize=MARKER_SIZE,
                  markerfacecolor=SURFACE.light if is_estimated else model.color.light,
                  markeredgecolor=model.color.light if is_estimated else SURFACE.light,
                  markeredgewidth=MARKER_RING, zorder=4, gid=element_id("marker", model))


def style_axes(axes: Axes, metric: Metric) -> None:
    axes.set_ylim(*metric.y_range)
    axes.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(metric.tick_step))
    axes.grid(axis="y", color=GRIDLINE.light, linewidth=0.7)
    axes.set_axisbelow(True)
    for side in ("top", "right", "left"):
        axes.spines[side].set_visible(False)
    axes.spines["bottom"].set_color(BASELINE.light)
    axes.spines["bottom"].set_linewidth(0.8)
    axes.tick_params(axis="both", which="both", length=0, labelsize=9, pad=5)


def draw_panel(axes: Axes, metric: Metric) -> None:
    for model in metric.models:
        draw_series(axes, metric, model, [(index, index, score) for index, score in present_points(metric.scores[model])])
    style_axes(axes, metric)
    axes.set_xlim(*X_LIMITS)
    axes.set_xticks(range(len(EFFORT_LEVELS)), EFFORT_LEVELS)
    axes.spines["bottom"].set_bounds(-0.35, len(EFFORT_LEVELS) - 1 + 0.35)
    axes.figure.canvas.draw()
    draw_end_labels(axes, metric)
    draw_start_labels(axes, metric)
    draw_meetings(axes, metric)


def cost_points(metric: Metric, model: Model) -> list[tuple[int, float, float]]:
    """The (level, cost, score) points where the model has both a score and a cost.

    >>> cost_points(INTELLIGENCE[0], GPT_5_6_SOL)[0]
    (1, 637.0, 33.5)
    """
    costs = metric.costs.get(model, [None] * len(EFFORT_LEVELS))
    return [(index, cost, score) for index, (score, cost) in enumerate(zip(metric.scores[model], costs))
            if score is not None and cost is not None]


def cost_label(value: float) -> str:
    """
    >>> [cost_label(value) for value in (0.002, 0.05, 1.5, 20, 1000, 12500)]
    ['$0.002', '$0.05', '$1.5', '$20', '$1k', '$12.5k']
    """
    return f"${value / 1000:.3g}k" if value >= 1000 else f"${value:.3g}"


def draw_cost_panel(axes: Axes, metric: Metric) -> None:
    """Plot score against the benchmark's own cost, log scale; each model's line still runs through its effort levels.

    Only each line's last point carries a label: its value and dim model name, plus its level if not max. Points
    crowd here, and the x axis is not effort levels, so a first-point label would anchor nothing; hover shows every point.
    """
    models = [model for model in metric.models if cost_points(metric, model)]
    for model in models:
        draw_series(axes, metric, model, cost_points(metric, model))
    style_axes(axes, metric)
    costs = [cost for model in models for _, cost, _ in cost_points(metric, model)]
    decades = math.log10(max(costs) / min(costs))
    axes.set_xscale("log")
    axes.set_xlim(min(costs) / 10 ** (0.12 * decades + 0.1), max(costs) * 10 ** (0.35 * decades + 0.2))
    tick_multiples = (1, 2, 5) if decades <= 1.5 else (1, 3) if decades <= 2.5 else (1,)
    axes.xaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=tick_multiples))
    axes.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    axes.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda value, _: cost_label(value)))
    axes.grid(axis="x", color=GRIDLINE.light, linewidth=0.7)
    axes.set_xlabel(metric.cost_unit, fontsize=8.5, color=TEXT_MUTED.light, labelpad=6)
    axes.figure.canvas.draw()
    for model in models:
        index, cost, score = cost_points(metric, model)[-1]
        level = "" if EFFORT_LEVELS[index] == "max" else f" · {EFFORT_LEVELS[index]}"  # most lines end at max
        parts = [value_part(metric, score), (f"{model.name}{level}", DIM_LABEL_STYLE)]
        place_label(axes, metric, (cost, score), parts, COST_END_PLACEMENTS, element_id("end", model), model, models)


def inline_svg(svg: str) -> str:
    """Make matplotlib's SVG fit inline in the page: no fixed size, theme colors as CSS variables, gids as CSS classes.

    >>> inline_svg('<?xml?>\\n<svg width="9pt" height="9pt" viewBox="0 0 9 9"><g id="end__gpt-6-sol" style="fill: #0b0b0b"/></svg>')
    '<svg class="chart" viewBox="0 0 9 9"><g class="end m-gpt-6-sol" style="fill: var(--ink-1)"/></svg>'
    """
    svg = svg[svg.index("<svg"):]
    # matplotlib's global "*" rule and metadata would leak into, or bloat, the page.
    svg = re.sub(r"<metadata>.*?</metadata>|<style[^>]*>.*?</style>", "", svg, flags=re.DOTALL)
    svg = re.sub(r' width="[\d.]+pt" height="[\d.]+pt"', ' class="chart"', svg, count=1)
    for color in COLORS:
        svg = svg.replace(color.light, f"var(--{color.variable})")
    return re.sub(r'id="([a-z-]+)((?:__[a-z0-9-]+)+)"',
                  lambda match: 'class="' + " ".join([match[1], *(f"m-{slug}" for slug in match[2].split("__")[1:])]) + '"',
                  svg)


def level_column_html(axes: Axes, metric: Metric, index: int) -> str:
    """A hover column over one effort level: a crosshair, rings on its points, and a tooltip with every model's score."""
    width, height = axes.figure.bbox.size
    left, top = axes.transData.transform((index - 0.5, metric.y_range[1]))
    right, bottom = axes.transData.transform((index + 0.5, metric.y_range[0]))
    rows, rings = [], []
    present = [(model, metric.scores[model][index]) for model in metric.models if metric.scores[model][index] is not None]
    for model, score in sorted(present, key=lambda item: -item[1]):
        estimated = " <i>est.</i>" if (model, EFFORT_LEVELS[index]) in metric.estimated else ""
        rows.append(f'<span class="row m-{model.slug}" style="--series: var(--{model.color.variable})">'
                    f'<b>{metric.label(score)}</b>{html.escape(model.name)}{estimated}</span>')
        ring_top = 100 * (top - axes.transData.transform((index, score))[1]) / (top - bottom)
        rings.append(f'<span class="ring m-{model.slug}" style="top:{ring_top:.2f}%"></span>')
    side = "tip-left" if index >= len(EFFORT_LEVELS) - 2 else "tip-right"
    return (f'<span class="column" style="left:{100 * left / width:.2f}%;width:{100 * (right - left) / width:.2f}%;'
            f'top:{100 * (1 - top / height):.2f}%;height:{100 * (top - bottom) / height:.2f}%">'
            f'{"".join(rings)}<span class="tip {side}"><span class="level">{EFFORT_LEVELS[index]}</span>{"".join(rows)}</span></span>')


def cost_point_html(axes: Axes, metric: Metric, model: Model, index: int, cost: float, score: float) -> str:
    """A hover target on one cost-chart point: a ring and a tooltip with its level, score and cost."""
    width, height = axes.figure.bbox.size
    x, y = axes.transData.transform((cost, score))
    estimated = " <i>est.</i>" if (model, EFFORT_LEVELS[index]) in metric.estimated else ""
    side = "tip-left" if x / width > 0.6 else "tip-right"
    return (f'<span class="cost-point m-{model.slug}" style="left:{100 * x / width:.2f}%;top:{100 * (1 - y / height):.2f}%">'
            f'<span class="tip {side}"><span class="level">{EFFORT_LEVELS[index]}</span>'
            f'<span class="row" style="--series: var(--{model.color.variable})"><b>{metric.label(score)}</b>'
            f'{html.escape(model.name)}{estimated}</span><span class="cost">{cost_label(cost)} · {metric.cost_unit.split(",")[0]}</span>'
            f'</span></span>')


# A chart without cost data shows its effort chart in both modes.
Mode = str


def plot_html(metric: Metric, layout: Layout, mode: Mode) -> str:
    """One layout's chart in one x-axis mode: the inline SVG plus its hover targets."""
    width, height = layout.figure_size
    bottom = AXES_BOTTOM + (COST_AXIS_LABEL_HEIGHT if mode == "cost" else 0)
    right = COST_AXES_RIGHT if mode == "cost" else AXES_RIGHT
    plt.rcParams["svg.hashsalt"] = f"{metric.name} {layout.name} {mode}"  # keeps SVG ids unique across the page
    figure = plt.figure(figsize=(width, height), dpi=72)
    axes = figure.add_axes((AXES_LEFT / width, bottom / height,
                            1 - (AXES_LEFT + right) / width, 1 - (AXES_TOP + bottom) / height))
    if mode == "cost":
        draw_cost_panel(axes, metric)
        hover = "".join(cost_point_html(axes, metric, model, index, cost, score) for model in metric.models
                        for index, cost, score in cost_points(metric, model))
        attributes = ""
    else:
        draw_panel(axes, metric)
        hover = "".join(level_column_html(axes, metric, index) for index in range(len(EFFORT_LEVELS))
                        if any(metric.scores[model][index] is not None for model in metric.models))
        box = axes.get_position()
        axes_percent = ",".join(f"{100 * value:.3f}" for value in (box.x0, 1 - box.y1, box.width, box.height))
        attributes = f' data-metric="{slugify(metric.name)}" data-axes="{axes_percent}"'
    svg = io.StringIO()
    figure.savefig(svg, format="svg", metadata={"Date": None})
    plt.close(figure)
    modes = mode if metric.costs else "effort cost"
    return f'<div class="plot {layout.name} {modes}"{attributes}>{inline_svg(svg.getvalue())}{hover}</div>'


def panel_html(metric: Metric) -> str:
    modes = ["effort", "cost"] if metric.costs else ["effort"]
    note = "" if metric.costs else '<p class="no-cost">No cost data for this benchmark. Showing scores by effort level.</p>'
    return (f'<figure class="panel"><figcaption><h3>{html.escape(metric.name)}</h3>'
            f'<p>{html.escape(metric.unit_and_source)}</p>{note}</figcaption>'
            f'{"".join(plot_html(metric, layout, mode) for layout in LAYOUTS for mode in modes)}</figure>')


BAR_ROW_HEIGHT = 0.42
BAR_THICKNESS = 0.56  # of a row
BAR_AXES_LEFT = 0.95  # holds the model names
BAR_AXES_BOTTOM = 0.32


def draw_bars(axes: Axes, metric: ReportedMetric) -> None:
    """One horizontal bar per reported model, top to bottom in model order, labeled with its score and effort level."""
    rows = list(reversed(metric.models))
    for row, model in enumerate(rows):
        score, effort = metric.scores[model]
        axes.barh(row, score, height=BAR_THICKNESS, color=model.color.light, zorder=3, gid=element_id("bar", model))
        axes.annotate(model.name, (0, row), xytext=(-8, 0), textcoords="offset points", ha="right", va="center",
                      fontsize=LABEL_FONT_SIZE, color=TEXT_PRIMARY.light, annotation_clip=False, gid=element_id("name", model))
    axes.set_xlim(0, 100)
    axes.set_ylim(-0.6, len(rows) - 0.4)
    axes.set_yticks([])
    axes.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(25))
    axes.grid(axis="x", color=GRIDLINE.light, linewidth=0.7)
    axes.set_axisbelow(True)
    for side in ("top", "right", "bottom"):
        axes.spines[side].set_visible(False)
    axes.spines["left"].set_color(BASELINE.light)
    axes.tick_params(axis="both", length=0, labelsize=9, pad=5)
    axes.figure.canvas.draw()
    for row, model in enumerate(rows):
        score, effort = metric.scores[model]
        for part in annotate_parts(axes, (score, row), [(f"{score:.1f}", POINT_LABEL_STYLE), (effort, DIM_LABEL_STYLE)],
                                   RIGHT, element_id("end", model)):
            part.set_annotation_clip(False)


def bar_plot_html(metric: ReportedMetric, layout: Layout) -> str:
    width = layout.figure_size[0]
    height = AXES_TOP + BAR_AXES_BOTTOM + BAR_ROW_HEIGHT * len(metric.models)
    plt.rcParams["svg.hashsalt"] = f"{metric.name} {layout.name}"
    figure = plt.figure(figsize=(width, height), dpi=72)
    axes = figure.add_axes((BAR_AXES_LEFT / width, BAR_AXES_BOTTOM / height,
                            1 - (BAR_AXES_LEFT + AXES_RIGHT) / width, 1 - (AXES_TOP + BAR_AXES_BOTTOM) / height))
    draw_bars(axes, metric)
    svg = io.StringIO()
    figure.savefig(svg, format="svg", metadata={"Date": None})
    plt.close(figure)
    return f'<div class="plot {layout.name} effort cost">{inline_svg(svg.getvalue())}</div>'


def bar_panel_html(metric: ReportedMetric) -> str:
    missing = ", ".join(model.name for model in metric.missing)
    return (f'<figure class="panel"><figcaption><h3>{html.escape(metric.name)}</h3>'
            f'<p>{html.escape(metric.unit_and_source)}</p>'
            f'<p class="missing">No reported score: {html.escape(missing)}</p></figcaption>'
            f'{"".join(bar_plot_html(metric, layout) for layout in LAYOUTS)}</figure>')


def table_rows() -> Iterator[tuple[str, Metric, Model]]:
    for category, metrics in SECTIONS:
        for metric in metrics:
            for model in metric.models:
                yield category, metric, model


def table_html() -> str:
    head = "".join(f"<th>{level}</th>" for level in EFFORT_LEVELS)
    rows = []
    for category, metric, model in table_rows():
        is_first = model is metric.models[0]
        cells = "".join(f"<td>{'' if score is None else metric.label(score)}</td>" for score in metric.scores[model])
        rows.append(f'<tr class="m-{model.slug}{" first" if is_first else ""}">'
                    f'<th scope="row">{html.escape(metric.name) if is_first else ""}</th>'
                    f'<td class="model">{html.escape(model.name)}</td>{cells}</tr>')
        if metric.costs and model in metric.costs:
            costs = "".join(f"<td>{'' if cost is None else cost_label(cost)}</td>" for cost in metric.costs[model])
            rows.append(f'<tr class="cost-row m-{model.slug}"><th scope="row"></th><td class="model">cost</td>{costs}</tr>')
    return f'<table><thead><tr><th>Metric</th><th>Model</th>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table>'


def layout_css() -> str:
    """Show each layout's charts only inside its viewport range, in the x-axis mode the switch selects.

    >>> print(layout_css().splitlines()[0])
    @media (min-width: 0px) and (max-width: 699.98px) { body:has(#x-effort:checked) .plot.phone.effort, body:has(#x-cost:checked) .plot.phone.cost { display: block; } }
    """
    ends = [f" and (max-width: {following.min_viewport - 0.02}px)" for following in LAYOUTS[1:]] + [""]
    return "\n".join(f"@media (min-width: {layout.min_viewport}px){end} {{ "
                     + ", ".join(f"body:has(#x-{mode}:checked) .plot.{layout.name}.{mode}" for mode in ("effort", "cost"))
                     + " { display: block; } }"
                     for layout, end in zip(LAYOUTS, ends))


def toggle_css() -> str:
    """Rules that hide a model everywhere when its checkbox is off, and show a meeting only when its two models alone are on."""
    rules = []
    for model in MODELS:
        off = f"body:has(#show-{model.slug}:not(:checked))"
        rules.append(f"{off} .m-{model.slug} {{ display: none; }}")
        rules.append(f"{off} label[for=show-{model.slug}] {{ opacity: .45; text-decoration: line-through; }}")
        rules.append(f"body:has(#show-{model.slug}:focus-visible) label[for=show-{model.slug}] "
                     f"{{ outline: 2px solid var(--ink-2); outline-offset: 2px; }}")
    for pair in itertools.combinations(MODELS, 2):
        states = "".join(f":has(#show-{model.slug}:{'checked' if model in pair else 'not(:checked)'})" for model in MODELS)
        rules.append(f"body{states} :is(.meeting, .meeting-line)"
                     f"{''.join(f'.m-{model.slug}' for model in pair)} {{ display: inline; }}")
    return "\n".join(rules)


PAGE_TEMPLATE = Template("""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>$title</title>
<style>
:root { color-scheme: light; $light }
@media (prefers-color-scheme: dark) { :root { color-scheme: dark; $dark } }
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink-1);
  font: 15px/1.45 "Helvetica Neue", system-ui, -apple-system, sans-serif; -webkit-text-size-adjust: 100%; }
.shell { padding: 0 clamp(16px, 2vw, 40px); }
header { padding: 28px 0 12px; }
h1 { margin: 0 0 6px; font-size: 24px; letter-spacing: -0.01em; }
.subtitle { margin: 0; max-width: 72ch; color: var(--ink-2); }
.legend { position: sticky; top: 0; z-index: 10; background: var(--page); border-bottom: 1px solid var(--border); }
.legend .shell { display: flex; flex-wrap: wrap; gap: 6px 18px; padding-top: 10px; padding-bottom: 10px; }
.legend label { display: inline-flex; align-items: center; gap: 8px; cursor: pointer; user-select: none;
  -webkit-user-select: none; padding: 4px 0; font-weight: 500; }
.legend label.model::before { content: ""; width: 18px; height: 3px; border-radius: 2px; background: var(--series); }
.axis-switch { display: inline-flex; align-items: center; gap: 0; margin-left: auto; color: var(--ink-3); font-size: 13px; }
.axis-switch span { margin-right: 8px; }
.legend .axis-switch label { padding: 3px 12px; border: 1px solid var(--baseline); font-weight: 500; font-size: 13px; color: var(--ink-2); }
.axis-switch label:first-of-type { border-radius: 6px 0 0 6px; }
.axis-switch label:last-of-type { border-radius: 0 6px 6px 0; border-left: 0; }
body:has(#x-effort:checked) label[for=x-effort], body:has(#x-cost:checked) label[for=x-cost] {
  background: var(--ink-1); border-color: var(--ink-1); color: var(--surface); }
body:has(#x-effort:focus-visible) label[for=x-effort], body:has(#x-cost:focus-visible) label[for=x-cost] {
  outline: 2px solid var(--ink-2); outline-offset: 2px; }
.missing { color: var(--ink-3); }
.no-cost { display: none; }
body:has(#x-cost:checked) .no-cost { display: block; }
.cost-row { display: none; color: var(--ink-3); }
body:has(#x-cost:checked) .cost-row { display: table-row; }
.toggle { position: absolute; opacity: 0; pointer-events: none; }
h2 { margin: 32px 0 12px; font-size: 13px; letter-spacing: .08em; text-transform: uppercase; color: var(--ink-2); }
.grid { display: grid; grid-template-columns: 1fr; gap: 16px; }
@media (min-width: 960px) { .grid { grid-template-columns: 1fr 1fr; } }
.panel { margin: 0; padding: 16px 16px 8px; background: var(--surface); border: 1px solid var(--border); border-radius: 10px; }
figcaption h3 { margin: 0; font-size: 16px; }
figcaption p { margin: 2px 0 0; font-size: 13px; color: var(--ink-3); }
.plot { position: relative; display: none; }
@media (max-width: 699.98px) { .panel { padding: 12px 10px 6px; } }
$layouts
.chart { display: block; width: 100%; height: auto; overflow: visible; }
.meeting, .meeting-line { display: none; }
.column { position: absolute; }
.column::before { content: ""; position: absolute; left: 50%; top: 0; bottom: 0; border-left: 1px solid var(--baseline); opacity: 0; }
.ring { position: absolute; left: 50%; width: 16px; height: 16px; margin: -8px 0 0 -8px; border-radius: 50%;
  box-shadow: 0 0 0 1.5px var(--ink-1); opacity: 0; }
.tip { position: absolute; top: 0; z-index: 5; display: grid; gap: 2px; padding: 7px 10px; min-width: 150px;
  background: var(--surface); border: 1px solid var(--border); border-radius: 8px; box-shadow: 0 4px 14px rgba(0,0,0,.14);
  font-size: 12.5px; white-space: nowrap; color: var(--ink-2); pointer-events: none; }
/* Hidden tips take no space, so they cannot widen the page on phones. */
.column:not(:hover) .tip { display: none; }
.tip-right { left: calc(50% + 10px); }
.tip-left { right: calc(50% + 10px); }
.tip .level { font-weight: 600; color: var(--ink-3); text-transform: uppercase; font-size: 11px; letter-spacing: .06em; }
.tip .row { display: flex; align-items: center; gap: 8px; }
.tip .row::before { content: ""; width: 12px; height: 2.5px; border-radius: 2px; background: var(--series); flex: none; }
.tip b { min-width: 3.2em; color: var(--ink-1); font-variant-numeric: tabular-nums; }
.tip i { color: var(--ink-3); }
.cost-point { position: absolute; width: 24px; height: 24px; margin: -12px 0 0 -12px; border-radius: 50%; }
.cost-point::after { content: ""; position: absolute; inset: 4px; border-radius: 50%; box-shadow: 0 0 0 1.5px var(--ink-1); opacity: 0; }
.cost-point:not(:hover) .tip { display: none; }
.cost-point .tip { top: -8px; }
.cost-point .tip-right { left: calc(100% + 4px); }
.cost-point .tip-left { right: calc(100% + 4px); }
.tip .cost { color: var(--ink-3); }
.cost-point:hover { z-index: 4; }
.cost-point:hover::after { opacity: 1; }
.column:hover { z-index: 4; }
.column:hover::before, .column:hover .ring { opacity: 1; }
.footnote { margin: 12px 0 0; max-width: 90ch; font-size: 12.5px; color: var(--ink-3); }
details { margin: 32px 0 48px; }
summary { cursor: pointer; font-weight: 600; color: var(--ink-2); }
.table-wrap { overflow-x: auto; margin-top: 12px; }
table { border-collapse: collapse; font-size: 13px; font-variant-numeric: tabular-nums; }
th, td { padding: 4px 10px; text-align: right; white-space: nowrap; }
thead th { color: var(--ink-3); font-weight: 500; border-bottom: 1px solid var(--baseline); }
tbody th, td.model { text-align: left; }
tr.first > * { border-top: 1px solid var(--grid); }
.zoomable .plot[data-metric] { cursor: crosshair; user-select: none; -webkit-user-select: none; }
.dragging .tip, .dragging .column::before, .dragging .ring { display: none; }
.selection { position: absolute; z-index: 6; pointer-events: none; border: 1px solid var(--ink-2);
  background: color-mix(in srgb, var(--ink-2) 10%, transparent); }
#zoom { width: 100vw; height: 100vh; max-width: none; max-height: none; margin: 0; padding: 24px 32px 16px; border: 0;
  background: var(--surface); color: var(--ink-1); }
#zoom[open] { display: flex; flex-direction: column; }
#zoom::backdrop { background: rgba(0, 0, 0, .45); }
#zoom:focus { outline: none; }
.zoom-head { display: flex; justify-content: space-between; align-items: flex-start; gap: 16px; }
.zoom-head h3 { margin: 0; font-size: 22px; }
.zoom-head p { margin: 2px 0 0; color: var(--ink-3); }
.zoom-head button { font: 28px/1 system-ui, sans-serif; color: var(--ink-2); background: none; border: 0; cursor: pointer; padding: 0 4px; }
.zoom-legend { display: flex; flex-wrap: wrap; gap: 6px 18px; margin: 12px 0 4px; font-weight: 500; }
.zoom-legend span { display: inline-flex; align-items: center; gap: 8px; }
.zoom-legend span::before { content: ""; width: 18px; height: 3px; border-radius: 2px; background: var(--series); }
.zoom-plot { flex: 1; min-height: 0; }
.zoom-plot svg { display: block; width: 100%; height: 100%; overflow: visible; }
$toggles
</style>
</head>
<body>
$inputs
<header class="shell"><h1>$title</h1><p class="subtitle">$subtitle</p></header>
<nav class="legend" aria-label="Models"><div class="shell">$legend</div></nav>
<main class="shell">
$sections
<details><summary>Data table</summary><div class="table-wrap">$table</div></details>
</main>
<dialog id="zoom" aria-label="Zoomed chart" tabindex="-1">
<div class="zoom-head"><div><h3></h3><p class="zoom-unit"></p><p class="zoom-range"></p></div>
<form method="dialog"><button aria-label="Close">×</button></form></div>
<div class="zoom-legend"></div>
<div class="zoom-plot"></div>
</dialog>
<script type="application/json" id="chart-data">$data</script>
<script>
$script
</script>
</body>
</html>
""")


def page_html() -> str:
    sections = []
    for name, metrics in SECTIONS:
        panels = "".join(panel_html(metric) for metric in metrics) \
            + "".join(bar_panel_html(metric) for metric in REPORTED.get(name, []))
        sections.append(f'<section><h2>{name}</h2><div class="grid">{panels}</div>'
                        f'<p class="footnote">{html.escape(FOOTNOTES[name])}</p></section>')
    return PAGE_TEMPLATE.substitute(
        title=TITLE,
        subtitle=html.escape(SUBTITLE),
        light=" ".join(f"--{color.variable}: {color.light};" for color in COLORS),
        dark=" ".join(f"--{color.variable}: {color.dark};" for color in COLORS),
        toggles=toggle_css(),
        layouts=layout_css(),
        inputs="".join(f'<input type="checkbox" class="toggle" id="show-{model.slug}" checked>' for model in MODELS)
               + '<input type="radio" class="toggle" name="x-axis" id="x-effort" checked>'
               + '<input type="radio" class="toggle" name="x-axis" id="x-cost">',
        legend="".join(f'<label class="model" for="show-{model.slug}" style="--series: var(--{model.color.variable})">'
                       f'{html.escape(model.name)}</label>' for model in MODELS)
               + '<span class="axis-switch" role="group" aria-label="X axis"><span>X axis</span>'
                 '<label for="x-effort">Effort level</label><label for="x-cost">Cost</label></span>',
        sections="\n".join(sections),
        table=table_html(),
        data=json.dumps(chart_data(), ensure_ascii=False).replace("</", "<\\/"),
        script=(Path(__file__).parent / "zoom.js").read_text(),
    )


def chart_data() -> dict[str, object]:
    """The scores the zoom script redraws from, keyed by the same slugs the page uses."""
    return {
        "levels": EFFORT_LEVELS,
        "xLimits": X_LIMITS,
        "models": [{"slug": model.slug, "name": model.name, "color": model.color.variable} for model in MODELS],
        "metrics": {slugify(metric.name): {
            "name": metric.name,
            "unit": metric.unit_and_source,
            "yRange": metric.y_range,
            "decimals": metric.decimals,
            "scores": {model.slug: metric.scores[model] for model in metric.models},
            "estimated": [[model.slug, EFFORT_LEVELS.index(level)] for model, level in metric.estimated],
        } for _, metrics in SECTIONS for metric in metrics},
    }


def write_table(path: Path) -> None:
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["category", "metric", "unit_and_source", "model", "measure", *EFFORT_LEVELS])
        for category, metric, model in table_rows():
            writer.writerow([category.lower(), metric.name, metric.unit_and_source, model.name, "score",
                             *("" if score is None else score for score in metric.scores[model])])
            if metric.costs and model in metric.costs:
                writer.writerow([category.lower(), metric.name, metric.cost_unit, model.name, "cost_usd",
                                 *("" if cost is None else cost for cost in metric.costs[model])])


def main() -> None:
    configure_fonts()
    OUTPUT_DIRECTORY.mkdir(exist_ok=True)
    (OUTPUT_DIRECTORY / "models-by-reasoning-effort.html").write_text(page_html())
    write_table(OUTPUT_DIRECTORY / "data.csv")


if __name__ == "__main__":
    main()
