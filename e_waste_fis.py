# e_waste_fis.py
# Sugeno fuzzy inference system for e-waste sorting
# Inputs: Metal Content (0–100), Contamination (0–100), Repairability (0–10), Hazard (0–10)
# Output (Sugeno constant): 25=Refurbish, 50=Material Recovery, 75=Hazardous Handling, 90=Safe Disposal

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Tuple

OUTPUT_CONSTANTS: Dict[str, float] = {
    "Refurbish/Reuse": 25.0,
    "Material Recovery": 50.0,
    "Hazardous Handling": 75.0,
    "Safe Disposal": 90.0,
}


# 1) Membership functions
def tri(x: np.ndarray | float, a: float, b: float, c: float) -> np.ndarray:
    """Triangular MF (a, b, c). Supports shoulder cases when a==b or b==c."""
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    left = (a < x) & (x < b)
    y[left] = (x[left] - a) / (b - a + 1e-12)
    y[x == b] = 1.0
    right = (b < x) & (x < c)
    y[right] = (c - x[right]) / (c - b + 1e-12)
    y[(x <= a) & (a == b)] = 1.0
    y[(x >= c) & (b == c)] = 1.0
    return np.clip(y, 0.0, 1.0)

def trap(x: np.ndarray | float, a: float, b: float, c: float, d: float) -> np.ndarray:
    """Trapezoidal MF (a, b, c, d). Supports shoulder cases when a==b or c==d."""
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    left = (a < x) & (x < b)
    y[left] = (x[left] - a) / (b - a + 1e-12)
    y[(b <= x) & (x <= c)] = 1.0
    right = (c < x) & (x < d)
    y[right] = (d - x[right]) / (d - c + 1e-12)
    y[(x <= a) & (a == b)] = 1.0
    y[(x >= d) & (c == d)] = 1.0
    return np.clip(y, 0.0, 1.0)


# 2) Variables & sets
@dataclass
class FuzzyVariable:
    name: str
    u_min: float
    u_max: float
    sets: Dict[str, Tuple[str, Tuple[float, ...]]]  # label -> (mf_type, params)

    def mu(self, label: str, x: float) -> float:
        typ, params = self.sets[label]
        if   typ == "tri":  return float(tri(x, *params))
        elif typ == "trap": return float(trap(x, *params))
        raise ValueError(f"Unknown MF type: {typ}")

# Inputs (exactly as in your 2.2.3)
MetalContent = FuzzyVariable(
    "Metal Content (%)", 0, 100, {
        "Low":    ("trap", (0, 0, 20, 40)),
        "Medium": ("tri",  (30, 50, 70)),
        "High":   ("trap", (60, 80, 100, 100)),
    }
)
Contamination = FuzzyVariable(
    "Contamination (0–100)", 0, 100, {
        "Low":      ("tri", (0, 0, 35)),
        "Moderate": ("tri", (25, 50, 75)),
        "High":     ("tri", (65, 100, 100)),
    }
)
Repairability = FuzzyVariable(
    "Repairability (0–10)", 0, 10, {
        "Poor":     ("tri", (0, 0, 4)),
        "Moderate": ("tri", (3, 5, 7)),
        "Good":     ("tri", (6, 10, 10)),
    }
)
Hazard = FuzzyVariable(
    "Hazard (0–10)", 0, 10, {
        "Safe":      ("trap", (0, 0, 2, 3)),
        "Risky":     ("tri",  (2, 5, 8)),
        "Hazardous": ("trap", (7, 8.5, 10, 10)),
    }
)

# 3) Rule base (13 rules)
@dataclass
class Rule:
    antecedents: List[Tuple[FuzzyVariable, str]]  # e.g., [(Hazard,"Hazardous"), (Contamination,"High")]
    consequent: str                               # label in OUTPUT_CONSTANTS
    def fire(self, inputs: Dict[str, float]) -> float:
        vals = [var.mu(lbl, inputs[var.name]) for var, lbl in self.antecedents]
        return float(min(vals)) if vals else 0.0  # AND=min

RULES: List[Rule] = [
    # Safety-first
    Rule([(Hazard, "Hazardous")], "Hazardous Handling"),
    Rule([(Hazard, "Risky"), (Contamination, "High")], "Hazardous Handling"),
    Rule([(Contamination, "High"), (Repairability, "Poor")], "Safe Disposal"),
    # Value-recovery
    Rule([(MetalContent, "High"), (Contamination, "Low")], "Material Recovery"),
    Rule([(MetalContent, "High"), (Hazard, "Safe")], "Material Recovery"),
    Rule([(MetalContent, "Medium"), (Repairability, "Poor"), (Hazard, "Safe")], "Material Recovery"),
    # Reuse
    Rule([(Repairability, "Good"), (Contamination, "Low"), (Hazard, "Safe")], "Refurbish/Reuse"),
    Rule([(Repairability, "Moderate"), (Contamination, "Low"), (Hazard, "Safe")], "Refurbish/Reuse"),
    # Tie-breakers
    Rule([(Repairability, "Moderate"), (MetalContent, "Medium"), (Contamination, "Moderate")], "Material Recovery"),
    Rule([(Repairability, "Poor"), (MetalContent, "Low"), (Contamination, "Moderate")], "Safe Disposal"),
    Rule([(Hazard, "Risky"), (MetalContent, "High")], "Hazardous Handling"),
    # Contamination-led
    Rule([(Contamination, "High"), (MetalContent, "Medium"), (Hazard, "Safe")], "Material Recovery"),
    Rule([(Contamination, "Moderate"), (Repairability, "Good"), (Hazard, "Risky")], "Hazardous Handling"),
]
