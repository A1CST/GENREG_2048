# ================================================================
# GENREG Evolved Encoder — Perception Layer
# ================================================================
# A single-layer encoder with an EVOLVED activation function that
# sits between raw board signals and the controller.
#
# Instead of the controller seeing noisy raw data, it sees a
# filtered representation shaped by evolution.
#
# Each genome evolves:
#   - Encoder weights (what patterns to extract from the board)
#   - Activation selection (which activation shape to use)
#   - Activation parameters (how to tune that shape)
#
# The activation catalog contains diverse nonlinearities.
# Evolution picks the one that best filters signal from noise
# for each genome's strategy.
# ================================================================

import math
import random
import copy
import numpy as np

# ================================================================
# ACTIVATION CATALOG
# ================================================================
# Each activation is a function f(x, params) -> y
# params is a dict of named floats that evolve per-genome.
#
# These are drawn from common activation research + GENREG's
# evolved activation discoveries. Each one "sees" the input
# differently — evolution picks the best lens.
# ================================================================

def act_tanh_scaled(x, p):
    """Classic tanh with evolved scale and shift."""
    return p["alpha"] * math.tanh(p["beta"] * x + p["gamma"])

def act_gated_linear(x, p):
    """Linear signal gated by a sigmoid — passes important signals, kills noise."""
    sig = 1.0 / (1.0 + math.exp(-max(-20, min(20, p["gate"] * x))))
    return p["scale"] * x * sig

def act_soft_threshold(x, p):
    """Soft step function — outputs ~0 below threshold, ~1 above.
    Good for binary features like 'is there a high tile here?'"""
    v = p["sharpness"] * (x - p["threshold"])
    v = max(-20, min(20, v))
    return p["scale"] / (1.0 + math.exp(-v))

def act_resonance(x, p):
    """Sine-based — creates periodic response. Can detect repeating
    patterns or create multi-modal sensitivity."""
    return p["amp"] * math.sin(p["freq"] * x + p["phase"])

def act_dual_path(x, p):
    """Two parallel paths blended: a bounded tanh path and a gated
    linear path. Most expressive — can approximate any of the above."""
    tanh_path = p["w_tanh"] * math.tanh(p["s_tanh"] * x)
    lin_x = p["s_lin"] * x
    sig = 1.0 / (1.0 + math.exp(-max(-20, min(20, lin_x))))
    lin_path = p["w_lin"] * x * sig
    return tanh_path + lin_path

def act_abs_gate(x, p):
    """Magnitude detector — responds to how far from zero, not direction.
    Good for detecting 'any tile present' regardless of value."""
    return p["scale"] * (1.0 - math.exp(-p["rate"] * abs(x)))

def act_quadratic_relu(x, p):
    """Quadratic growth above threshold, zero below. Amplifies
    strong signals, ignores weak ones."""
    v = x - p["threshold"]
    if v <= 0:
        return 0.0
    return p["scale"] * v * v

def act_identity_plus(x, p):
    """Near-identity with a learnable nonlinear nudge.
    Lets the raw signal through but adds evolved structure."""
    return x + p["nudge"] * math.tanh(p["bend"] * x)


# Catalog: (function, param_names, default_values, bounds)
ACTIVATION_CATALOG = [
    (act_tanh_scaled,    {"alpha": 1.0, "beta": 1.0, "gamma": 0.0},
                          {"alpha": (-3.0, 3.0), "beta": (0.1, 5.0), "gamma": (-2.0, 2.0)}),
    (act_gated_linear,   {"gate": 1.0, "scale": 1.0},
                          {"gate": (0.1, 5.0), "scale": (-3.0, 3.0)}),
    (act_soft_threshold, {"sharpness": 3.0, "threshold": 0.5, "scale": 1.0},
                          {"sharpness": (0.5, 10.0), "threshold": (-1.0, 2.0), "scale": (-3.0, 3.0)}),
    (act_resonance,      {"amp": 1.0, "freq": 2.0, "phase": 0.0},
                          {"amp": (-2.0, 2.0), "freq": (0.5, 8.0), "phase": (-3.14, 3.14)}),
    (act_dual_path,      {"w_tanh": 0.5, "s_tanh": 1.0, "w_lin": 0.5, "s_lin": 1.0},
                          {"w_tanh": (-2.0, 2.0), "s_tanh": (0.1, 5.0),
                           "w_lin": (-2.0, 2.0), "s_lin": (0.1, 5.0)}),
    (act_abs_gate,       {"scale": 1.0, "rate": 2.0},
                          {"scale": (-3.0, 3.0), "rate": (0.1, 8.0)}),
    (act_quadratic_relu, {"threshold": 0.0, "scale": 1.0},
                          {"threshold": (-1.0, 1.0), "scale": (0.1, 5.0)}),
    (act_identity_plus,  {"nudge": 0.3, "bend": 1.0},
                          {"nudge": (-1.0, 1.0), "bend": (0.1, 5.0)}),
]

NUM_ACTIVATIONS = len(ACTIVATION_CATALOG)


# ================================================================
# ENCODER (CPU — for inference)
# ================================================================
class GENREGEncoder:
    """
    Evolved perception layer. Single linear transform + evolved activation.

    input_dim  → encoder_dim  (linear, evolved weights)
                             (evolved activation applied element-wise)

    Each genome stores:
      - enc_w:  (encoder_dim, input_dim) weight matrix
      - enc_b:  (encoder_dim,) bias vector
      - act_id: which activation from the catalog
      - act_params: dict of params for that activation
    """

    def __init__(self, input_dim, encoder_dim=32):
        self.input_dim = input_dim
        self.encoder_dim = encoder_dim

        # Random init
        self.enc_w = [[random.uniform(-0.5, 0.5) for _ in range(input_dim)]
                       for _ in range(encoder_dim)]
        self.enc_b = [random.uniform(-0.1, 0.1) for _ in range(encoder_dim)]

        # Random activation from catalog
        self.act_id = random.randint(0, NUM_ACTIVATIONS - 1)
        func, defaults, bounds = ACTIVATION_CATALOG[self.act_id]
        self.act_params = dict(defaults)
        self.act_bounds = dict(bounds)
        # Per-neuron activation params (preserves GPU-evolved diversity).
        # List of dicts, one per encoder neuron.
        # Initialised from the shared defaults so legacy code paths still work.
        self.act_params_per_neuron = [dict(defaults) for _ in range(encoder_dim)]

    def forward(self, raw_signals):
        """
        raw_signals: list[float] of length input_dim
        returns: list[float] of length encoder_dim
        """
        func = ACTIVATION_CATALOG[self.act_id][0]
        output = []
        # Prefer per-neuron params if available (evolved GPU state)
        use_per_neuron = (
            hasattr(self, "act_params_per_neuron")
            and self.act_params_per_neuron is not None
            and len(self.act_params_per_neuron) == self.encoder_dim
        )
        for i in range(self.encoder_dim):
            s = self.enc_b[i]
            for j in range(self.input_dim):
                s += self.enc_w[i][j] * raw_signals[j]
            p = self.act_params_per_neuron[i] if use_per_neuron else self.act_params
            output.append(func(s, p))
        return output

    def clone(self):
        new = GENREGEncoder(self.input_dim, self.encoder_dim)
        new.enc_w = copy.deepcopy(self.enc_w)
        new.enc_b = copy.deepcopy(self.enc_b)
        new.act_id = self.act_id
        new.act_params = dict(self.act_params)
        new.act_bounds = dict(self.act_bounds)
        if hasattr(self, "act_params_per_neuron") and self.act_params_per_neuron is not None:
            new.act_params_per_neuron = [dict(p) for p in self.act_params_per_neuron]
        return new

    def mutate(self, rate=0.05, scale=0.1):
        """Mutate weights and activation params."""
        # Weights
        for i in range(self.encoder_dim):
            for j in range(self.input_dim):
                if random.random() < rate:
                    self.enc_w[i][j] += random.gauss(0, scale)
        for i in range(self.encoder_dim):
            if random.random() < rate:
                self.enc_b[i] += random.gauss(0, scale)

        # Activation params
        for key, val in self.act_params.items():
            if random.random() < rate:
                lo, hi = self.act_bounds[key]
                self.act_params[key] = max(lo, min(hi, val + random.gauss(0, scale * 0.5)))

        # Small chance to switch activation entirely (exploration)
        if random.random() < rate * 0.1:
            self.act_id = random.randint(0, NUM_ACTIVATIONS - 1)
            _, defaults, bounds = ACTIVATION_CATALOG[self.act_id]
            self.act_params = dict(defaults)
            self.act_bounds = dict(bounds)
