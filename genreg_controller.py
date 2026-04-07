# ================================================================
# GENREG v1.0 — Neural Controller (Functional Genome)
# ================================================================
# A small forward-pass-only neural network.
# - No backprop
# - No gradients
# - Weights evolve by mutation
#
# Inputs:  environment signals (floats)
# Outputs: 4 discrete actions (UP/DOWN/LEFT/RIGHT)
# ================================================================

import random
import math
import copy


# ---------------------------------------------------------------
# Activation functions (simple, stable, mutation-friendly)
# ---------------------------------------------------------------
def relu(x):
    return x if x > 0 else 0


def tanh(x):
    return math.tanh(x)


# ================================================================
# GENREG Neural Controller
# ================================================================
class GENREGController:
    def __init__(self, input_size, hidden_size=16, output_size=4):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Randomly initialize weights
        self.w1 = [[random.uniform(-0.5, 0.5) for _ in range(input_size)]
                   for _ in range(hidden_size)]

        self.b1 = [random.uniform(-0.1, 0.1) for _ in range(hidden_size)]

        self.w2 = [[random.uniform(-0.5, 0.5) for _ in range(hidden_size)]
                   for _ in range(output_size)]

        self.b2 = [random.uniform(-0.1, 0.1) for _ in range(output_size)]

    # ------------------------------------------------------------
    def clone(self):
        """Deep copy controller and weights."""
        new = GENREGController(self.input_size, self.hidden_size, self.output_size)
        new.w1 = copy.deepcopy(self.w1)
        new.w2 = copy.deepcopy(self.w2)
        new.b1 = copy.deepcopy(self.b1)
        new.b2 = copy.deepcopy(self.b2)
        return new

    # ------------------------------------------------------------
    def mutate(self, rate=0.05, scale=0.1):
        """Gaussian mutation across all weights."""
        def mutate_matrix(mat):
            for i in range(len(mat)):
                for j in range(len(mat[i])):
                    if random.random() < rate:
                        mat[i][j] += random.gauss(0, scale)

        def mutate_vector(vec):
            for i in range(len(vec)):
                if random.random() < rate:
                    vec[i] += random.gauss(0, scale)

        mutate_matrix(self.w1)
        mutate_matrix(self.w2)
        mutate_vector(self.b1)
        mutate_vector(self.b2)

    # ------------------------------------------------------------
    def forward(self, inputs):
        """
        inputs: list[float]
        returns: action index (0-3)
        """

        # Hidden layer
        hidden = []
        for i in range(self.hidden_size):
            s = self.b1[i]
            for j in range(self.input_size):
                s += self.w1[i][j] * inputs[j]
            hidden.append(tanh(s))   # stable activation

        # Output layer
        outputs = []
        for i in range(self.output_size):
            s = self.b2[i]
            for j in range(self.hidden_size):
                s += self.w2[i][j] * hidden[j]
            outputs.append(s)

        # Pick action with max activation
        action = max(range(self.output_size), key=lambda i: outputs[i])
        return action
