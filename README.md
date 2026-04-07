[FINDINGS.md](https://github.com/user-attachments/files/26524644/FINDINGS.md)
# GENREG 2048 — Findings

Experimental results from training GENREG on the 2048 game, April 2026.

---

## 1. Landscape Design Drives Results

Every major improvement came from redesigning the fitness signal, not the network or the optimizer.

**Proximity bonus direction fix:** The original proximity bonus rewarded 512-tile genomes based on the score threshold for *reaching* 512 (3500). Once a genome scored above 3500, the landscape went flat — no gradient toward 1024. Fixing it to point at the *next* tile's threshold (7000) gave evolution a continuous signal to climb. This single change was necessary (but not sufficient) for reaching 1024.

**Move cap removal:** Training games were hard-capped at 530 moves. At inference, the best genomes needed 1000+ moves to reach 1024. The model was never allowed to play long enough to discover 1024-level strategies during training. Raising the cap to 3000 unblocked the path.

**Checkpoint serialization bugs:** Three bugs in checkpoint save/load silently destroyed trained state every session:
1. Per-neuron activation parameters collapsed to scalar (only neuron 0 survived)
2. Activation function switches didn't reset parameters to the new function's defaults
3. Save/load didn't persist per-neuron parameter lists

Each created the appearance of a training plateau. The model was rediscovering its peak performance from a collapsed state every session. After fixing all three, progress accumulated across checkpoint cycles.

## 2. Current Performance

GENREG V3-V5 with the 1024 config reliably reaches 512 with 70-80% of the population. 1024 tiles appear within a few hundred generations. The population has not yet converged on 1024 consistently — it appears in 2-5% of genomes per generation, and best single-game scores reach 7,000-8,500.

2048 has not been reached in training.

For comparison, the fully-optimized DQN baseline (938,885 parameters, CNN, action masking, hand-crafted reward shaping) reached a best tile of 1024 with average score 3,636 in the same time budget. GENREG reaches the same best tile with ~1,929 parameters — 487x fewer.

## 3. Architecture Is Not the Bottleneck

A controlled sweep tested hidden dimensions of 8, 16, and 32 (2000 generations each, 3 trials):

| Dims | 1024 count | >=512% | Avg Score | First 1024 |
|-----:|----------:|-------:|----------:|-----------:|
| 8 | 14 | 69% | 3,296 | gen 87 |
| 16 | 14 | 65% | 3,262 | gen 261 |
| 32 | 5 | 59% | 2,964 | gen 457 |

The 1024 plateau is identical across architectures. Bigger networks converge slower. The 1024 config's hidden_size=8 was already optimal.

## 4. Energy Economy Is a Factor

Diagnostic runs showed the best genomes have a ~40% invalid move rate. Each invalid move costs -2 energy. Nearly half the model's energy budget is wasted pressing directions that don't work.

Testing reduced invalid move penalties (0.5, 1.0, 2.0):

| Penalty | >=512% at gen 1000 | 1024 count | First 1024 |
|--------:|-------------------:|:----------:|:----------:|
| 2.0 | 69% | 14 | gen 87 |
| 1.0 | 55% | ~10 | gen 50 |
| 0.5 | 48% | ~5 | gen 50 |

Lower penalty makes 1024 appear faster but the population doesn't consolidate as well. The original penalty of 2.0 produces the healthiest population overall. The model needs to learn energy efficiency, not be given it for free.

## 5. Evolved Reproduction (V4)

When genomes are given evolvable control over their own mutation parameters (mut_rate, mut_scale, exploration drive), the population self-organizes:

- Mutation rate drops from 0.065 to ~0.016 (fewer mutations per child)
- Mutation scale stays moderate at ~0.14 (decent-sized mutations when they happen)
- Exploration drive initially drops to 0.006, then recovers to ~0.03 when 1024 genomes enter the elite pool

The population discovers a strategy of "fewer but bigger bets" rather than constant small noise. Rare explorers that break through to 1024 get ratchet protection, and their higher-exploration traits propagate.

Trust protein parameters (gain, scale, decay) must be excluded from per-genome mutation control. If genomes can control their own trust mutation rate, they evolve to inflate their fitness signal rather than improve gameplay.

## 6. Crossover (V5)

V5 adds neuron-level crossover between elite parents. The crossover probability is itself an evolvable trait — genomes that benefit from crossover evolve higher rates, those that don't evolve it to zero. Early results show the population self-adjusting crossover rates. Whether crossover helps or hurts over long training runs is still an open question.

## 7. Key Bugs Found

| Bug | Effect | Fix |
|-----|--------|-----|
| Per-neuron params collapsed to scalar on save | Every checkpoint destroyed evolved activation diversity | Save/load all neurons |
| Activation switch didn't reset params | Genomes had mismatched activation function + parameters | Reset params on switch |
| save_checkpoint dropped act_params_per_neuron | Load rebuilt per_neuron from wrong activation defaults | Persist in checkpoint |
| Move cap = starting_energy + 500 | Games killed at 530 moves, 1024 needs 1000+ | Cap at 3000 |
| Proximity bonus used current tile threshold | No gradient from 512 toward 1024 | Use next tile threshold |

**Lesson:** Before diagnosing a plateau as a landscape problem, verify checkpoint round-trips preserve trained behavior.

## 8. What's Next

The path to 2048 likely requires one or more of:
- Sustained improvement in move efficiency (reducing the 40% invalid rate)
- More training time for the population to consolidate at 1024 before pushing higher
- Possibly spatial awareness in the input encoding (the controller sees a flat vector, not a 2D grid)
- V5 crossover may provide the genetic diversity needed for the 1024-to-2048 jump — still being evaluated

---

*Findings documented April 2026. All experiments reproducible from the code in this repository.*
