# GENREG_2048
GENREG attempt to beat 2048
[FINDINGS.md](https://github.com/user-attachments/files/26524571/FINDINGS.md)
# GENREG — Consolidated Findings

Experimental results from development and testing of the GENREG neuroevolution framework, April 2026.

---

## 1. Landscape Design Drives Results

The single most impactful lever in every experiment was the fitness landscape — not the architecture, not the optimizer, not the hyperparameters.

**Multi-number regression:** The same architecture (256 → 64 → 32 → 1) with the same population (500) and same mutation rate was tested on learning 100 distinct input-output mappings. With summed fitness (broken landscape), 0 of 100 targets were learned. With geometric mean fitness (fixed landscape), 94 of 100 targets were within 1.0 of the correct value. The only change was the fitness function.

**CIFAR-10 classification:** Four iterations of landscape design on the same 402K-parameter architecture improved test accuracy from 12.2% to 29.2%. Changes included EMA-smoothed fitness, class-balanced sampling, margin-based scoring, and a mutation rate floor. No architecture modifications.

**2048 training:** Three landscape bugs (checkpoint serialization, move cap, proximity direction) were each individually responsible for preventing the model from reaching 1024. Fixing all three took the system from "never hit 1024 in 43,000 generations" to "1024 at generation 87."

## 2. What GENREG Can and Cannot Learn

**Learnable:**
- 2048 gameplay to 1024 tile level (1,929 parameters, gradient-free)
- Structured regression with 0.99 test correlation on unseen data
- XOR parity above random at all bit widths tested (2-64)
- Constraint satisfaction with zero gradient (6 hard step functions, solved in 2 generations)
- Discrete multi-step reasoning (count → primality test → conditional branch, 2.5x random baseline)
- Hash approximation for functions with detectable structure (XOR fold: +2.3 bits)

**Not learnable:**
- SHA-256 or any hash with proper cryptographic diffusion
- XOR parity above ~57% for bit widths > 4 (representational ceiling, not landscape)
- 2048 tile in training (reached 1024 but not 2048)

## 3. Evolved Perception (V3)

Adding an evolved encoder between the raw board signals and the controller produced the largest architectural improvement. The encoder selects from 8 activation functions and tunes their parameters per-genome.

Key observation: after training, the population typically converges on 1-2 activation functions. On the 2048 task, quadratic ReLU dominated (500/500 genomes by generation 250). On other tasks, different activations win. The population discovers the right mathematical lens for the problem.

V3 compressed time to first milestones dramatically compared to V1.

## 4. Self-Organizing Reproduction (V4)

When genomes are given control over their own mutation parameters, the population self-organizes toward a specific reproductive strategy: low mutation frequency but moderate mutation magnitude, with very low exploration (jump mutation) probability.

This means the population prefers "fewer but bigger bets" over "constant small noise." The explore parameter initially drops near zero as exploiters dominate, then partially recovers when rare breakthrough genomes (e.g., those reaching 1024) get ratchet protection and enter the elite pool.

Trust protein parameters (gain, scale, decay) must be excluded from per-genome mutation control. If genomes can control their own trust mutation rate, they evolve to inflate their fitness signal rather than improve gameplay.

## 5. Gradient vs Evolution: The 10-Task Comparison

Both gradient descent (Adam) and GENREG were tested on 10 tasks of varying smoothness:

| Task | Type | Gradient | GENREG | Winner |
|------|------|----------|--------|--------|
| Linear regression | Smooth | -0.71 | -0.03 | GENREG* |
| Polynomial regression | Smooth | -0.03 | -0.17 | Gradient |
| 10-class classification | Smooth | 1.00 | 0.88 | Gradient |
| XOR parity 8-bit | Non-differentiable | 0.55 | 0.62 | GENREG |
| Sorting 8 numbers | Permutation | 0.46 | 0.31 | Gradient |
| Modular arithmetic | Non-differentiable | 0.16 | 0.22 | GENREG |
| Constraint satisfaction | Step functions | 1.00 | 1.00 | Tie** |
| Discrete reasoning | Non-differentiable | 0.24 | 0.27 | GENREG |
| Permutation learning | Combinatorial | 0.03 | 0.09 | GENREG |
| Time series | Semi-smooth | 1.00 | 0.95 | Gradient |

*Linear regression result is a compute-budget artifact. **Constraint tie occurred because gradient was given a smooth surrogate loss.

The split is clean: gradient wins on smooth differentiable tasks, GENREG wins on discrete/non-differentiable tasks. Neither is categorically superior.

## 6. The Parity Ceiling Is Representational

XOR parity was tested with:
- 8 different landscape designs (all plateau at 54-57%)
- 9 architecture configurations from 4K to 263K parameters (62x range, 3% accuracy change)
- 6 bit widths from 2 to 64 (performance plateaus above 4 bits)

The 2-bit case is nearly solved (92%), confirming the architecture CAN represent XOR. The ceiling at higher bit widths is an interaction between the MLP's representational capacity and the combinatorial difficulty of XOR at scale. No landscape redesign or architecture scaling broke through.

## 7. Checkpoint Integrity Matters

Three serialization bugs were discovered that silently destroyed trained state on every save/load cycle:

1. **Per-neuron activation parameters** were collapsed to scalar values (only neuron 0 survived)
2. **Activation function switches** didn't reset parameters to the new function's defaults
3. **Save/load** didn't persist per-neuron parameter lists

Each bug created the appearance of a training plateau. The model was continuously rediscovering its peak performance from a collapsed starting state rather than building on previous generations. After fixing all three, training progress accumulated cleanly across checkpoint cycles.

**Lesson:** Before diagnosing a training plateau as a fitness landscape problem, verify that checkpoint round-trips preserve the trained behavior.

## 8. Open Questions

- Can 2048 be reached with the current architecture? The 1024-to-2048 gap may require either more network capacity, a fundamentally different input representation (spatial/CNN), or a multi-stage training approach.
- Does crossover (V5) help or hurt? Early results show the population self-adjusting crossover rates. Whether it converges to zero (crossover doesn't help) or stabilizes (crossover provides useful genetic diversity) is an open empirical question.
- Can the trust protein system be validated on tasks with genuine temporal structure? The Find 42 testbed was too simple. Game environments with sustained play (2048, Snake) are where temporal trust should show its advantage, but a controlled comparison against snapshot fitness on those tasks has not been completed.

---

*Findings documented April 2026. All experiments are reproducible from the code in this repository.*
