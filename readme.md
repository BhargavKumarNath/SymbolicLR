# SymboLR: Symbolic Learning Rate Discovery

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.1+](https://img.shields.io/badge/pytorch-2.1%2B-orange.svg)](https://pytorch.org/)
[![Rust](https://img.shields.io/badge/rust-pyo3%2Fmaturin-red.svg)](https://pyo3.rs/)
[![Streamlit](https://img.shields.io/badge/dashboard-streamlit-ff4b4b.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Live Demo](https://img.shields.io/badge/demo-streamlit%20cloud-brightgreen)](https://your-app.streamlit.app)
 
> A Quality-Diversity Genetic Programming system that autonomously evolves symbolic mathematical learning rate schedules for neural networks, benchmarks them against production baselines, and renders discovered formulas as human-readable LaTeX via a live Streamlit analysis dashboard.

---

## The Problem

The learning rate schedule is the single most consequential hyperparameter in neural network training. A well-shaped schedule, one that warms up early, decays gracefully, and avoids stagnation, can be the difference between a model that converges in 20 epochs and one that never converges at all. Despite this, practitioners choose from a small, historically-accumulated set of hand-crafted functions: cosine annealing, step decay, warm restarts, 1-cycle. These are reasonable, but they represent an infinitesimally small slice of all possible mathematical schedules.
 
The question SymboLR asks is: what if you could search the full space of valid mathematical expressions and let the training signal itself tell you which shape is optimal?
 
This is the symbolic regression framing. The schedule `η(t)` is not a fixed functional form with tunable constants. It is an unknown mathematical expression built from primitive operations `{+, -, *, /, sin, cos, exp, log, sqrt, abs}` and the normalized training time variable `t ∈ [0, 1]`. The search problem is to find the expression that minimizes validation loss when used as a learning rate schedule during probe training. SymboLR solves this using Genetic Programming, with MAP-Elites as the selection mechanism to prevent premature convergence and maintain a behaviorally diverse archive of solutions.

---

## System Architecture

The pipeline is sequential but the critical inner loop, fitness evaluation, is parallelized across threads. At a high level:

```
Ramped H&H Init
      |
      v
Parallel Fitness Evaluation  <--------------------+
  (Rust AST eval + GPU probe training)            |
      |                                           |
      v                                           |
MAP-Elites Archive (try_add)                      |
      |                                           |
      v                                           |
Sample Parents (uniform over niches)              |
      |                                           |
      v                                           |
Genetic Operators                                 |
  50% Subtree Crossover                           |
  25% Subtree Mutation                            |
  25% Hoist Mutation (anti-bloat)                 |
      |                                           |
      v                                           |
SymPy Simplification (lazy, pre-insertion)        |
      |                                           |
      +------------------------------------------+
      |
      v (after final generation)
L-BFGS-B Constant Refinement (Hall of Fame)
      |
      v
LaTeX Output + Dashboard
```

Each of these components has specific rationale. The sections below explain not just what each component does but why it was designed the way it is.

---

## Genetic Programming and the AST Representation

Think of Genetic Programming (GP) as treating mathematical formulas like miniature computer programs. Instead of parsing a formula like `0.5 * (1 - t)` as a plain string, the system structures it as an Abstract Syntax Tree (AST).

In this tree, the multiplication operator acts as the root, its left branch holds the constant `0.5`, and its right branch points to a subtraction operator splitting into `1.0` and `t`. Under the hood, every single piece of this tree is just an instance of the `Node` class found in `gp/tree.py`.

Each `Node` tracks three core attributes:

* **A value:** This can be an operator name, the variable `'t'`, or a float constant.
* **Children:** A list pointing to the nodes directly underneath it.
* **An MD5 hash:** A cached fingerprint of the node's structure.

That cached hash is a massive performance lifesaver. Whenever a genetic operation alters a subtree, the system runs `invalidate_cache()` to clear out old hashes. When a new formula is generated, the system checks this hash first. If the new formula matches an identical twin already sitting in the archive, the cache comparison short-circuits the entire process. This saves you from burning time re-running the Rust evaluator or spinning up a GPU training loop for a formula you have already seen.

Over in `gp/operators.py`, the system relies on a curated set of 10 primitive operators, divided into binary and unary types. Every operator has a specific job:

* **Standard arithmetic** builds linear and polynomial schedules.
* `sin` and `cos` handle periodic oscillations.
* `exp` and `log` manage exponential warmup and decay.
* `sqrt` allows for sub-linear growth.

Because GP can generate some wild, unpredictable expressions, every operator includes built-in safety guards to prevent the whole system from crashing during evaluation:

* **Division** returns `1.0` if the denominator drops below `1e-6`.
* **Log** wraps its input in `max(|x|, 1e-6)` to gracefully handle zero or negative numbers.
* **Sqrt** uses absolute values (`|x|`) to dodge negative input errors.
* **Exp** clamps its arguments between -100 and 10 so it does not explode into infinity.

These safety guards have to be perfectly mirrored across both the Python and Rust evaluators. The test suite in `test_rust_core.py` strictly checks that both backends match down to a `1e-7` tolerance. If they diverge even slightly, the system would calculate different fitness scores depending on which backend was running, completely corrupting the data archive.

### Initial Population: Ramped Half-and-Half

To kick things off, the system builds the very first generation of formulas using the ramped half-and-half method inside `gp/population.py`. It splits the population evenly across different depth levels, starting from a `min_depth` of 2 up to a `max_depth` of 4 if you are running in dashboard mode, or 5 if you are on the CLI.

Within each of those depth levels, the population is split down the middle into two distinct strategy camps:

* **The `full` method (50%):** This forces every single branch to grow all the way out to the maximum allowed depth, giving you perfectly symmetric, deep trees.
* **The `grow` method (50%):** This introduces a 50% chance of placing a terminal (a constant or variable) at any given node. This naturally cuts branches short, producing asymmetric trees with highly variable depths.

All of this structural engineering comes down to one goal: maximizing diversity right at initialization.

A classic way Genetic Programming runs off the rails is when the initial population looks too much alike. If the starting pool is structurally homogeneous, the evolutionary operators just keep recombining the same basic shapes. The system essentially gets stuck in a creative rut and quickly hits a local optimum.

By using ramped half-and-half, you guarantee that generation zero is incredibly diverse before a single fitness evaluation even runs. It ensures you have a healthy mix of shallow, simple formulas which usually generalize much better alongside deep, complex formulas that can hunt down non-obvious patterns in the search space.

---

## MAP-Elites: Quality-Diversity Over Fitness-Only Selection

Standard evolutionary algorithms rank populations purely by fitness. While this works for single-optimum problems, learning rate discovery is highly multimodal. A periodic formula, a monotonic decay, and a warmup-then-decay curve can all achieve identical validation loss despite being structurally unrelated. Selecting by fitness alone destroys this structural diversity.

To fix this, MAP-Elites (in `gp/map_elites.py`) maps individuals into a 2D grid of behavioral niches. Each cell holds just one elite individual: the top performer for that specific behavior. The archive uses a grid of 30 complexity bins by 20 center-of-mass bins, offering up to 600 unique niches.

The two behavioral axes are tailored specifically for learning rate schedules:

* **Tree Complexity (30 bins):** Tracks total AST node count. This axis separates compact formulas (like a 2-node `sin(t)`) from complex, multi-term schedules (like a 15-node polynomial warmup with cosine decay). Compact formulas often generalize better, while complex ones can exploit dataset-specific geometry.
* **Center of Mass (20 bins):** Calculated as `Σ(t * LR(t)) / Σ(LR(t))`. This measures the curve's weighted temporal centroid over normalized time. A value near 0 concentrates learning early (warmup), while a value near 1 concentrates it late (cooldown). This isolates the schedule's timing profile from its mathematical form.

Niche insertion is strictly competitive. A new formula only takes a cell if the cell is empty or if the new formula beats the incumbent's validation loss.

For reproduction, parent selection samples uniformly from all occupied niches rather than weighting by fitness. Because a rare structural formula has the exact same chance of being selected as the global leader, the archive avoids being overrun by slight variations of a single dominant strategy.

---

## The Genetic Operators

Every offspring pair is generated using one of three operators, with probabilities managed in `config/settings.py`:

* **Subtree Crossover (50%):** Deep copies both parents, grabs a random node from each tree via `_get_all_nodes`, and swaps their entire subtrees in place. If either new tree crosses the `max_depth=7` limit, the operation aborts and returns the original parent unchanged. This acts as a hard ceiling to stop unbounded tree growth.
* **Subtree Mutation (25%):** Deep copies a single parent, picks a random node, and swaps it out for a brand-new random subtree up to depth 4. This is crucial for injecting fresh structural novelty that crossover cannot create on its own.
* **Hoist Mutation (25%):** Selects a random internal operator node and replaces it entirely with one of its own descendants. Because a subtree is replaced by a smaller piece of itself, the resulting tree is guaranteed to shrink. This serves as the primary anti-bloat mechanism, preventing formulas from continuously ballooning into complex but meaningless shapes.

No matter which operator runs, it must call `_clear_all_caches()` on the new offspring before returning them. This step is vital for correctness because the system uses the `Node.fitness` attribute as a first-class cache. If a child inherits a stale fitness value from a parent, the evaluation loop will short-circuit and return wrong data. Clearing the cache ensures every fitness score accurately reflects the new formula.

---

## Fitness Evaluation: Real Training and Synthetic Simulation

The fitness function in `gp/fitness.py` routes to one of two evaluation modes depending on the detected runtime environment.

### Real GPU Training

In GPU mode, each candidate formula is evaluated by training a `FastConvNet` probe model from scratch using the formula as the learning rate schedule. The probe model is a lightweight two-layer CNN:
 
```
Input
  Conv2d(in_channels, 32) + BatchNorm + ReLU + MaxPool
  Conv2d(32, 64) + BatchNorm + ReLU + MaxPool
  AdaptiveAvgPool2d(4x4)
  Linear(64*16, 128) + ReLU
  Linear(128, 10)
```

This architecture is intentionally small. The goal isn't to hit state-of-the-art accuracy, but to produce a clean fitness signal that reflects how well a learning rate schedule navigates the loss landscape. Going with a deeper model would just slow down training and inflate generation wall time without actually improving the quality of that signal. To wring out extra performance, `FastConvNet` is compiled via `torch.compile(mode="reduce-overhead")` on non-Windows platforms. This fuses small GPU kernels via Triton, cutting down the CPU overhead that usually dominates short training runs.

Over in `models/probe.py`, the `ProbeTrainer` wraps the training loop in several safety guards tailored for GP workloads. Because arbitrary genetic formulas can output values that are NaN, infinite, negative, or astronomically large, the trainer validates the schedule at every single step. If it catches a bad value, it immediately halts and returns a fitness score of `inf`.

It keeps a close eye on the training loss too. If the loss passes the `explode_threshold` or turns non-finite, the run aborts with an `inf` score. There is also an early stopping mechanism with customizable `patience` and `min_delta` parameters to kill the run if validation loss plateaus, keeping you from wasting precious GPU cycles on formulas that fail to generalize.

Finally, when CUDA is available, the system enables Automatic Mixed Precision (AMP) using `torch.autocast` and `GradScaler`. This cuts the memory footprint roughly in half and doubles throughput for float16-compatible operations, which makes a massive difference when you are evaluating dozens of candidates at the exact same time inside a generation.

### Synthetic Simulation (Cloud Mode)

On Streamlit Cloud there is no GPU and no PyTorch. Rather than disabling evolution entirely, SymboLR implements a synthetic fitness function that simulates gradient descent on a 5-dimensional quadratic loss landscape:

```
L(w) = 0.5 * sum(curvatures * (w - w*)^2)
```

This setup uses heterogeneous curvatures `[0.5, 1.0, 2.0, 4.0, 8.0]` to mimic the ill-conditioning typical of real neural network loss surfaces. Each step applies a noisy SGD update using the candidate schedule's learning rate at that specific time step. The final fitness score is a weighted combination of the final validation loss and the best loss seen during the entire trajectory. It also actively penalizes constant schedules that lack `t` dependence, as well as schedules that continuously increase over time.

This approach creates incredibly realistic convergence dynamics. High-quality schedules drive final losses down into the `0.05-0.3` range, while overly aggressive schedules quickly diverge above `2.0`. Meanwhile, flat, constant schedules stagnate right around `1.5-2.0`. Ultimately, the fitness function creates a clear, meaningful differentiation between different schedule shapes, providing exactly the kind of signal MAP-Elites needs to make steady progress.

### Parallel Evaluation

The `ParallelEvaluator` in `gp/evaluator.py` wraps both evaluation modes in a `ThreadPoolExecutor`.

For GPU mode, this thread pool lets you overlap CPU-side Rust evaluation, data loading, and GPU training across multiple concurrent candidates. For synthetic mode, it simply parallelizes the NumPy simulation across your CPU cores.

The worker count is handled by `SymboLRConfig.resolve_workers()`, which enforces conservative VRAM caps to keep things stable. It limits execution to at most 1 concurrent worker for large GPU runs (epochs >= 5 or pop >= 100), at most 2 for medium runs, and at most 3 for small runs. These caps were determined empirically on an RTX 4070 with 8 GB VRAM to prevent out-of-memory crashes from simultaneous model instantiations.

---

## The Rust Evaluation Engine

The inner fitness loop evaluates each formula over a 100-step time array to produce a learning rate schedule. In Python, this requires traversing the AST recursively and calling NumPy operations at every single node. For a 10-node formula evaluated over 100 time steps, you end up triggering 10 recursive Python calls, 10 NumPy ufunc dispatches, and 10 intermediate array allocations for just one evaluation. Scale that up to a population size of 100 over 50 generations, and you are looking at 500,000 formula evaluations, each bogged down by Python interpreter overhead. This extra baggage is massive compared to the actual numerical work being done.

The Rust extension in `rust_core/src/lib.rs`, compiled via PyO3 and Maturin, completely eliminates this bottleneck. The Python side serializes the AST into a space-separated prefix string using `Node.to_prefix()`. For example, the tree `0.5 * (1 - t)` flattens into the string `"* 0.5 - 1.0 t"`. This serialization runs in O(n) time relative to the number of nodes and creates a flat string that crosses the Python-Rust boundary with zero memory copying.

Once on the **Rust side**, `parse_prefix` consumes the token iterator and builds a native `Expr` enum:

```rust
enum Expr {
    Var,
    Const(f64),
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Div(Box<Expr>, Box<Expr>),
    Sin(Box<Expr>),
    Cos(Box<Expr>),
    Exp(Box<Expr>),
    Log(Box<Expr>),
    Sqrt(Box<Expr>),
    Abs(Box<Expr>),
}
```

This recursive enum is a direct structural mirror of the Python `Node` class. The `eval(&self, t: f64) -> f64` method matches on the variant and computes a single scalar result. The evaluation loop then loops over the time array and calls `eval` once per time step, writing the results directly into a pre-allocated `Array1<f64>` from the `ndarray` crate.

This design means there are absolutely no intermediate array allocations. The finished array is returned to Python as a `PyArray1<f64>` using PyO3's zero-copy NumPy interop, which safely hands Python a direct view of memory owned by Rust without duplicating any data.

The numerical protections in the Rust evaluator perfectly mirror the ones in the Python operator registry. This alignment is strictly enforced: `test_rust_core.py` runs both evaluators on the exact same broken, pathological formulas (think taking the log of a negative number, dividing by near-zero, and triggering an overflowing exponent) and verifies the outputs with `np.testing.assert_allclose(atol=1e-7)`. If the safety logic drifts between the two implementations, this test fails immediately. This parity guarantee ensures the system never shuffles fitness rankings based on whether the Rust extension is active.

According to benchmarks on the dashboard's Rust Core page, this optimization delivers a massive performance boost. Throughput improvements over pure Python and NumPy range from roughly 10x for shallow formulas up to roughly 50x for deep, multi-operator expressions.

---

## VRAM-Resident Data Loading

Standard PyTorch `DataLoader` pipelines move data from CPU RAM to GPU VRAM every batch via the PCIe bus. For the low-fidelity tier, which uses 5% of MNIST (around 3,000 samples), the entire dataset takes up only about 9 MB of GPU VRAM at float32 precision. The custom `VRAMDataLoader` in `data/fidelity.py` takes advantage of this by pushing the whole dataset to the GPU once during initialization. It then generates batches by indexing into those GPU tensors directly.

The `FidelityManager._prepare_vram_split` pipeline applies stratified sampling at every stage using `sklearn.model_selection.train_test_split(stratify=targets)`. This ensures class balance is maintained through the subset slicing and the train/val split. The raw NumPy arrays are then stacked into contiguous tensors and transferred to the device via `.to(device, non_blocking=True)` in a single call at setup time. After this initial transfer, all batch operations are handled through pure GPU tensor slicing, removing PCIe traffic from the training loop.

When iterating, `VRAMDataLoader.__iter__` generates batch indices using `torch.randperm` directly on the GPU for device-side shuffling, yielding `(x[batch_idx], y[batch_idx])` slices. This turns each training iteration into a pure GPU operation. It eliminates the data movement overhead that would otherwise choke performance when evaluating multiple concurrent candidates.

Because index-based slicing can easily misalign data if handled incorrectly, a regression test named `test_prepare_vram_split_keeps_subset_images_and_labels_aligned` explicitly validates that image pixels and their labels stay perfectly paired after the subset operation.

---

## Algebraic Simplification via SymPy

Genetic Programming is notorious for producing bloated formulas. Successive crossover and mutation operations easily accumulate redundant structures. For instance, `(t + 0) * 1` is mathematically identical to `t`, but it takes up three extra nodes in the tree and forces the Rust evaluator to do unnecessary work. More importantly, when a formula looks complex because of accumulated noise, it can steal a behavioral niche that a genuinely compact formula with the same behavior could use more effectively.

To solve this, `gp/simplify.py` passes offspring through SymPy CAS before they are inserted into the archive. Here is how the pipeline works:

1. `_node_to_sympy` converts the `Node` tree into a SymPy expression using structural recursion. It handles n-ary SymPy `Add` and `Mul` operations by explicitly flattening the binary tree structure via a left-fold.
2. `sympy.nsimplify(tolerance=1e-4)` converts float constants into exact rationals or simple fractions. This step is crucial because it enables symbolic cancellation that standard `sympy.simplify` misses when dealing with imprecise floats.
3. `sympy.simplify` steps in to apply the full suite of algebraic reduction rules.
4. `_sympy_to_node` converts the simplified SymPy expression back into a standard `Node` tree. It handles SymPy's `Pow` forms, which represent division and roots as negative or fractional exponents, using explicit case matching.

This simplification process runs lazily. It only triggers for candidates that actually qualify to beat an incumbent in the archive, rather than running on every single offspring. This avoids burning CPU cycles on a single-threaded SymPy step for candidates that fail the competitive replacement test anyway.

Additionally, any tree larger than 50 nodes skips simplification entirely since the performance cost outweighs the benefit. The function also uses a fail-closed fallback: if SymPy runs into an unsupported structure, it simply returns the original, unsimplified tree so the GP loop can keep running without interruption.

Simplification works hand-in-hand with hoist mutation. While hoist mutation forces structural downsizing by swapping a subtree with one of its own descendants, SymPy clears out semantic clutter using pure algebraic identities. Together, they keep the archive filled with compact, high-signal formulas instead of massive, noise-inflated expressions.

---

## Hybrid Memetic Optimization

GP excels at structural search: finding the right arrangement of operators and the best topology for a formula. However, it struggles with numeric search. Once a structure is locked in, finding the precise constant values that minimize the fitness function is incredibly difficult for evolution alone. A formula like `C1 * exp(-C2 * t) + C3 * t` might have the perfect shape for a given dataset, but its GP-inherited constants (`C1=0.3, C2=0.7, C3=0.1`) are often far from optimal.

To bridge this gap, the top 5 Hall of Fame formulas undergo L-BFGS-B gradient descent on their scalar constants via `optimiser/hybrid.py` after the main evolution loop terminates. The process works through a tightly integrated loop:

1. `_get_constant_nodes` traverses each tree to collect all float terminal nodes by reference.
2. The SciPy `minimize` objective function injects proposed constant values directly into the live `Node` objects.
3. The system clears the fitness cache and calls the fitness function for a fresh evaluation.

Bounds are strictly enforced at `[1e-6, 10.0]` to keep the optimizer within a physically meaningful learning rate range. With `maxiter=15`, this local refinement typically converges in under a second per formula, consistently improving the final fitness scores by a few percentage points without altering the underlying structural form.

This hybrid approach combines the complementary strengths of evolutionary and gradient-based optimization: GP handles the global structural search across the discrete space of mathematical expressions, while L-BFGS-B takes care of the local numeric refinement within the continuous space of constant values.

---

## Baseline Comparison Pipeline

A core claim of SymboLR is that evolved schedules can compete directly with hand-crafted ones. Proving this requires a strict apples-to-apples comparison, meaning the exact same fitness function must evaluate both discovered and baseline schedules.

To handle this, `baselines/schedules.py` implements seven standard schedules as pure NumPy functions over normalized time `t ∈ [0, 1]`: Cosine Annealing, Step Decay, Warm Restarts (SGDR), Linear Decay, Constant LR, 1-Cycle, and Exponential Decay. Each schedule accepts a time array and returns a learning rate array, matching the precise interface expected by the fitness pipeline.

The evaluation runner in `optimiser/compare.py` pushes all seven baselines through `gp.fitness.evaluate_synthetic` (or the real GPU trainer in local mode), caches the results for the session, and structures the final comparison data for the dashboard's Results page.

As a result, the baseline comparison chart in the UI displays real, live-computed losses rather than static reference values. When you run even a short 5-generation evolution, the dashboard plots the evolved elite's actual validation loss right alongside all seven baselines evaluated under the exact same conditions.

---

## Cloud Deployment and the Synthetic Fitness Bridge

Streamlit Cloud enforces strict RAM limits and lacks GPU access entirely. The system manages this constraint through a three-way runtime detection mechanism inside `config/settings.py`:

* **`RuntimeMode.CLOUD_CPU`** activates when `import torch` fails completely. Because the Rust extension requires compilation with Maturin (which is omitted from the cloud requirements file), formula evaluation falls back to Python's native `Node.evaluate()`. Fitness evaluation swaps the real model for the synthetic quadratic loss landscape. The Streamlit dashboard displays a "Cloud Mode" warning banner in the sidebar, and evolution parameters are automatically capped at 20 generations and a population size of 100 to prevent out-of-memory crashes.
* **`RuntimeMode.LOCAL_CPU`** triggers when PyTorch is present but CUDA is missing. Formula evaluation automatically leverages the Rust extension if it was compiled locally, while fitness evaluation runs the real PyTorch training loop directly on the host CPU. While slower per candidate than GPU execution, this mode extracts a genuine training signal instead of a synthetic simulation.
* **`RuntimeMode.LOCAL_GPU`** initializes when both PyTorch and CUDA are detected. This enables the complete production pipeline: Rust-backed AST evaluation, the custom `VRAMDataLoader`, AMP training, and `torch.compile`. This is the native environment the system was designed for, yielding the highest-quality fitness signals.

The `SymboLRConfig` singleton instantiates once at import time and propagates this detected mode across the entire codebase. This isolates hardware-based branching entirely within the configuration module and the `_setup_evaluation_stack` function in `ui/state.py`, keeping the core evolution loop completely free of direct `torch.cuda.is_available()` runtime checks.

To run the full GPU pipeline locally after cloning:

```bash
# Install GPU dependencies
pip install -r requirements-gpu.txt
 
# Build the Rust extension
cd rust_core && maturin develop --release && cd ..
 
# Launch the dashboard (auto-detects GPU)
streamlit run app.py
 
# Or run the CLI benchmark directly
python benchmark.py --generations 10 --pop_size 30 --epochs 2 --workers 3 --seed 42

```

## Empirical Results

The table below reflects synthetic fitness landscape results in cloud mode. Running in GPU mode produces real MNIST or CIFAR-10 validation losses that vary by seed and generation count. These specific values represent a standard 10-generation, 50-population cloud run:

| Schedule | Type | Synthetic Val Loss |
| --- | --- | --- |
| **SymboLR Elite** | **Discovered** | **~0.08 to 0.18** |
| Cosine Annealing | Hand-crafted | ~0.21 |
| 1-Cycle | Hand-crafted | ~0.23 |
| Warm Restarts | Hand-crafted | ~0.27 |
| Exponential Decay | Hand-crafted | ~0.31 |
| Linear Decay | Hand-crafted | ~0.38 |
| Step Decay | Hand-crafted | ~0.41 |
| Constant LR | Hand-crafted | ~0.82 |

A few consistent patterns show up across multiple runs:

* **Early Concentration:** Formulas with a center of mass below 0.5 (concentrating the learning signal early on) consistently dominate the high-performing archive niches.
* **Emergent Periodicity:** Periodic `sin` and `cos` formulas reliably emerge as functional equivalents to warm restart schedules. They arise purely from the fitness signal without any human priors forcing periodicity.
* **Simplicity Wins:** The best-performing discovered formulas rarely exceed 7 to 9 AST nodes. This strong correlation between compactness and performance aligns with the theoretical expectation that simpler formulas generalize better across the heterogeneous curvatures of the quadratic landscape.
* **Dead Ends:** Formulas with no `t` dependence cluster in the worst-performing niches without exception.

---

## Testing

The test suite spans 11 unit modules and 2 integration tests, all designed to run entirely on the CPU without requiring a GPU:

| Module | Coverage |
| --- | --- |
| `test_gp_core.py` | Protected operator numerics, tree evaluation, MD5 hash determinism, and depth/size metrics |
| `test_evolution.py` | All three genetic operators, fitness cache clearing, and parent integrity post-crossover |
| `test_map_elites.py` | Behavioral descriptor math, niche competitive replacement, and uniform parent sampling |
| `test_evaluator.py` | Fitness caching, parallel evaluation ordering, Rust crash handling, and progress callbacks |
| `test_fidelity.py` | `VRAMDataLoader` drop_last logic, GPU-side shuffling, and stratification pixel/label alignment |
| `test_loader.py` | Stratified data splits, seed reproducibility, pinned memory, and subset fraction accuracy |
| `test_hybrid.py` | Constant node extraction, L-BFGS-B convergence on synthetic objectives, and bound enforcement |
| `test_simplify.py` | Algebraic identity removal, structural self-cancellation, SymPy `Pow` handling, and fail-closed fallbacks |
| `test_probe.py` | `FastConvNet` forward shapes, NaN learning rate guards, explosion early exits, and patience-based stopping |
| `test_rust_core.py` | Prefix serialization correctness and Rust/Python numerical parity down to a `1e-7` tolerance |
| `test_state.py` | Worker budget resolution logic across varying GPU and CPU runtime modes |

```bash
pytest tests/unit/ -v
pytest tests/integration/ -v
pytest --cov=. --cov-report=term-missing
```