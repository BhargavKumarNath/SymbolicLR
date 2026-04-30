# SymboLR: Comprehensive System Analysis
> A Genetic Programming Framework for Symbolic Learning Rate Discovery

---

## Table of Contents
 
1. [System Overview](#1-system-overview)
2. [Architecture & Workflow](#2-architecture--workflow)
3. [Data & Knowledge Base](#3-data--knowledge-base)
4. [Key Features & Innovations](#4-key-features--innovations)
5. [Strengths, Limitations & Improvements](#5-strengths-limitations--improvements)

---

## 1. System Overview
### What SymboLR Does
SymboLR is a **Quality-Diversity Genetic Programming** system that autonomously evolves mathematical formulas for neural network learning rate schedules. Rather than using hand-crafted schedules like cosine annealing or step decay, SymboLR treats the schedule discovery problem as a **symbolic regression** task: Given only a probe model, a dataset, and a training budget, it searches the space of all possible mathematical expressions and returns the formulas that minimize validation loss.
 
The discovered schedules are represented as **Abstract Syntax Trees (ASTs)**, evaluated via a compiled Rust extension, and maintained in a **MAP-Elites behavioral archive** that simultaneously optimizes for both quality (low validation loss) and diversity (schedules with different complexity/timing characteristics).

### Core Objectives

- **Automate Schedule Design**: Replace human intuition about warmup, decay, and oscillation with evolved symbolic expressions.
- **Maintain Behavioral Diversity**: Use Quality-Diversity optimisation to ensure the archive contains fundamentally different types of schedules, not just slight variations of one winner.
- **Maximize Evaluation Throughput**: Accelerate the inner fitness loop via Rust-compiled AST evaluation, VRAM-resident data loading, and AMP-enabled GPU training.
- **Produce Human-Interpretable Results**: Algebraically simplify discovered formulas via SymPy CAS and render them as LaTeX, so a practitioner can understand and trust what was found.
- **Compare Against Baselines**: Benchmark discovered schedules against cosine annealing, step decay, warm restarts, linear decay, and constant LR.

### Primary Use Cases

- Research into the geometry of the neural network loss landscape and how learning rate dynamics interact with it.
- AutoML pipelines where training schedules need to be tailored to specific architectures or datasets.
- Portfolio demonstration of production-grade systems integrating evolutionary computation, GPU-accelerated ML, Rust/Python interop, and modern UI engineering.

---

## 2. Architecture & Workflow

### High-Level Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        SymboLR Pipeline                                 │
│                                                                         │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐            │
│  │  Init    │──▶│ Evaluate │──▶│ Archive  │──▶│  Evolve  │──┐         │
│  │ Ramped   │   │ Rust+GPU │   │MAP-Elites│   │3 Operators│  │         │
│  │ H&H Pop  │   │Concurrent│   │  2D Grid │   │ Xover/Mut│  │         │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘  │         │
│                                                               │ loop    │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐                 │         │
│  │ Hall of  │◀──│  L-BFGS-B│◀──│  SymPy   │◀────────────────┘         │
│  │  Fame    │   │ Memetic  │   │Simplify  │                            │
│  │  Output  │   │ Refine   │   │  Trees   │                            │
│  └──────────┘   └──────────┘   └──────────┘                            │
└─────────────────────────────────────────────────────────────────────────┘
```

### Component Breakdown

#### 2.1 Population Initialization (`gp/population.py`)

The system uses **Ramped Half-and-Half** to bootstrap generation zero. This is a cannonical GP in inialization strategy that maximizes structural diversity before a single evaluation is run.

- The population is divided evenly across depth levels from `min_depth` to `max_depth`.
- Within each depth level, half the individuals use the `full` method (symmetric trees where every branch reaches the maximum depth) and half use `grow` (asymmetric trees with a 50% terminal probability at each node).
- Terminals consist of the time variable `t` plus nine float constants `{0.1, 0.2, ..., 0.9}`.
- Operators are drawn from a registry of 10 protected mathematical functions (see Section 5).

```python
# From gp/population.py
def ramped_half_and_half(pop_size, min_depth, max_depth):
    for i in range(pop_size):
        depth = depths[i % len(depths)]
        method = 'full' if i % 2 == 0 else 'grow'
        population.append(generate_tree(1, depth, method))
```

#### 2.2 AST Representation (`gp/tree.py`)

Each individual is a `Node` object forming a recursive AST. Nodes can be:

- **Operator Nodes**: Binary (`+`, `-`, `*`, `/`) or unary (`sin`, `cos`, `exp`, `log`, `sqrt`, `abs`)
- **Variable Terminals**: The string `'t'`, representing normalized training time in `[0, 1]`
- **Constant Terminals**: Python `float` values

Key design decisions:
 
- **MD5 Hash Caching** (`_hash_cache`) enables O(1) identity checks across generations without deep equality traversal.
- **Fitness Caching** (`fitness` attribute) prevents re-evaluating identical ASTs that reappear via crossover.
- **`to_prefix()`** serializes the tree as a space-separated prefix string consumed by the Rust evaluator.
- **`invalidate_cache()`** recursively clears both the hash and fitness caches after any genetic operator modifies a subtree.

#### 2.3 Operator Set & Protection (`gp/operators.py`)

Every operator in the GP primitive set includes a numerical guard against undefined mathematical behavior:

| Operator | Protection |
|----------|-----------|
| `/` | Returns `1.0` if `\|denominator\| < 1e-6` |
| `log` | Operates on `max(\|x\|, 1e-6)` |
| `sqrt` | Operates on `\|x\|` |
| `exp` | Clamps exponent to `[-100, 10]` |
| `sin`, `cos`, `abs`, `+`, `-`, `*` | No protection needed |

These protections are mirrored identically in the Rust core (`rust_core/src/lib.rs`), ensuring Python test evaluations and production Rust evaluations produce numerically identical results.

#### 2.4 Rust-Accelerated Evaluation (`rust_core/src/lib.rs`, `gp/evaluator.py`)

The inner evaluation loop is the hottest code path in the system. Python AST traversal with NumPy incurs per-call overhead that becomes prohibitive at population sizes of 50–200 evaluated concurrently over 10+ generations.

The Rust extension (`symbolr_rust`) is compiled via PyO3/Maturin and exposes a single function:

```rust
fn evaluate_fast(prefix_expr: &str, t_array: PyReadonlyArray1<f64>) -> PyResult<&PyArray1<f64>>
```

The Rust evaluator:
1. Parses the prefix string into a native Rust `Expr` enum (a recursive algebraic data type mirroring the Python `Node`).
2. Evaluates element-wise in a single tight loop over the time array, no intermediate array allocations.
3. Applies the same numerical protections as the Python operators.
4. Returns a `PyArray1<f64>` via zero-copy NumPy interop.
The `ParallelEvaluator` wraps this in a `ThreadPoolExecutor`:

```
For each individual in the generation:
  Thread N:
    1. Call symbolr_rust.evaluate_fast(prefix, t_array)  → lr_schedule (numpy)
    2. model_factory() → fresh FastConvNet on GPU
    3. ProbeTrainer.evaluate_schedule(model, loaders, lr_schedule, epochs) → val_loss
    4. Cache val_loss on tree.fitness
```
 
The thread pool is sized conservatively (`_resolve_worker_budget`) to avoid GPU OOM:
 
```
CUDA, epochs >= 5 or pop >= 100  → max 1 worker
CUDA, epochs >= 3 or pop >= 60   → max 2 workers
CUDA, otherwise                  → max 3 workers
CPU                              → min(requested, 4, pop_size)
```

#### 2.5 MAP-Elites Archive (`gp/map_elites.py`)

MAP-Elites is the heart of the system. Instead of a single population, it maintains a 2D grid where each cell holds the single best individual in a behavioral niche.
 
**Behavioral Descriptors:**
 
- **Dimension 1 — Tree Complexity**: AST node count, binned into 30 buckets. Distinguishes compact formulas (`sin(t)`, 2 nodes) from complex multi-term expressions.
- **Dimension 2 — Center of Mass**: Computed as `Σ(t · LR(t)) / Σ(LR(t))`, the weighted centroid of the learning rate curve over time. Binned into 20 buckets. A value near 0 means learning is concentrated early (warmup-style); near 1 means late-concentrated (cooldown-style).
**Niche assignment:** Both descriptors are computed by the Rust evaluator on the same `t_array` used for fitness. This ensures descriptor computation is fast and consistent.
 
**Insertion rule:** A new individual replaces the niche occupant only if its validation loss is strictly lower. Archive size is bounded by `30 × 20 = 600` niches maximum.
 
**Parent sampling:** Parents are selected **uniformly at random** from all occupied niches not by fitness rank. This gives rare structural configurations equal reproductive probability, strongly preventing premature convergence.

#### 2.6 Genetic Operators (`gp/evolution.py`)
 
Three operators are applied probabilistically per offspring pair:
 
| Operator | Probability | Mechanism |
|----------|-------------|-----------|
| Subtree Crossover | 50% | Swap two random subtrees between parents, reject offspring exceeding `max_depth=7` |
| Subtree Mutation | 25% | Replace a random node with a freshly generated random subtree |
| Hoist Mutation | 25% | Replace a subtree with one of its own descendants, guaranteeing smaller offspring |
 
All operators deep-copy parents before modification and call `_clear_all_caches()` on offspring to prevent stale fitness values propagating into the new generation.
 
#### 2.7 Algebraic Simplification (`gp/simplify.py`)
 
After genetic operations, offspring are passed through **SymPy CAS** to prune algebraic bloat:
 
```
Node AST → _node_to_sympy() → SymPy Expr
         → sympy.nsimplify(tolerance=1e-4)
         → sympy.simplify()
         → _sympy_to_node() → pruned Node AST
```
 
This collapses identities like `(t + 0) * 1` → `t`, `t / t` → `1`, and `t - t` → `0` before archive insertion. Trees exceeding 50 nodes are skipped (cost/benefit unfavorable). A strict fallback returns the original tree if SymPy encounters an unsupported structure, preventing GP loop crashes.
 
The simplification is applied **lazily** — only on candidates that would actually improve a niche — to avoid paying the SymPy cost for the majority of offspring that fail the competitive replacement test.
 
#### 2.8 Hybrid Memetic Optimization (`optimiser/hybrid.py`)
 
After evolution terminates, the top-k Hall of Fame formulas undergo **L-BFGS-B gradient descent** on their scalar constants:
 
```python
# Extract all float terminal nodes from the AST
constant_nodes = _get_constant_nodes(optimized_tree)
 
# scipy.optimize.minimize with bounds=(1e-6, 10.0)
res = minimize(objective, initial_guess, method='L-BFGS-B', bounds=..., options={'maxiter': 15})
```
 
The `objective` function injects SciPy-proposed values directly into the live AST nodes (by reference), clears the fitness cache, and calls the `ProbeTrainer` for a fresh evaluation. This hybrid GP + gradient approach combines GP's strength at structural search with gradient descent's precision at numeric refinement.
 
#### 2.9 Probe Model (`models/probe.py`)
 
The fitness function evaluates each formula by training a `FastConvNet`, a lightweight 2-layer CNN designed for rapid throughput on the RTX 4070:
 
```
Input → Conv2d(32) + BN + ReLU + MaxPool
      → Conv2d(64) + BN + ReLU + MaxPool
      → AdaptiveAvgPool(4×4)
      → FC(64×16 → 128) + ReLU
      → FC(128 → 10)
```
 
The `ProbeTrainer` wraps training with:
- **AMP (Automatic Mixed Precision)** via `torch.autocast` and `GradScaler` — halves memory, roughly doubles throughput on CUDA.
- **Early stopping** with configurable `patience` and `min_delta`.
- **Explosion guard**: if loss exceeds `explode_threshold` or is non-finite, return `inf` immediately.
- **LR guard**: if the GP-produced LR is NaN, Inf, ≤ 0, or > 10, return `inf` immediately.
- `torch.compile(mode="reduce-overhead")` on non-Windows platforms for Triton kernel fusion.

### End-to-End Data Flow
 
```
User clicks "Start Evolution"
        │
        ▼
start_evolution() → uuid run_id → _RUNS dict → _EXECUTOR.submit(_run_evolution_job)
        │
        ▼
[Background Thread]
FidelityManager.get_low_fidelity()     → VRAMDataLoader (MNIST 5% → GPU VRAM)
ramped_half_and_half(pop_size)         → List[Node]
ParallelEvaluator.evaluate_population()
    ├── Thread 1: symbolr_rust.evaluate_fast(prefix, t_array) → lr_schedule
    │             ProbeTrainer.evaluate_schedule(model, loaders, lr) → val_loss
    ├── Thread 2: (same)
    └── Thread N: (same)
        │
        ▼
archive.try_add(simplify_tree(ind), fit)  ← lazy SymPy simplification
        │
        ▼
[For each generation]
archive.sample_parents(pop_size)          ← uniform niche sampling
genetic operators (crossover/mutation)    → offspring
evaluate_population(offspring)            → fitnesses
try_add to archive (competitive replace)
_publish_progress() → _RUNS[run_id]       ← throttled (≤1 publish/sec)
        │
        ▼
[Post-evolution]
hybrid_optimize_constants(hof_trees)      ← L-BFGS-B on float terminals
        │
        ▼
[Streamlit Main Thread]
sync_state() → _get_run_state() → st.session_state
st.fragment(run_every=1.5s) renders live progress
Page routing renders charts from session_state
```
 
---

## 3. Data & Knowledge Base
 
### 3.1 Dataset Tiers
 
SymboLR uses a **multi-fidelity evaluation strategy** to allocate compute efficiently:
 
| Tier | Dataset | Fraction | Samples | Use Case |
|------|---------|----------|---------|----------|
| Low | MNIST | 5% | ~3,000 | GP evolution, rapid candidate screening |
| Medium | CIFAR-10 | 20% | ~10,000 | Refinement of shortlisted candidates |
| High | CIFAR-10 | 100% | 50,000 | Final elite ranking |
 
In the current Streamlit-launched evolution, only the low-fidelity tier is active (sufficient for demonstrating discovered formulas and comparing against baselines).
 
### 3.2 Data Preprocessing & Stratification
 
All dataset splits use **stratified sampling** (`sklearn.model_selection.train_test_split` with `stratify=targets`) to preserve class balance at every fidelity level and train/val split.
 
The `FidelityManager._prepare_vram_split()` pipeline:
1. Extracts raw numpy arrays from the torchvision dataset.
2. Applies a stratified subset fraction.
3. Applies a stratified 80/20 train/val split.
4. Applies torchvision transforms per-sample.
5. Stacks into contiguous tensors and `.to(device, non_blocking=True)` directly to GPU VRAM.
A regression test (`test_prepare_vram_split_keeps_subset_images_and_labels_aligned`) explicitly validates that image pixels and labels remain correctly aligned after the subset operation — a subtle but critical correctness requirement.
 
### 3.3 VRAMDataLoader
 
The custom `VRAMDataLoader` eliminates PCIe CPU→GPU transfers entirely:
 
```python
class VRAMDataLoader:
    def __iter__(self):
        indices = torch.randperm(self.n_samples, device=self.x.device)  # GPU-side shuffle
        for start_idx in range(0, self.n_samples, self.batch_size):
            batch_idx = indices[start_idx:end_idx]
            yield self.x[batch_idx], self.y[batch_idx]  # pure VRAM slicing
```
 
With a 3,000-sample MNIST subset at float32 (28×28×1), the total VRAM footprint is approximately 9 MB — trivial on an 8 GB RTX 4070. This makes batch iteration a pure kernel operation with no data movement overhead.
 
### 3.4 Knowledge Representation
 
There is no persistent vector store or embedding index. The system's "knowledge base" is the **MAP-Elites archive** itself which is a live, in-memory `Dict[Tuple[int,int], Tuple[float, Node]]` mapping behavioral niches to their best-known formula. This archive functions as a sparse quality-diversity map, encoding everything the system has discovered about the fitness landscape.
 
The archive snapshot is serialized to JSON-compatible dicts for Streamlit charting, capped at 500 points with stratified loss-based sampling to keep deepcopy costs bounded at large archive sizes.
 
---

## 4. Key Features & Innovations
 
### 4.1 Rust/Python Hybrid Evaluation
 
The PyO3-compiled `symbolr_rust.evaluate_fast` function is the most performance-critical innovation. The Rust `Expr` enum is a recursive algebraic data type that:
 
- Parses prefix notation in O(n) with a single iterator pass.
- Evaluates element-wise with scalar dispatch, no intermediate array allocations, no NumPy overhead.
- Mirrors Python's numerical protections exactly, enabling test suites to validate parity with `np.testing.assert_allclose(atol=1e-7)`.
The estimated throughput improvement over pure Python NumPy evaluation is 10–50× for complex expressions.
 
### 4.2 Quality-Diversity Archive
 
Standard GP uses fitness-proportional or tournament selection, which tends to converge on a small number of dominant individuals. MAP-Elites prevents this by maintaining diversity as a first-class objective. The behavioral descriptors (complexity + center of mass) are domain-specifically chosen: They capture the two most meaningful axes of learning rate schedule variation how complex the formula is and when it concentrates the learning signal.
 
### 4.3 Algebraic Bloat Control
 
GP bloat (unbounded tree growth without fitness improvement) is a known failure mode that wastes archive slots and slows Rust evaluation. SymboLR addresses this at two levels:
 
- **Hoist mutation** (25% probability): structurally guaranteed to reduce tree size.
- **SymPy simplification**: algebraically reduces trees to their canonical minimal form before archive insertion.
- **Size cap**: trees exceeding 50 nodes skip SymPy (cost/benefit unfavorable) and are handled by the hoist pressure instead.
### 4.4 Protected Operator Set
 
All 10 operators include numerical guards against division-by-zero, log of zero, sqrt of negative, and exponential overflow. These protections are deliberately conservative designed to keep learning rates in a plausible range `[1e-6, 10]` without silently corrupting the fitness signal. The Rust implementation mirrors these protections exactly, ensuring that the test suite's Python-vs-Rust parity check is a meaningful correctness guarantee.
 
### 4.5 VRAM-Resident Data Loading
 
The `VRAMDataLoader` eliminates PCIe transfers entirely by keeping the entire low-fidelity dataset resident in GPU VRAM. At the low-fidelity tier (~3,000 MNIST samples), this is practical with 8 GB VRAM. The benefit is most pronounced at high worker counts, where multiple threads would otherwise saturate the PCIe bus with competing data transfers.
 
### 4.6 AMP + torch.compile
 
The `ProbeTrainer` uses:
- `torch.autocast` + `GradScaler` for FP16 mixed precision, approximately doubling throughput and halving memory relative to FP32.
- `torch.compile(mode="reduce-overhead")` on Linux/macOS, which fuses small GPU kernels via Triton and minimizes CPU overhead — particularly valuable for the small batches and rapid model instantiations characteristic of GP evaluation.
### 4.7 Fitness Caching with Hash-Based Identity
 
Each `Node` computes an MD5 hash from its structural content recursively. Identical subtrees across different generations produce identical hashes, enabling O(1) cache lookups that short-circuit the Rust → GPU pipeline entirely. This is particularly valuable after hoist mutations and SymPy simplifications, which frequently produce structurally equivalent formulas from different parent combinations.
 
### 4.8 Progress Publishing with Throttling
 
The background thread publishes progress to `_RUNS` at most once per second (`_PUBLISH_THROTTLE_S = 1.0`) for heavy fields (archive snapshot, LR curves), while lightweight fields (gen_log, progress ratios) are updated every callback. This prevents the background thread from blocking on expensive `deepcopy` operations of large archives at high population sizes.
 
### 4.9 Test Suite Depth
 
The project includes 11 test modules across unit and integration tiers:
 
| Module | What It Tests |
|--------|--------------|
| `test_gp_core.py` | Operator protections, tree evaluation, hashing, metrics |
| `test_evolution.py` | All GP operators, fitness cache clearing, tournament selection |
| `test_map_elites.py` | Behavioral descriptors, niche competition, parent sampling |
| `test_evaluator.py` | Fitness caching, parallel evaluation, OOM handling, progress callbacks |
| `test_fidelity.py` | VRAM loader drop_last, shuffling, stratification alignment |
| `test_loader.py` | Stratified splits, seed reproducibility, pinned memory |
| `test_hybrid.py` | Constant extraction, L-BFGS-B convergence, bounds clipping |
| `test_simplify.py` | Algebraic identity removal, cancellation, fallback safety |
| `test_probe.py` | FastConvNet shapes, NaN LR handling, early stopping, explosion guard |
| `test_rust_core.py` | Prefix serialization, Rust/Python numerical parity |
| `test_state.py` | Worker budget resolution logic |
 
Integration tests cover the full Streamlit app boot and the complete benchmark pipeline with mocked GPU dependencies.
 
---

## 5. Strengths, Limitations & Improvements
 
### 5.1 Current Strengths
 
- **End-to-end correctness**: Rust/Python parity is validated at `1e-7` tolerance in `test_rust_core.py`. The test suite catches regressions across the entire stack.
- **Performance-aware design**: Every hot path like formula evaluation, data loading, model training etc has been explicitly optimized for the target hardware (RTX 4070, Ryzen 9).
- **Human-readable outputs**: SymPy LaTeX rendering of discovered formulas makes results interpretable to practitioners, not just ML researchers.
- **Robust evolution loop**: The system handles the full range of GP failure modes (bloat, constant collapse, numeric instability) without crashing.
- **Live observability**: The Streamlit dashboard gives genuine real-time insight into archive growth and convergence, not just a post-hoc report.
### 5.2 Limitations & Bottlenecks
 
#### UI Freezes Under High Configuration Load
 
**This is the most user-facing limitation**: when users set high generation counts (≥ 10), large population sizes (≥ 100), and multiple epochs (≥ 3) simultaneously, the Streamlit dashboard can become unresponsive. The root causes are:
 
1. **Archive deepcopy cost**: `_build_archive_snapshot()` deepcopies up to 500 archive entries. At large archive sizes, this can block the background thread for hundreds of milliseconds per publish cycle, creating visible stutter.
2. **SymPy bottleneck**: SymPy simplification is CPU-bound and single-threaded. At pop_size=200, simplifying 100 offspring per generation can take several seconds of wall time, during which no progress is published.
3. **Streamlit fragment rerun**: `st.fragment(run_every=1.5)` triggers a full fragment rerun every 1.5 seconds. If the background thread is blocked on SymPy or deepcopy, the UI renders stale state but still incurs rerun overhead.
4. **ThreadPoolExecutor saturation**: At pop_size=200 with workers=4, the thread pool queues 200 evaluation tasks. Early completions increment progress correctly, but the remaining long-tail evaluations stall the gen_log publish.
The `_PUBLISH_THROTTLE_S = 1.0` and archive size cap of 500 points partially mitigate this, but the fundamental issue is that SymPy simplification and archive snapshot construction are synchronous and CPU-bound in the background thread.
 
#### Other Limitations
 
- **Single-fidelity in practice**: The medium and high-fidelity tiers (`FidelityManager.get_medium_fidelity`, `get_high_fidelity`) are implemented but not wired into the evolution loop or the UI. The current system evaluates all candidates on the same low-fidelity MNIST tier regardless of their archive rank.
- **No baseline schedule evaluation**: `baselines/schedules.py` and `optimiser/compare.py` are empty stubs. The baseline comparison chart in `ui/charts.py` uses hardcoded illustrative values (`BASELINE_DATA`) rather than results computed by actually running the baseline schedules on the same probe task.
- **pyo3 version pin**: The Rust core is pinned to `pyo3 = 0.20.0`, which does not support Python 3.13+ natively. Upgrading to pyo3 0.22+ would require API changes in the macro usage.
- **No persistent run history**: `_RUNS` is an in-memory dict that is evicted after 4 terminal runs. A completed evolution is lost on Streamlit server restart.
- **SymPy on every offspring**: Despite the lazy insertion check, the current `_run_evolution_job` calls `simplify_tree` inside the niche improvement check loop, which processes every offspring that would improve a niche, not strictly every offspring, but at high archive occupancy this can still be a significant fraction.
- **No surrogate model**: Every candidate requires a real GPU training run for fitness evaluation. At pop_size=100 and 10 generations, this is 1,000+ training runs. A neural surrogate (LSTM or GNN) on the archive could pre-screen candidates before GPU evaluation, reducing wall time by an order of magnitude.
### 5.3 Concrete Improvement Suggestions
 
**Short-term (correctness & stability):**
 
1. **Implement `baselines/schedules.py`** — Add actual implementations of CosineAnnealing, StepDecay, WarmRestarts, LinearDecay as `Node`-compatible schedule generators, and wire them through `ParallelEvaluator` to produce real baseline numbers for the comparison chart.
2. **Wire multi-fidelity cascade** — After each GP generation, promote the top-5 archive elites to medium-fidelity evaluation. Demote individuals whose medium-fidelity loss diverges from their low-fidelity estimate by more than a threshold. This concentrates compute on genuinely promising formulas.
3. **Move SymPy to a process pool** — Replace `ThreadPoolExecutor` for simplification with `ProcessPoolExecutor` to avoid the GIL and parallelize SymPy's CPU-bound CAS work across cores.
4. **Persist run history to SQLite** — Store completed `_RUNS` entries (gen_log, hof, archive_snapshot as JSON) to a local SQLite file. This enables resumable runs and cross-session comparison.
**Medium-term (performance & scale):**
 
5. **Neural surrogate pre-screening** — Train a lightweight GNN on (AST structure → predicted val_loss) using the growing archive as training data. Use it to reject the bottom 50% of offspring before GPU evaluation. This halves the evaluation budget per generation after an initial warm-up period.
6. **Async Streamlit architecture** — Replace `st.fragment(run_every=1.5)` with `st.write_stream` or WebSocket-based push updates to eliminate polling overhead and make the UI feel more responsive during long runs.
7. **Operator probability adaptation** — Track per-operator offspring fitness improvements over a sliding window of generations. Reweight crossover/mutation/hoist probabilities adaptively toward whichever operator is currently producing the most archive improvements.
8. **PyO3 upgrade to 0.22+** — Unlock Python 3.13 compatibility and the new `#[pyclass]` features that would allow exporting the Rust `Expr` type directly to Python for inspection and debugging.
**Long-term (research impact):**
 
9. **Transfer evaluation** — After discovering elite schedules on MNIST/CIFAR-10 CNNs, evaluate them on a transformer fine-tuning task (e.g., DistilBERT on SST-2) to test cross-architecture generalization. This would be a significant research result.
10. **MLflow experiment tracking** — Log hyperparameters, gen_log, and final Hall of Fame to MLflow for systematic reproducibility analysis across seeds and hardware configurations.
11. **Grammar-guided GP** — Constrain the tree grammar to prevent degenerate structures (e.g., constant-only formulas, pure cosine with no `t` dependence) before evaluation, rather than discovering they are unfit through wasted GPU compute.
---

