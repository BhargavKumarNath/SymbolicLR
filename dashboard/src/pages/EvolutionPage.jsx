import React, { useState } from 'react';
import SectionHeader from '../components/SectionHeader';
import CodeBlock from '../components/CodeBlock';

const EvolutionPage = () => {
    const [step, setStep] = useState(0);

    const steps = [
        {
            title: "Population Initialization",
            subtitle: "Ramped Half-and-Half",
            color: "var(--blue)",
            content: "The first generation uses ramped half-and-half to maximize structural diversity. Population is split across tree depths 2–5, with half using the 'full' method (perfect symmetry) and half using 'grow' (asymmetric, variable depth). This prevents the early collapse seen in homogeneous starting pools.",
            code: `# gp/population.py
def ramped_half_and_half(pop_size, min_depth, max_depth):
    population = []
    depths = range(min_depth, max_depth + 1)
    for i in range(pop_size):
        depth = depths[i % len(depths)]
        method = 'full' if i % 2 == 0 else 'grow'
        population.append(generate_tree(1, depth, method))
    return population`,
        },
        {
            title: "AST Evaluation",
            subtitle: "Tree → LR Schedule",
            color: "var(--orange)",
            content: "Each candidate formula is serialized to prefix notation and evaluated over a 100-step time array. The Rust backend handles this in a tight loop with zero intermediate allocations. The output is clamped to [1e-7, 10.0] and sanitized for NaN/Inf before fitness evaluation.",
            code: `# gp/rust_bridge.py
def evaluate_schedule(tree, t_array):
    prefix = tree.to_prefix()  # "* 0.067 cos t"
    result = symbolr_rust.evaluate_fast(prefix, t_array)
    result = np.nan_to_num(result, nan=1e-4, posinf=1.0, neginf=1e-7)
    return np.clip(result, 1e-7, 10.0)`,
        },
        {
            title: "Fitness Evaluation",
            subtitle: "GPU Probe Training",
            color: "var(--purple)",
            content: "Fitness is measured by training a FastConvNet from scratch on 5% of MNIST using the candidate schedule. AMP is enabled, early stopping triggers after 2 stagnant epochs, and schedules with NaN/exploding losses return ∞. A parsimony penalty rewards simpler formulas.",
            code: `# models/probe.py
def evaluate_schedule(model, train_loader, val_loader, lr_schedule, epochs):
    for step, (inputs, targets) in enumerate(train_loader):
        lr = lr_schedule[step]
        if not math.isfinite(lr) or lr > 10.0:
            return float('inf')  # Kill divergent schedules
        optimizer.param_groups[0]['lr'] = lr
        # AMP forward + backward pass...
    return best_val_loss`,
        },
        {
            title: "MAP-Elites Archive",
            subtitle: "3D Behavioral Grid",
            color: "var(--green)",
            content: "Each formula is assigned a behavioral niche based on: (1) AST size (complexity), (2) center-of-mass (temporal concentration), and (3) total variation (smoothness). A formula replaces the incumbent only if it achieves lower validation loss. This maintains 6,000 unique behavioral niches simultaneously.",
            code: `# gp/map_elites.py
def _compute_descriptors(self, tree):
    schedule = evaluate_schedule(tree, self.t_array)
    com = np.sum(t_array * schedule) / np.sum(schedule)
    tv = np.sum(np.abs(np.diff(schedule))) / max_tv
    return size_idx, com_idx, smoothness_idx

def try_add(self, tree, raw_loss):
    niche = self._compute_descriptors(tree)
    if niche in self.archive:
        incumbent_loss, _, age = self.archive[niche]
        if raw_loss >= incumbent_loss + age * self.age_penalty:
            return False  # Incumbent wins
    self.archive[niche] = (raw_loss, deepcopy(tree), 0)`,
        },
        {
            title: "Genetic Operators",
            subtitle: "Structural Variation",
            color: "var(--teal)",
            content: "Five operators with EMA-based adaptive selection rates: Subtree crossover (45%) swaps random subtrees between parents. Subtree mutation (25%) replaces a node with a fresh random subtree. Hoist mutation (10%) replaces a node with one of its descendants — guaranteed size reduction. Point mutation (10%) changes a single operator/terminal. Constant perturbation (10%) adds Gaussian noise to a constant.",
            code: `# gp/evolution.py
def hoist_mutation(parent):
    # Anti-bloat: replace node with its own descendant
    internal_nodes = [n for n in nodes if n.children]
    n = random.choice(internal_nodes)
    descendants = _get_all_nodes(n)[1:]
    chosen = random.choice(descendants)
    n.value = chosen.value
    n.children = chosen.children  # Always smaller
    return p`,
        },
        {
            title: "Algebraic Simplification",
            subtitle: "SymPy CAS Pipeline",
            color: "var(--red)",
            content: "Before archive insertion, each offspring passes through SymPy. The pipeline: (1) constant fold subexpressions with no 't', (2) nsimplify converts floats to rationals, (3) sympy.simplify applies algebraic identities. Result: (t + 0.0) * 1.0 → t. Trees over 30 nodes skip simplification for performance. Fail-closed: exceptions return the original tree.",
            code: `# gp/simplify.py
def simplify_tree(tree):
    original = constant_fold(tree)
    if original.size() > 30:
        return original  # Skip expensive simplification
    try:
        sp_expr = _node_to_sympy(original)
        simplified = sympy.simplify(sp_expr)
        return _sympy_to_node(simplified)
    except Exception:
        return original  # Fail closed`,
        },
    ];

    return (
        <div style={{ padding: "60px" }}>
            <SectionHeader
                tag="How GP Works"
                title="The Evolution Loop"
                subtitle="Six carefully engineered stages, each designed to maximize search efficiency while maintaining structural diversity across the formula space."
            />

            {/* Step selector */}
            <div style={{ display: "flex", gap: 8, marginBottom: 40, flexWrap: "wrap" }}>
                {steps.map((s, i) => (
                    <button
                        key={i}
                        onClick={() => setStep(i)}
                        style={{
                            padding: "8px 16px", borderRadius: 8,
                            background: step === i ? s.color : "var(--bg2)",
                            border: `1px solid ${step === i ? s.color : "var(--border)"}`,
                            color: step === i ? "#000" : "var(--text-2)",
                            fontSize: 12, fontWeight: 600,
                            transition: "all var(--transition)", cursor: "pointer",
                            opacity: step === i ? 1 : 0.7
                        }}
                    >{i + 1}. {s.title}</button>
                ))}
            </div>

            {/* Step detail */}
            <div className="fade-in" key={step} style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24 }}>
                <div>
                    <div className="glass" style={{ padding: "36px", borderColor: `${steps[step].color}44` }}>
                        <div style={{
                            display: "inline-flex", alignItems: "center", gap: 8,
                            padding: "4px 12px", borderRadius: 20, marginBottom: 20,
                            background: `${steps[step].color}22`, border: `1px solid ${steps[step].color}44`,
                            color: steps[step].color, fontSize: 11, fontWeight: 600, letterSpacing: "0.08em"
                        }}>STAGE {step + 1}</div>
                        <h2 style={{ fontSize: 24, fontWeight: 700, letterSpacing: "-0.02em", marginBottom: 8 }}>
                            {steps[step].title}
                        </h2>
                        <div style={{ fontSize: 13, color: steps[step].color, marginBottom: 20, fontWeight: 500 }}>
                            {steps[step].subtitle}
                        </div>
                        <p style={{ fontSize: 14, color: "var(--text-2)", lineHeight: 1.8 }}>
                            {steps[step].content}
                        </p>
                    </div>

                    {/* Navigation */}
                    <div style={{ display: "flex", gap: 12, marginTop: 16 }}>
                        <button
                            onClick={() => setStep(s => Math.max(0, s - 1))}
                            disabled={step === 0}
                            style={{
                                flex: 1, padding: "12px", borderRadius: 8,
                                background: "var(--bg2)", border: "1px solid var(--border)",
                                color: step === 0 ? "var(--text-3)" : "var(--text)",
                                fontSize: 13, cursor: step === 0 ? "not-allowed" : "pointer"
                            }}
                        >← Previous</button>
                        <button
                            onClick={() => setStep(s => Math.min(steps.length - 1, s + 1))}
                            disabled={step === steps.length - 1}
                            style={{
                                flex: 1, padding: "12px", borderRadius: 8,
                                background: step === steps.length - 1 ? "var(--bg2)" : steps[step].color,
                                border: "none",
                                color: step === steps.length - 1 ? "var(--text-3)" : "#000",
                                fontSize: 13, fontWeight: 600,
                                cursor: step === steps.length - 1 ? "not-allowed" : "pointer"
                            }}
                        >Next →</button>
                    </div>
                </div>

                <div>
                    <CodeBlock code={steps[step].code} lang="Python" />
                    {/* Progress indicators */}
                    <div style={{ marginTop: 20, display: "flex", gap: 6 }}>
                        {steps.map((_, i) => (
                            <div key={i} style={{
                                flex: i === step ? 3 : 1, height: 3, borderRadius: 2,
                                background: i === step ? steps[step].color : "var(--border)",
                                transition: "all 0.4s ease"
                            }} />
                        ))}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default EvolutionPage;
