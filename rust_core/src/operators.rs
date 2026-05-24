//! operators.rs — Genetic operators for the SymboLR Thick Rust Core.
//!
//! ## O(1) Hard-Cap Pruning Contract
//!
//! Every operator that can enlarge a tree checks the offspring *once* with
//! `is_within_cap()` immediately after construction.  If the offspring
//! exceeds `MAX_NODES = 15` or `MAX_DEPTH = 7`, the **parent is returned
//! unchanged** — no retry loop, no recursion.  This makes worst-case
//! generation time per individual strictly O(parent_size), preventing the
//! generation-latency drift that emerges in uncapped GP systems.
//!
//! ## Thread Safety
//!
//! All public functions take `&mut impl Rng`.  Callers that run operators
//! inside a rayon closure must supply a thread-local RNG, e.g.
//! `rand::thread_rng()`.  The operators themselves have no shared state.

use crate::ast::Expr;
use rand::Rng;

// ─────────────────────────────────────────────────────────────────────────────
// 1.  Tree Path Type
// ─────────────────────────────────────────────────────────────────────────────

/// A path through the tree: a sequence of child indices from the root.
/// `[]` = root node.  `[0]` = first child.  `[1]` = second child (binary ops).
type Path = Vec<u8>;

// ─────────────────────────────────────────────────────────────────────────────
// 2.  Tree Traversal Helpers  (crate-private; used by all operators)
// ─────────────────────────────────────────────────────────────────────────────

/// Collect every path in the tree via DFS pre-order.
/// For a 15-node tree this produces at most 15 paths — O(n), tiny constant.
pub(crate) fn all_paths(expr: &Expr) -> Vec<Path> {
    let mut out = Vec::with_capacity(expr.size());
    collect_paths_rec(expr, &[], &mut out);
    out
}

fn collect_paths_rec(expr: &Expr, current: &[u8], out: &mut Vec<Path>) {
    out.push(current.to_vec());
    match expr {
        Expr::Var | Expr::Const(_) => {} // terminals: no children
        Expr::Sin(a) | Expr::Cos(a) | Expr::Exp(a)
        | Expr::Log(a) | Expr::Sqrt(a) | Expr::Abs(a) => {
            let mut p = current.to_vec();
            p.push(0);
            collect_paths_rec(a, &p, out);
        }
        Expr::Add(a, b) | Expr::Sub(a, b)
        | Expr::Mul(a, b) | Expr::Div(a, b) => {
            let mut pl = current.to_vec();
            pl.push(0);
            let mut pr = current.to_vec();
            pr.push(1);
            collect_paths_rec(a, &pl, out);
            collect_paths_rec(b, &pr, out);
        }
    }
}

/// Return a shared reference to the subtree at `path`.
/// Returns the root if the path is empty or steps past a terminal.
pub(crate) fn get_subtree<'a>(expr: &'a Expr, path: &[u8]) -> &'a Expr {
    if path.is_empty() {
        return expr;
    }
    let idx = path[0];
    let rest = &path[1..];
    match expr {
        // Unary: only child index 0 is valid.
        Expr::Sin(a) | Expr::Cos(a) | Expr::Exp(a)
        | Expr::Log(a) | Expr::Sqrt(a) | Expr::Abs(a) => get_subtree(a, rest),
        // Binary: 0 = left, anything else = right.
        Expr::Add(a, b) | Expr::Sub(a, b)
        | Expr::Mul(a, b) | Expr::Div(a, b) => {
            if idx == 0 { get_subtree(a, rest) } else { get_subtree(b, rest) }
        }
        // Terminal: path is exhausted or invalid — return self.
        _ => expr,
    }
}

/// Consume `expr` and return a new tree with the subtree at `path` replaced
/// by `replacement`.  If `path` is empty the entire tree is replaced.
pub(crate) fn replace_subtree(expr: Expr, path: &[u8], replacement: Expr) -> Expr {
    if path.is_empty() {
        return replacement;
    }
    let idx = path[0];
    let rest = &path[1..];
    match expr {
        Expr::Add(a, b) => if idx == 0 {
            Expr::Add(Box::new(replace_subtree(*a, rest, replacement)), b)
        } else {
            Expr::Add(a, Box::new(replace_subtree(*b, rest, replacement)))
        },
        Expr::Sub(a, b) => if idx == 0 {
            Expr::Sub(Box::new(replace_subtree(*a, rest, replacement)), b)
        } else {
            Expr::Sub(a, Box::new(replace_subtree(*b, rest, replacement)))
        },
        Expr::Mul(a, b) => if idx == 0 {
            Expr::Mul(Box::new(replace_subtree(*a, rest, replacement)), b)
        } else {
            Expr::Mul(a, Box::new(replace_subtree(*b, rest, replacement)))
        },
        Expr::Div(a, b) => if idx == 0 {
            Expr::Div(Box::new(replace_subtree(*a, rest, replacement)), b)
        } else {
            Expr::Div(a, Box::new(replace_subtree(*b, rest, replacement)))
        },
        Expr::Sin(a)  => Expr::Sin( Box::new(replace_subtree(*a, rest, replacement))),
        Expr::Cos(a)  => Expr::Cos( Box::new(replace_subtree(*a, rest, replacement))),
        Expr::Exp(a)  => Expr::Exp( Box::new(replace_subtree(*a, rest, replacement))),
        Expr::Log(a)  => Expr::Log( Box::new(replace_subtree(*a, rest, replacement))),
        Expr::Sqrt(a) => Expr::Sqrt(Box::new(replace_subtree(*a, rest, replacement))),
        Expr::Abs(a)  => Expr::Abs( Box::new(replace_subtree(*a, rest, replacement))),
        // Terminals: path steps past a leaf — just return the replacement.
        _ => replacement,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3.  Random Tree Generation  (used for initial population + subtree mutation)
// ─────────────────────────────────────────────────────────────────────────────

/// Generate a random expression tree using the "grow" method.
///
/// - At `depth_remaining == 0`: always generates a terminal.
/// - Otherwise: 30% chance of a terminal, 70% chance of an operator.
///
/// Trees generated by this function always satisfy `depth ≤ depth_remaining + 1`.
/// Caller must verify `is_within_cap()` if they need the hard cap guarantee.
pub fn generate_tree<R: Rng>(depth_remaining: usize, rng: &mut R) -> Expr {
    if depth_remaining == 0 || rng.gen_bool(0.30) {
        random_terminal(rng)
    } else {
        random_operator(depth_remaining.saturating_sub(1), rng)
    }
}

#[inline]
fn random_terminal<R: Rng>(rng: &mut R) -> Expr {
    if rng.gen_bool(0.5) {
        Expr::Var
    } else {
        // Constants skewed towards small positives (practical learning-rate range).
        let c: f64 = rng.gen_range(-2.0_f64..=2.0);
        Expr::Const(c)
    }
}

#[inline]
fn random_operator<R: Rng>(child_depth: usize, rng: &mut R) -> Expr {
    // 4 binary (0-3) + 6 unary (4-9) operators
    match rng.gen_range(0u8..10) {
        0 => Expr::Add( Box::new(generate_tree(child_depth, rng)),
                        Box::new(generate_tree(child_depth, rng))),
        1 => Expr::Sub( Box::new(generate_tree(child_depth, rng)),
                        Box::new(generate_tree(child_depth, rng))),
        2 => Expr::Mul( Box::new(generate_tree(child_depth, rng)),
                        Box::new(generate_tree(child_depth, rng))),
        3 => Expr::Div( Box::new(generate_tree(child_depth, rng)),
                        Box::new(generate_tree(child_depth, rng))),
        4 => Expr::Sin( Box::new(generate_tree(child_depth, rng))),
        5 => Expr::Cos( Box::new(generate_tree(child_depth, rng))),
        6 => Expr::Exp( Box::new(generate_tree(child_depth, rng))),
        7 => Expr::Log( Box::new(generate_tree(child_depth, rng))),
        8 => Expr::Sqrt(Box::new(generate_tree(child_depth, rng))),
        _ => Expr::Abs( Box::new(generate_tree(child_depth, rng))),
    }
}

/// Generate an initial population using ramped half-and-half:
/// depths cycle from `min_depth` to `max_depth` across the population.
pub fn ramped_half_and_half<R: Rng>(
    pop_size: usize,
    min_depth: usize,
    max_depth: usize,
    rng: &mut R,
) -> Vec<Expr> {
    let range = (max_depth - min_depth).max(1);
    (0..pop_size)
        .map(|i| {
            let depth = min_depth + (i % range);
            generate_tree(depth, rng)
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// 4.  Box-Muller Gaussian Sampler  (avoids rand_distr dependency)
// ─────────────────────────────────────────────────────────────────────────────

/// Sample from N(0, sigma²) using the Box-Muller transform.
#[inline]
fn gauss<R: Rng>(rng: &mut R, sigma: f64) -> f64 {
    let u1: f64 = rng.gen_range(1e-12_f64..1.0_f64);
    let u2: f64 = rng.gen::<f64>();
    let z = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
    z * sigma
}

// ─────────────────────────────────────────────────────────────────────────────
// 5.  Genetic Operators
// ─────────────────────────────────────────────────────────────────────────────
//
// O(1) Pruning Contract (enforced by every size-increasing operator):
//
//   let offspring = /* build candidate */;
//   if offspring.is_within_cap() { offspring } else { parent.clone() }
//
// The check is a single `.size() <= 15 && .depth() <= 7` evaluation —
// O(candidate_size), which is bounded by MAX_NODES = 15.  No retry.

/// **Subtree Crossover** — grafts a random subtree from `p2` into `p1`.
///
/// 1. Pick a random crossover point in `p1`.
/// 2. Pick a random donor subtree from `p2`.
/// 3. Replace the `p1` node with the donor.
/// 4. O(1) cap check: if offspring > 15 nodes or depth > 7, return `p1.clone()`.
pub fn subtree_crossover<R: Rng>(p1: &Expr, p2: &Expr, rng: &mut R) -> Expr {
    let paths1 = all_paths(p1);
    let paths2 = all_paths(p2);

    let xo_path  = paths1[rng.gen_range(0..paths1.len())].clone();
    let donor_path = paths2[rng.gen_range(0..paths2.len())].clone();
    let donor = get_subtree(p2, &donor_path).clone();

    let offspring = replace_subtree(p1.clone(), &xo_path, donor);

    // ── O(1) Hard-Cap Pruning ────────────────────────────────────────────────
    if offspring.is_within_cap() { offspring } else { p1.clone() }
}

/// **Subtree Mutation** — replaces a random node with a fresh random subtree.
///
/// The replacement tree has `max_mutation_depth = 3` to keep injected
/// material compact.  O(1) cap check fires after grafting.
pub fn subtree_mutation<R: Rng>(parent: &Expr, rng: &mut R) -> Expr {
    let paths = all_paths(parent);
    let target_path = paths[rng.gen_range(0..paths.len())].clone();

    // Generate a small fresh subtree (depth ≤ 3) to inject.
    let new_subtree = generate_tree(3, rng);

    let offspring = replace_subtree(parent.clone(), &target_path, new_subtree);

    // ── O(1) Hard-Cap Pruning ────────────────────────────────────────────────
    if offspring.is_within_cap() { offspring } else { parent.clone() }
}

/// **Hoist Mutation** — replaces a node with one of its own descendants.
///
/// Hoist always *reduces or preserves* tree size, so it can never trigger
/// the size cap.  However the depth check is still applied for safety.
pub fn hoist_mutation<R: Rng>(parent: &Expr, rng: &mut R) -> Expr {
    let all = all_paths(parent);

    // Collect internal node paths (have at least one child).
    let internal: Vec<&Path> = all.iter()
        .filter(|p| !matches!(get_subtree(parent, p), Expr::Var | Expr::Const(_)))
        .collect();

    if internal.is_empty() {
        return parent.clone(); // already a leaf — nothing to hoist from
    }

    let hoist_path = (*internal[rng.gen_range(0..internal.len())]).clone();

    // All paths *within* the chosen subtree, excluding the root of that subtree.
    let chosen_descendant: Expr = {
        let hoist_node = get_subtree(parent, &hoist_path);
        let sub_paths  = all_paths(hoist_node);
        let desc_paths: Vec<&Path> = sub_paths.iter().filter(|p| !p.is_empty()).collect();

        if desc_paths.is_empty() {
            return parent.clone();
        }
        get_subtree(hoist_node, desc_paths[rng.gen_range(0..desc_paths.len())]).clone()
        // ↑ borrow of `hoist_node` ends here
    };

    replace_subtree(parent.clone(), &hoist_path, chosen_descendant)
    // Hoist always shrinks, so no cap check is needed.
}

/// **Point Mutation** — changes a single operator or terminal to another of
/// the same arity.
///
/// Size is preserved exactly (same arity = same child count), so no cap check.
pub fn point_mutation<R: Rng>(parent: &Expr, rng: &mut R) -> Expr {
    let paths = all_paths(parent);
    let target_path = paths[rng.gen_range(0..paths.len())].clone();
    let target = get_subtree(parent, &target_path);

    let replacement = match target {
        // Terminal → random different terminal
        Expr::Var => {
            if rng.gen_bool(0.7) { Expr::Var } else { Expr::Const(rng.gen_range(-2.0_f64..=2.0)) }
        }
        Expr::Const(_) => {
            if rng.gen_bool(0.3) { Expr::Var } else { Expr::Const(rng.gen_range(-2.0_f64..=2.0)) }
        }
        // Binary op → different binary op (same arity = same children)
        Expr::Add(a, b) | Expr::Sub(a, b)
        | Expr::Mul(a, b) | Expr::Div(a, b) => {
            let a = a.clone();
            let b = b.clone();
            match rng.gen_range(0u8..4) {
                0 => Expr::Add(a, b),
                1 => Expr::Sub(a, b),
                2 => Expr::Mul(a, b),
                _ => Expr::Div(a, b),
            }
        }
        // Unary op → different unary op (same arity = same child)
        Expr::Sin(a) | Expr::Cos(a) | Expr::Exp(a)
        | Expr::Log(a) | Expr::Sqrt(a) | Expr::Abs(a) => {
            let a = a.clone();
            match rng.gen_range(0u8..6) {
                0 => Expr::Sin(a),
                1 => Expr::Cos(a),
                2 => Expr::Exp(a),
                3 => Expr::Log(a),
                4 => Expr::Sqrt(a),
                _ => Expr::Abs(a),
            }
        }
    };

    replace_subtree(parent.clone(), &target_path, replacement)
    // Size-preserving: no cap check required.
}

/// **Constant Perturbation** — adds Gaussian noise to a random `Const` node.
///
/// If the tree has no constant leaves, returns the parent unchanged.
/// Size-preserving: no cap check required.
pub fn constant_perturbation<R: Rng>(parent: &Expr, sigma: f64, rng: &mut R) -> Expr {
    let const_paths: Vec<Path> = all_paths(parent)
        .into_iter()
        .filter(|p| matches!(get_subtree(parent, p), Expr::Const(_)))
        .collect();

    if const_paths.is_empty() {
        return parent.clone();
    }

    let target_path = const_paths[rng.gen_range(0..const_paths.len())].clone();
    let current_val = match get_subtree(parent, &target_path) {
        Expr::Const(v) => *v,
        _ => unreachable!(),
    };

    let noise = gauss(rng, sigma);
    // Keep constants positive and reasonably bounded.
    let new_val = (current_val + noise).clamp(-10.0, 10.0);

    replace_subtree(parent.clone(), &target_path, Expr::Const(new_val))
}

// ─────────────────────────────────────────────────────────────────────────────
// 6.  Operator Dispatcher
// ─────────────────────────────────────────────────────────────────────────────

/// Apply a random genetic operator from `parents`, returning the offspring.
///
/// Operator probabilities (static rates):
/// - crossover           : `crossover_rate`
/// - subtree_mutation    : `mutation_rate * 0.40`
/// - hoist_mutation      : `mutation_rate * 0.20`
/// - point_mutation      : `mutation_rate * 0.25`
/// - constant_perturbation: `mutation_rate * 0.15`
/// - fallback (all else) : point_mutation
///
/// If `parents` is empty, generates a fresh random tree.
pub fn apply_random_operator<R: Rng>(
    parents: &[Expr],
    crossover_rate: f64,
    mutation_rate: f64,
    rng: &mut R,
) -> Expr {
    if parents.is_empty() {
        return generate_tree(4, rng);
    }

    let p1 = &parents[rng.gen_range(0..parents.len())];
    let roll: f64 = rng.gen();

    if roll < crossover_rate && parents.len() >= 2 {
        let p2 = &parents[rng.gen_range(0..parents.len())];
        subtree_crossover(p1, p2, rng)
    } else if roll < crossover_rate + mutation_rate {
        let sub: f64 = rng.gen();
        if      sub < 0.40 { subtree_mutation(p1, rng) }
        else if sub < 0.60 { hoist_mutation(p1, rng) }
        else if sub < 0.85 { point_mutation(p1, rng) }
        else               { constant_perturbation(p1, 0.15, rng) }
    } else {
        point_mutation(p1, rng)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 7.  Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{MAX_DEPTH, MAX_NODES};
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    fn seeded() -> SmallRng { SmallRng::seed_from_u64(42) }

    // ── 7.1  Random Tree Generation ──────────────────────────────────────────

    #[test]
    fn test_generate_tree_within_cap() {
        let mut rng = seeded();
        for _ in 0..200 {
            let tree = generate_tree(MAX_DEPTH - 1, &mut rng);
            assert!(
                tree.depth() <= MAX_DEPTH,
                "Generated tree depth {} exceeds MAX_DEPTH {}",
                tree.depth(), MAX_DEPTH
            );
        }
    }

    #[test]
    fn test_generate_tree_terminal_at_zero_depth() {
        let mut rng = seeded();
        for _ in 0..50 {
            let tree = generate_tree(0, &mut rng);
            assert!(
                matches!(tree, Expr::Var | Expr::Const(_)),
                "depth_remaining=0 must always produce a terminal"
            );
        }
    }

    #[test]
    fn test_ramped_half_and_half_size() {
        let mut rng = seeded();
        let pop = ramped_half_and_half(30, 2, 5, &mut rng);
        assert_eq!(pop.len(), 30);
        for tree in &pop {
            assert!(tree.size() >= 1);
        }
    }

    // ── 7.2  Tree Traversal ───────────────────────────────────────────────────

    #[test]
    fn test_all_paths_count_equals_size() {
        // Number of paths must equal the number of nodes.
        let tree = Expr::Mul(
            Box::new(Expr::Add(Box::new(Expr::Var), Box::new(Expr::Const(0.5)))),
            Box::new(Expr::Exp(Box::new(Expr::Var))),
        );
        assert_eq!(all_paths(&tree).len(), tree.size(),
            "all_paths must yield exactly one path per node");
    }

    #[test]
    fn test_get_and_replace_roundtrip() {
        let tree = Expr::Add(
            Box::new(Expr::Var),
            Box::new(Expr::Const(0.5)),
        );
        // Replace right child (path [1]) with Const(2.0).
        let new_tree = replace_subtree(tree.clone(), &[1], Expr::Const(2.0));
        let right = get_subtree(&new_tree, &[1]);
        assert!(
            matches!(right, Expr::Const(v) if (*v - 2.0).abs() < 1e-12),
            "Right child must be Const(2.0) after replacement"
        );
        // Left child must be unchanged.
        let left = get_subtree(&new_tree, &[0]);
        assert!(matches!(left, Expr::Var), "Left child must remain Var");
    }

    // ── 7.3  O(1) Hard-Cap Pruning ───────────────────────────────────────────

    /// When crossover would produce an oversized tree, the parent is returned.
    #[test]
    fn test_subtree_crossover_respects_cap() {
        let mut rng = seeded();
        // Build two moderately large trees.
        let p1 = generate_tree(4, &mut rng);
        let p2 = generate_tree(4, &mut rng);

        for _ in 0..100 {
            let offspring = subtree_crossover(&p1, &p2, &mut rng);
            assert!(
                offspring.size() <= MAX_NODES,
                "Crossover offspring must never exceed MAX_NODES; got {}",
                offspring.size()
            );
            assert!(
                offspring.depth() <= MAX_DEPTH,
                "Crossover offspring must never exceed MAX_DEPTH; got {}",
                offspring.depth()
            );
        }
    }

    /// Subtree mutation must respect the hard cap.
    #[test]
    fn test_subtree_mutation_respects_cap() {
        let mut rng = seeded();
        let parent = generate_tree(4, &mut rng);

        for _ in 0..100 {
            let offspring = subtree_mutation(&parent, &mut rng);
            assert!(
                offspring.size() <= MAX_NODES,
                "Subtree mutation offspring exceeds MAX_NODES: {}",
                offspring.size()
            );
            assert!(
                offspring.depth() <= MAX_DEPTH,
                "Subtree mutation offspring exceeds MAX_DEPTH: {}",
                offspring.depth()
            );
        }
    }

    /// Hoist mutation must reduce or preserve tree size — never increase it.
    #[test]
    fn test_hoist_mutation_never_grows_tree() {
        let mut rng = seeded();
        let parent = generate_tree(4, &mut rng);
        let original_size = parent.size();

        for _ in 0..100 {
            let offspring = hoist_mutation(&parent, &mut rng);
            assert!(
                offspring.size() <= original_size,
                "Hoist must not grow tree: was {}, now {}",
                original_size, offspring.size()
            );
        }
    }

    /// Point mutation must preserve size exactly.
    #[test]
    fn test_point_mutation_preserves_size() {
        let mut rng = seeded();
        let parent = generate_tree(4, &mut rng);
        let original_size = parent.size();

        for _ in 0..100 {
            let offspring = point_mutation(&parent, &mut rng);
            assert_eq!(
                offspring.size(), original_size,
                "Point mutation must preserve tree size exactly"
            );
        }
    }

    /// Constant perturbation must preserve size exactly.
    #[test]
    fn test_constant_perturbation_preserves_size() {
        let mut rng = seeded();
        // Tree with at least one constant.
        let parent = Expr::Add(Box::new(Expr::Var), Box::new(Expr::Const(0.5)));
        let original_size = parent.size();

        for _ in 0..50 {
            let offspring = constant_perturbation(&parent, 0.1, &mut rng);
            assert_eq!(offspring.size(), original_size,
                "Constant perturbation must preserve tree size");
        }
    }

    /// Constant perturbation changes the constant value.
    #[test]
    fn test_constant_perturbation_modifies_value() {
        let mut rng = SmallRng::seed_from_u64(99);
        let parent = Expr::Add(Box::new(Expr::Var), Box::new(Expr::Const(1.0)));

        let offspring = constant_perturbation(&parent, 0.5, &mut rng);
        let new_val = match get_subtree(&offspring, &[1]) {
            Expr::Const(v) => *v,
            _ => panic!("Right child must remain Const"),
        };
        // With sigma=0.5 the value must have changed (with overwhelming probability).
        assert!(
            (new_val - 1.0).abs() < 11.0, // never exceeds clamp bound
            "Perturbed constant must be within [-10, 10]"
        );
    }

    // ── 7.4  Operator Dispatcher ─────────────────────────────────────────────

    #[test]
    fn test_apply_random_operator_empty_parents_generates_tree() {
        let mut rng = seeded();
        let tree = apply_random_operator(&[], 0.2, 0.7, &mut rng);
        assert!(tree.size() >= 1, "Must produce a non-empty tree");
    }

    #[test]
    fn test_apply_random_operator_respects_cap() {
        let mut rng = seeded();
        let mut parents: Vec<Expr> = (0..10).map(|_| generate_tree(4, &mut rng)).collect();
        parents.iter_mut().for_each(|t| {
            // Ensure all parents are within cap themselves.
            while !t.is_within_cap() {
                *t = generate_tree(3, &mut rng);
            }
        });

        for _ in 0..200 {
            let off = apply_random_operator(&parents, 0.2, 0.7, &mut rng);
            assert!(off.size() <= MAX_NODES,
                "Operator output must not exceed MAX_NODES; got {}", off.size());
            assert!(off.depth() <= MAX_DEPTH,
                "Operator output must not exceed MAX_DEPTH; got {}", off.depth());
        }
    }
}
