import numpy as np
import hashlib
from src.symbolr.engine.evaluator import BaseEvaluator
import symbolr_rust
from config.settings import get_config

class SyntheticEvaluator(BaseEvaluator):
    """
    Simulates training on a noisy quadratic loss landscape.
    This provides realistic loss curves without requiring PyTorch or a GPU.
    Used primarily for the Online Showcase mode and rapid local testing.
    """
    def __init__(self, time_steps: int = 100):
        self.t_array = np.linspace(0.0, 1.0, time_steps, dtype=np.float64)

    def evaluate(self, formulas: list[str]) -> list[float]:
        # 1. Ask Rust to compute the LR arrays for the formulas
        try:
            schedules_list = symbolr_rust.evaluate_batch(formulas, self.t_array)
        except Exception as e:
            print(f"Failed to evaluate schedules natively: {e}")
            return [float("inf")] * len(formulas)

        fitnesses = []
        for schedule in schedules_list:
            fitnesses.append(self._simulate(np.array(schedule, dtype=np.float64)))
            
        return fitnesses

    def _simulate(self, lr_schedule: np.ndarray) -> float:
        cfg = get_config()
        
        if not np.all(np.isfinite(lr_schedule)) or np.all(lr_schedule < 1e-7) or np.any(lr_schedule > 10.0):
            return float("inf")

        n_steps = len(lr_schedule)
        if n_steps == 0: return float("inf")

        raw_mean = np.mean(np.abs(lr_schedule))
        if raw_mean > 1e-7:
            scale_factor = 0.01 / raw_mean
            normalized = lr_schedule * scale_factor
        else:
            normalized = np.full(n_steps, 0.01)
            
        normalized = np.clip(normalized, 1e-5, 0.1)

        schedule_hash = int(hashlib.md5(lr_schedule.tobytes()).hexdigest()[:8], 16)
        n_evals = getattr(cfg, 'synth_n_evaluations', 3)
        ensemble_losses = []
        
        n_dims = getattr(cfg, 'synth_n_dims', 5)
        noise_scale = getattr(cfg, 'synth_noise_scale', 0.02)
        curvatures = np.array([0.5, 1.0, 2.0, 4.0, 8.0])[:n_dims]
        if len(curvatures) < n_dims:
            curvatures = np.pad(curvatures, (0, n_dims - len(curvatures)), constant_values=1.0)
        
        for eval_idx in range(n_evals):
            rng = np.random.RandomState((schedule_hash + eval_idx) % (2**32 - 1))
            
            w = rng.randn(n_dims) * 2.0
            w_star = np.zeros(n_dims)
            losses = np.zeros(n_steps)
            
            best_loss = float("inf")
            failed = False
            
            for step in range(n_steps):
                lr = float(normalized[step])
                diff = w - w_star
                loss = 0.5 * np.sum(curvatures * diff ** 2)
                
                noisy_loss = loss + rng.randn() * noise_scale * (1.0 + loss)
                current_loss = max(0.0, noisy_loss)
                losses[step] = current_loss
                best_loss = min(best_loss, current_loss)
                
                if loss > 1000.0 or not np.isfinite(loss):
                    failed = True
                    break
                    
                grad = curvatures * diff
                grad_noise = rng.randn(n_dims) * 0.1 * np.sqrt(1.0 + np.abs(loss))
                w = w - lr * (grad + grad_noise)
                w = np.clip(w, -50.0, 50.0)
                
            if failed:
                ensemble_losses.append(float('inf'))
                continue
                
            final_diff = w - w_star
            val_loss = 0.5 * np.sum(curvatures * final_diff ** 2)
            combined = 0.6 * val_loss + 0.4 * best_loss
            ensemble_losses.append(combined)

        if all(l == float('inf') for l in ensemble_losses):
            return float('inf')
            
        avg_loss = float(np.mean([l for l in ensemble_losses if np.isfinite(l)]))
        
        lr_std = np.std(lr_schedule)
        if lr_std < 1e-6:
            avg_loss += 0.5
            
        if lr_schedule[-1] > lr_schedule[0] * 1.5:
            avg_loss += 0.3
            
        return float(np.clip(avg_loss, 0.0, 50.0))
