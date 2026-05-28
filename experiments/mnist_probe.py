import os
import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.symbolr.core.evaluator import BaseEvaluator
from src.symbolr.core.bridge import RustEvolutionBridge
from src.symbolr.artifacts.pytorch_export import export_to_pytorch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
        
    def forward(self, x):
        return self.net(x)

def get_dataloaders():
    # Download into data/mnist_test
    os.makedirs('data/mnist_test', exist_ok=True)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    
    trainset = torchvision.datasets.MNIST(root='./data/mnist_test', train=True, download=True, transform=transform)
    valset = torchvision.datasets.MNIST(root='./data/mnist_test', train=False, download=True, transform=transform)
    
    # Hardware Optimization: Giant batch size of 2048 + pin_memory for fast CPU->GPU transfer
    trainloader = DataLoader(trainset, batch_size=2048, shuffle=True, pin_memory=True)
    valloader = DataLoader(valset, batch_size=2048, shuffle=False, pin_memory=True)
    
    return trainloader, valloader

def prefix_to_infix(prefix_str: str) -> str:
    """Converts a SymboLR prefix notation string into human-readable math."""
    tokens = prefix_str.strip().split()
    stack = []
    binary_ops = {'+', '-', '*', '/'}
    unary_ops = {'sin', 'cos', 'exp', 'log', 'abs', 'sqrt'}
    
    for token in reversed(tokens):
        if token in binary_ops:
            if len(stack) < 2: return prefix_str
            left = stack.pop()
            right = stack.pop()
            stack.append(f"({left} {token} {right})")
        elif token in unary_ops:
            if len(stack) < 1: return prefix_str
            operand = stack.pop()
            stack.append(f"{token}({operand})")
        else:
            try:
                val = float(token)
                stack.append(f"{val:.4f}")
            except ValueError:
                stack.append(token)
                
    return stack[0] if len(stack) == 1 else prefix_str

# -----------------------------------------------------------------------------------
# Manual Training 
# -----------------------------------------------------------------------------------

def train_manual_baselines(trainloader, valloader, epochs=5):
    print("\n=== Training Baseline Neural Networks (Fixed LR=1e-3) ===")
    results = {}
    optimizers_to_test = {
        "Adam": optim.Adam,
        "SGD": optim.SGD,
        "RMSprop": optim.RMSprop,
        "AdamW": optim.AdamW
    }
    
    for opt_name, OptClass in optimizers_to_test.items():
        print(f"\n--- Training {opt_name} Baseline ---")
        model = SimpleNN().to(DEVICE)
        optimizer = OptClass(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        scaler = torch.amp.GradScaler('cuda')
        
        start_time = time.time()
        for epoch in range(epochs):
            model.train()
            for inputs, labels in tqdm(trainloader, desc=f"{opt_name} Epoch {epoch+1}/{epochs}", leave=False):
                inputs, labels = inputs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in valloader:
                inputs, labels = inputs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        avg_loss = val_loss / len(valloader)
        accuracy = 100 * correct / total
        print(f"{opt_name} finished in {time.time()-start_time:.2f}s")
        print(f"{opt_name} Validation Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")
        results[opt_name] = (avg_loss, accuracy)
        
    return results


# -----------------------------------------------------------------------------------
# SymboLR Training
# -----------------------------------------------------------------------------------

class FastProxyEvaluator(BaseEvaluator):
    """A proxy evaluator for GP that trains fairly fast (1 epoch, 10k samples) to get a stable signal"""
    def __init__(self, epochs=20):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        trainset = torchvision.datasets.MNIST(root='./data/mnist_test', train=True, download=False, transform=transform)
        valset = torchvision.datasets.MNIST(root='./data/mnist_test', train=False, download=False, transform=transform)
        
        # Proxy dataset size: scaled to 20000 samples for the 45-minute mega-run
        self.trainloader = DataLoader(Subset(trainset, range(20000)), batch_size=2048, shuffle=True, pin_memory=True)
        self.valloader = DataLoader(Subset(valset, range(4000)), batch_size=2048, shuffle=False, pin_memory=True)
        self.epochs = epochs
        self.total_steps = len(self.trainloader) * self.epochs
        self.t = torch.linspace(0.0, 1.0, self.total_steps, device=DEVICE)

    def _parse_and_evaluate_ast(self, prefix_str: str) -> torch.Tensor:
        tokens = prefix_str.strip().split()
        stack = []
        for token in reversed(tokens):
            if token in ['x', 't']: stack.append(self.t)
            elif token == '+': stack.append(stack.pop() + stack.pop())
            elif token == '-': stack.append(stack.pop() - stack.pop())
            elif token == '*': stack.append(stack.pop() * stack.pop())
            elif token == '/':
                left, right = stack.pop(), stack.pop()
                stack.append(left / (torch.abs(right) + 1e-6))
            elif token == 'sin': stack.append(torch.sin(stack.pop()))
            elif token == 'cos': stack.append(torch.cos(stack.pop()))
            elif token == 'exp': stack.append(torch.exp(torch.clamp(stack.pop(), max=20.0)))
            elif token == 'log': stack.append(torch.log(torch.abs(stack.pop()) + 1e-6))
            elif token == 'abs': stack.append(torch.abs(stack.pop()))
            else:
                try: stack.append(torch.full_like(self.t, float(token)))
                except ValueError: stack.append(torch.full_like(self.t, 0.01))
        return torch.clamp(stack[0], 1e-6, 10.0)

    def evaluate(self, formulas: list[str]) -> list[float]:
        half_steps = self.total_steps // 2
        halfway_results = []
        
        # Phase 1: Train all formulas for 50% of the epoch
        for idx, formula in enumerate(tqdm(formulas, desc="Phase 1: Successive Halving (50% steps)", leave=False)):
            try:
                lr_array = self._parse_and_evaluate_ast(formula)
                model = SimpleNN().to(DEVICE)
                optimizer = optim.Adam(model.parameters(), lr=1e-3)
                criterion = nn.CrossEntropyLoss()
                scaler = torch.amp.GradScaler('cuda')
                
                model.train()
                step_idx = 0
                for epoch in range(self.epochs):
                    for inputs, labels in self.trainloader:
                        if step_idx >= half_steps: break
                        inputs, labels = inputs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = 1e-3 * lr_array[step_idx].item()
                        optimizer.zero_grad(set_to_none=True)
                        with torch.amp.autocast('cuda'):
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                        step_idx += 1
                    if step_idx >= half_steps: break
                
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for inputs, labels in self.valloader:
                        inputs, labels = inputs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
                        with torch.amp.autocast('cuda'):
                            outputs = model(inputs)
                            val_loss += criterion(outputs, labels).item()
                avg_val_loss = val_loss / len(self.valloader)
                
                if math.isfinite(avg_val_loss):
                    # Save state strictly on CPU to avoid VRAM explosion
                    state = {k: v.cpu() for k, v in model.state_dict().items()}
                    halfway_results.append((idx, avg_val_loss, state, optimizer.state_dict(), lr_array))
            except Exception:
                pass

        # Phase 2: Cull bottom 50% and finish the survivors
        halfway_results.sort(key=lambda x: x[1])
        survivors = halfway_results[:max(1, len(halfway_results)//2)]
        final_losses = [float('inf')] * len(formulas)
        
        for idx, halfway_loss, state, opt_state, lr_array in tqdm(survivors, desc="Phase 2: Finishing Top 50%", leave=False):
            try:
                model = SimpleNN().to(DEVICE)
                model.load_state_dict({k: v.to(DEVICE) for k, v in state.items()})
                optimizer = optim.Adam(model.parameters(), lr=1e-3)
                optimizer.load_state_dict(opt_state)
                criterion = nn.CrossEntropyLoss()
                scaler = torch.amp.GradScaler('cuda')
                
                model.train()
                step_idx = 0
                for epoch in range(self.epochs):
                    for inputs, labels in self.trainloader:
                        if step_idx < half_steps: 
                            step_idx += 1
                            continue # Skip first half
                        inputs, labels = inputs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = 1e-3 * lr_array[step_idx].item()
                        optimizer.zero_grad(set_to_none=True)
                        with torch.amp.autocast('cuda'):
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                        step_idx += 1
                
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for inputs, labels in self.valloader:
                        inputs, labels = inputs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
                        with torch.amp.autocast('cuda'):
                            outputs = model(inputs)
                            val_loss += criterion(outputs, labels).item()
                avg_val_loss = val_loss / len(self.valloader)
                
                if math.isfinite(avg_val_loss):
                    parsimony = 0.001 * len(formulas[idx].split())
                    final_losses[idx] = avg_val_loss + parsimony
            except Exception:
                pass
                
        return final_losses

def discover_schedule_with_symbolr(epochs=20):
    print(f"=== Booting SymboLR to Discover Optimal LR Schedule (Horizon: {epochs} Epochs) ===")
    evaluator = FastProxyEvaluator(epochs=epochs)
    bridge = RustEvolutionBridge(
        eval_callback=evaluator.evaluate,
        max_generations=40, # 45-minute Mega Run Scale
        pop_size=50, 
        seed=42
    )
    
    best_formula = ""
    for result in bridge.stream():
        print(f"Gen {result.generation_number:02d} | Surrogate Loss: {result.best_mse:.4f} | Formula: {result.top_formula_prefix}")
        best_formula = result.top_formula_prefix
        
    print(f"Discovered Best Equation: {best_formula}\n")
    return best_formula

def train_with_symbolr(trainloader, valloader, best_formula, epochs=5):
    print("=== Training Neural Network with SymboLR Discovered Schedule ===")
    model = SimpleNN().to(DEVICE)
    # Using Adam with 1e-3 base LR. The discovered formula acts as a dynamic multiplier!
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda')
    
    # Pre-compute the LR curve over the full trainloader sequence
    total_steps = len(trainloader) * epochs
    t = torch.linspace(0.0, 1.0, total_steps, device=DEVICE)
    
    # We parse the formula using the full timeline
    tokens = best_formula.strip().split()
    stack = []
    for token in reversed(tokens):
        if token in ['x', 't']: stack.append(t)
        elif token == '+': stack.append(stack.pop() + stack.pop())
        elif token == '-': stack.append(stack.pop() - stack.pop())
        elif token == '*': stack.append(stack.pop() * stack.pop())
        elif token == '/':
            left, right = stack.pop(), stack.pop()
            stack.append(left / (torch.abs(right) + 1e-6))
        elif token == 'sin': stack.append(torch.sin(stack.pop()))
        elif token == 'cos': stack.append(torch.cos(stack.pop()))
        elif token == 'exp': stack.append(torch.exp(torch.clamp(stack.pop(), max=20.0)))
        elif token == 'log': stack.append(torch.log(torch.abs(stack.pop()) + 1e-6))
        elif token == 'abs': stack.append(torch.abs(stack.pop()))
        else:
            try: stack.append(torch.full_like(t, float(token)))
            except ValueError: stack.append(torch.full_like(t, 0.01))
            
    full_lr_array = torch.clamp(stack[0], 1e-6, 10.0)
    
    start_time = time.time()
    step_idx = 0
    for epoch in range(epochs):
        model.train()
        for inputs, labels in tqdm(trainloader, desc=f"SymboLR Epoch {epoch+1}/{epochs}", leave=False):
            inputs, labels = inputs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            
            # Apply dynamic LR multiplier on top of Adam's 1e-3 base
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-3 * full_lr_array[step_idx].item()
                
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            step_idx += 1
            
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in valloader:
            inputs, labels = inputs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    avg_loss = val_loss / len(valloader)
    accuracy = 100 * correct / total
    print(f"SymboLR Training finished in {time.time()-start_time:.2f}s")
    print(f"Validation Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%\n")
    return avg_loss, accuracy

if __name__ == "__main__":
    print(f"Running Experiment on Device: {DEVICE}")
    trainloader, valloader = get_dataloaders()
    
    # 1. Train Manual Baselines
    baseline_results = train_manual_baselines(trainloader, valloader, epochs=20)
    
    # 2. Discover Optimal Schedule via SymboLR (Simulating 20 epochs on proxy)
    best_formula = discover_schedule_with_symbolr(epochs=20)
    human_readable = prefix_to_infix(best_formula)
    print(f"\n[Success] SymboLR Discovered: LR = 1e-3 * {human_readable}\n")
    
    # 3. Train with Discovered Schedule
    symbolr_loss, symbolr_acc = train_with_symbolr(trainloader, valloader, best_formula, epochs=20)
    
    print("================================================================")
    print("                      FINAL COMPARISON                          ")
    print("================================================================")
    for opt_name, (loss, acc) in baseline_results.items():
        print(f"Baseline {opt_name.ljust(10)} (LR=1e-3):   Loss: {loss:.4f}  | Accuracy: {acc:.2f}%")
    print("----------------------------------------------------------------")
    print(f"SymboLR Dynamic (Adam base): Loss: {symbolr_loss:.4f}  | Accuracy: {symbolr_acc:.2f}%")
    print(f"Discovered Formula: LR = 1e-3 * {human_readable}")
    print("================================================================")
    
    best_baseline = min(baseline_results.items(), key=lambda x: x[1][0])
    if symbolr_loss < best_baseline[1][0]:
        print(f"WINNER: SymboLR successfully beat ALL baselines, including {best_baseline[0]}!")
    else:
        print(f"WINNER: {best_baseline[0]} still won! SymboLR might need more generations or a larger proxy dataset.")
