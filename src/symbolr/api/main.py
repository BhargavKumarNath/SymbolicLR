import os
import json
import asyncio
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from rich.console import Console

from src.symbolr.engine.bridge import RustEvolutionBridge

console = Console()

app = FastAPI(
    title="SymboLR Local Compute Engine (SSE Enabled)",
    description="High-performance FastAPI bridge connecting the React dashboard to the Rust/PyO3 Quality-Diversity Engine.",
    version="3.0.0"  # Phase 4 update
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/stream-evolve")
async def stream_evolve(population_size: int = 50, generations: int = 50, target_epochs: int = 100):
    """
    Triggers the Thick Rust Core in a background thread and streams Live Server-Sent Events (SSE).
    """
    console.rule("[bold cyan]SymboLR: Incoming Streaming API Request (Rust Core)[/bold cyan]")
    
    # 1. Ingest Surrogate Dataset
    data_path = os.path.join("data", "surrogate_labels.npy")
    if not os.path.exists("data"):
        os.makedirs("data", exist_ok=True)
        
    if not os.path.exists(data_path):
        console.print(f"[dim]Mocking surrogate dataset at {data_path}...[/dim]")
        dummy_labels = np.random.rand(target_epochs).astype(np.float64)
        np.save(data_path, dummy_labels)
        
    # Zero-copy memory mapping of the surrogate dataset
    console.print(f"[dim]Ingesting dataset via mmap: {data_path}[/dim]")
    probe_labels = np.load(data_path, mmap_mode='r')
    
    # Ensure C-contiguous for Rust (if it isn't already)
    if not probe_labels.flags['C_CONTIGUOUS']:
        probe_labels = np.ascontiguousarray(probe_labels)

    # 2. Instantiate Rust Evolution Engine
    try:
        from src.symbolr.torch_impl.evaluator import CUDABatchEvaluator
        evaluator = CUDABatchEvaluator(data_labels=probe_labels)
        
        bridge = RustEvolutionBridge(
            eval_callback=evaluator.evaluate,
            max_generations=generations,
            pop_size=population_size,
            seed=42
        )
    except Exception as e:
        console.print(f"[bold red]Failed to initialize Rust bridge:[/bold red] {e}")
        raise HTTPException(status_code=500, detail=str(e))

    # 3. Stream Generator
    async def event_generator():
        stream_iter = bridge.stream()
        
        while True:
            try:
                # Execute one generation (parallel Rust execution) without blocking the asyncio loop
                result = await asyncio.to_thread(next, stream_iter, None)
                
                if result is None:
                    # Max generations reached
                    yield f"event: COMPLETE\ndata: {json.dumps({'status': 'done'})}\n\n"
                    break
                
                # Format to match the React frontend's schema expectations
                payload = {
                    "generation": result.generation_number,
                    "best_loss": result.best_mse,
                    "top_formula_latex": result.top_formula_latex,
                    "archive_size": result.archive_size
                }
                
                console.print(f"[dim]Processed Gen {result.generation_number} | Best MSE: {result.best_mse:.4f} | Archive: {result.archive_size}[/dim]")
                
                yield f"data: {json.dumps(payload)}\n\n"
                
            except Exception as e:
                console.print(f"[bold red]Stream error:[/bold red] {e}")
                yield f"event: ERROR\ndata: {json.dumps({'detail': str(e)})}\n\n"
                break

    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    console.print("[bold magenta]Booting SymboLR Streaming Compute Hub on port 8000...[/bold magenta]")
    uvicorn.run("src.symbolr.api.main:app", host="0.0.0.0", port=8000, reload=True)
