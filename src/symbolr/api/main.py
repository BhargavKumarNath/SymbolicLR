import os
import json
import asyncio
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from rich.console import Console

from src.symbolr.core.bridge import RustEvolutionBridge

console = Console()

app = FastAPI(
    title="SymboLR Compute Engine",
    description="FastAPI bridge connecting the React dashboard to the Rust/PyO3 evolutionary engine.",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/stream-evolve")
async def stream_evolve(
    population_size: int = 50,
    generations: int = 50,
    target_epochs: int = 100,
    seed: int = 42,
):
    """
    Stream evolution telemetry as Server-Sent Events.

    Each SSE event contains a GenerationResult JSON payload.
    The seed parameter is forwarded to the Rust core — runs with the same
    seed, population size, and generation count are fully reproducible.
    """
    console.rule("[bold cyan]SymboLR: Incoming Evolution Request[/bold cyan]")

    # Surrogate dataset — seeded so the fitness landscape is reproducible
    data_path = os.path.join("data", "surrogate_labels.npy")
    os.makedirs("data", exist_ok=True)

    if not os.path.exists(data_path):
        console.print(f"[dim]Generating seeded surrogate dataset at {data_path}...[/dim]")
        rng = np.random.RandomState(seed)
        dummy_labels = rng.rand(target_epochs).astype(np.float64)
        np.save(data_path, dummy_labels)

    probe_labels = np.load(data_path, mmap_mode='r')
    if not probe_labels.flags['C_CONTIGUOUS']:
        probe_labels = np.ascontiguousarray(probe_labels)

    try:
        from src.symbolr.torch_impl.evaluator import CUDABatchEvaluator
        evaluator = CUDABatchEvaluator(data_labels=probe_labels)

        bridge = RustEvolutionBridge(
            eval_callback=evaluator.evaluate,
            max_generations=generations,
            pop_size=population_size,
            seed=seed,
        )
    except Exception as e:
        console.print(f"[bold red]Bridge init failed:[/bold red] {e}")
        raise HTTPException(status_code=500, detail=str(e))

    async def event_generator():
        stream_iter = bridge.stream()
        while True:
            try:
                result = await asyncio.to_thread(next, stream_iter, None)
                if result is None:
                    yield f"event: COMPLETE\ndata: {json.dumps({'status': 'done'})}\n\n"
                    break

                payload = {
                    "generation":       result.generation_number,
                    "best_loss":        result.best_mse,
                    "average_loss":     result.average_mse,
                    "top_formula_latex": result.top_formula_latex,
                    "top_formula_prefix": result.top_formula_prefix,
                    "archive_size":     result.archive_size,
                    "new_entries":      result.new_entries,
                    "gen_time_ms":      result.gen_time_ms,
                }
                console.print(
                    f"[dim]Gen {result.generation_number:02d} | "
                    f"best={result.best_mse:.4f} | archive={result.archive_size}[/dim]"
                )
                yield f"data: {json.dumps(payload)}\n\n"

            except Exception as e:
                console.print(f"[bold red]Stream error:[/bold red] {e}")
                yield f"event: ERROR\ndata: {json.dumps({'detail': str(e)})}\n\n"
                break

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.symbolr.api.main:app", host="0.0.0.0", port=8000, reload=True)
