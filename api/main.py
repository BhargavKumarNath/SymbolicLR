import os
import json
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from rich.console import Console

# Import the core evolution engine from the root benchmark module
from benchmark import run_evolution

console = Console()

app = FastAPI(
    title="SymboLR Local Compute Engine (SSE Enabled)",
    description="High-performance FastAPI bridge connecting the React dashboard to the Rust/PyO3 Quality-Diversity Engine.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/stream-evolve")
async def stream_evolve(population_size: int, generations: int, target_epochs: int):
    """
    Triggers the engine in a background thread and streams Live Server-Sent Events (SSE).
    """
    console.rule("[bold cyan]SymboLR: Incoming Streaming API Request[/bold cyan]")
    
    queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    SEED = 42

    def on_generation(metrics: dict):
        # Thread-safe handoff to the asyncio event loop
        loop.call_soon_threadsafe(queue.put_nowait, ("metrics", metrics))
        
    def worker():
        try:
            # Execute heavy evolution loop
            run_evolution(
                generations=generations,
                pop_size=population_size,
                epochs=target_epochs,
                workers=4,
                seed=SEED,
                console=console,
                wandb_enabled=False,
                generation_callback=on_generation
            )
            
            # Send final completion artifact
            artifact_path = os.path.join("results", f"run_seed{SEED}.json")
            if os.path.exists(artifact_path):
                with open(artifact_path, "r") as f:
                    final_data = json.load(f)
                loop.call_soon_threadsafe(queue.put_nowait, ("complete", final_data))
            else:
                loop.call_soon_threadsafe(queue.put_nowait, ("error", "Artifact not found"))
        except Exception as e:
            loop.call_soon_threadsafe(queue.put_nowait, ("error", str(e)))

    # Dispatch the synchronous worker to a background thread
    asyncio.create_task(asyncio.to_thread(worker))

    async def event_generator():
        while True:
            msg_type, data = await queue.get()
            if msg_type == "metrics":
                yield f"data: {json.dumps(data)}\n\n"
            elif msg_type == "complete":
                yield f"event: COMPLETE\ndata: {json.dumps(data)}\n\n"
                break
            elif msg_type == "error":
                yield f"event: ERROR\ndata: {json.dumps({'detail': data})}\n\n"
                break

    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    console.print("[bold magenta]Booting SymboLR Streaming Compute Hub on port 8000...[/bold magenta]")
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
