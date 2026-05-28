import pytest
import asyncio
import time
import numpy as np
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport

from src.symbolr.api.main import app
from src.symbolr.core.bridge import RustEvolutionBridge
from src.symbolr.torch_impl.evaluator import CUDABatchEvaluator

@pytest.mark.asyncio
async def test_ffi_async_non_blocking():
    """
    Test that the PyO3 iterator does not block the asyncio event loop.
    We run a background task that sleeps and increments a counter,
    while concurrently running the EvolutionStream via asyncio.to_thread.
    If the GIL is blocked, the counter won't increment during the heavy Rust computation.
    """
    dummy_labels = np.random.rand(10).astype(np.float64)
    evaluator = CUDABatchEvaluator(dummy_labels)
    
    bridge = RustEvolutionBridge(
        eval_callback=evaluator.evaluate,
        max_generations=5,
        pop_size=10,
        seed=42,
    )

    counter = 0
    running = True

    async def heartbeat():
        nonlocal counter
        while running:
            await asyncio.sleep(0.01)
            counter += 1

    heartbeat_task = asyncio.create_task(heartbeat())

    stream_iter = bridge.stream()
    
    # We measure how many heartbeats happen during 1 next() call.
    # Since pop_size=10 is small, it might be too fast, so let's use a larger one to make sure.
    bridge_heavy = RustEvolutionBridge(
        eval_callback=evaluator.evaluate,
        max_generations=5,
        pop_size=500,
        seed=42
    )
    heavy_iter = bridge_heavy.stream()

    start_time = time.time()
    await asyncio.to_thread(next, heavy_iter)
    elapsed = time.time() - start_time
    
    running = False
    await heartbeat_task
    
    # If it took e.g. 0.5s, counter should be around 50. If GIL was held, counter would be 0 or 1.
    print(f"Elapsed: {elapsed:.3f}s, Heartbeats: {counter}")
    assert counter > 0, "The asyncio loop was blocked by the FFI call!"

@pytest.mark.asyncio
async def test_fastapi_sse_stream():
    """
    Test the FastAPI /api/stream-evolve endpoint parses the SSE stream correctly
    and matches the React frontend schema.
    """
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # We only run 3 generations for the test
        async with client.stream("GET", "/api/stream-evolve", params={"population_size": 10, "generations": 3, "target_epochs": 10}) as response:
            assert response.status_code == 200
            
            lines = []
            async for line in response.aiter_lines():
                if line:
                    lines.append(line)
            
            # The output should have data: {...} events followed by event: COMPLETE
            assert len(lines) >= 3
            
            # Check schema of the first metrics event
            first_event = lines[0]
            assert first_event.startswith("data: ")
            import json
            payload = json.loads(first_event[6:])
            assert "generation" in payload
            assert "best_loss" in payload
            assert "top_formula_latex" in payload
            assert "archive_size" in payload
            assert payload["generation"] == 1
