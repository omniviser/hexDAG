#!/usr/bin/env python
"""hexDAG Approval Flow Demo: notify → suspend → human decision → resume.

Demonstrates async human-in-the-loop with three framework features:

- **Notification port** — reaches the approver (console adapter here;
  swap in Slack/email adapters for production)
- **WaitNode** — parks the pipeline until the external decision arrives
- **resume_with_event()** — resumes from the checkpoint with the human
  decision injected as the wait node's output

Run:
    uv run python examples/approval_flow/run_approval_flow.py

In this demo the "human" approves immediately in-process. In production
the suspend and resume happen in different processes (web request, queue
worker) — use a file/SQLite/Redis checkpoint storage instead of
InMemoryMemory and the same code works across restarts.
"""

import asyncio
import sys
from pathlib import Path

EXAMPLE_DIR = Path(__file__).parent

# Make approval_helpers importable by module path (used in the YAML)
sys.path.insert(0, str(EXAMPLE_DIR))
# Add project root for hexdag itself when running from a checkout
sys.path.insert(0, str(EXAMPLE_DIR.parent.parent))

from hexdag.kernel.pipeline_runner import PipelineRunner  # noqa: E402
from hexdag.stdlib.adapters.memory import InMemoryMemory  # noqa: E402

PIPELINE = EXAMPLE_DIR / "approval_pipeline.yaml"


async def main() -> None:
    # Checkpoint storage makes suspend/resume (and crash recovery) work.
    # Use FileMemoryAdapter or SQLiteMemoryAdapter to survive restarts.
    checkpoint_storage = InMemoryMemory()
    runner = PipelineRunner(checkpoint_storage=checkpoint_storage)

    # ── 1. Run until the pipeline suspends at the wait node ──────────
    print("=== Phase 1: run until approval is required ===")
    result = await runner.run(
        PIPELINE,
        input_data={"order_id": "ORD-1042", "amount": 1850},
    )

    assert result.status == "suspended", f"expected suspension, got {result.status}"
    print(f"\nPipeline suspended (run_id={result.run_id})")
    print(f"Waiting for event: {result.suspend_metadata.get('event_key')}\n")

    # ── 2. The human decides (hours later, in another process...) ────
    print("=== Phase 2: human approves ===")
    decision = {
        "approved": True,
        "approver": "j.kwapisz@omniviser.ai",
        "comment": "Margins check out.",
    }

    # ── 3. Resume with the decision injected as the wait node output ─
    resumed = await runner.resume_with_event(
        PIPELINE,
        run_id=result.run_id,
        event_data=decision,
    )

    print(f"\nFinal status: {resumed.node_results['finalize']['status']}")
    print(f"Approved by:  {resumed.node_results['finalize']['approver']}")
    print(f"Comment:      {resumed.node_results['finalize']['comment']}")


if __name__ == "__main__":
    asyncio.run(main())
