import asyncio
import os
import shutil
import logging
from pathlib import Path
from typing import Callable

from glmocr import parse

from . import store

logger = logging.getLogger("ocr-worker")

RESULTS_DIR = Path(__file__).parent.parent / "data" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OCR_TIMEOUT = 300
MAX_CONCURRENT = int(os.environ.get("OCR_MAX_CONCURRENT", "3"))


async def _process_task(task: dict, on_update: Callable[[str], None]):
    task_id = task["task_id"]
    file_path = task["file_path"]
    upload_dir = Path(file_path).parent
    output_dir = RESULTS_DIR / task_id

    logger.info("Processing task %s: %s", task_id, task["file_name"])
    on_update(task_id)

    try:
        pipeline_result = await asyncio.wait_for(
            asyncio.to_thread(parse, file_path),
            timeout=OCR_TIMEOUT,
        )
        pipeline_result.save(output_dir=str(output_dir))

        result_data = {
            "markdown": pipeline_result.markdown_result or "",
            "json": pipeline_result.json_result,
            "file_name": task["file_name"],
        }

        store.complete_task(task_id, result_data)
        logger.info("Task %s completed", task_id)
    except Exception as e:
        logger.exception("Task %s failed", task_id)
        store.fail_task(task_id, str(e))
    finally:
        on_update(task_id)
        shutil.rmtree(upload_dir, ignore_errors=True)


async def worker_loop(on_update: Callable[[str], None] = lambda _: None):
    logger.info("OCR worker started (max_concurrent=%d)", MAX_CONCURRENT)
    sem = asyncio.Semaphore(MAX_CONCURRENT)
    pending: set[asyncio.Task] = set()

    async def _run(task: dict):
        async with sem:
            await _process_task(task, on_update)

    while True:
        task = await asyncio.to_thread(store.claim_next_pending)
        if task:
            t = asyncio.create_task(_run(task))
            pending.add(t)
            t.add_done_callback(pending.discard)
        else:
            await asyncio.sleep(1)
