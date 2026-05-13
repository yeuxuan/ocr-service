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

OCR_TIMEOUT_IMAGE = int(os.environ.get("OCR_TIMEOUT_IMAGE", "300"))
OCR_TIMEOUT_PDF = int(os.environ.get("OCR_TIMEOUT_PDF", "1800"))
# PP-DocLayoutV3 (PyTorch) 并发初始化存在 meta tensor 竞态，必须串行
MAX_CONCURRENT = int(os.environ.get("OCR_MAX_CONCURRENT", "1"))

_PDF_SUFFIXES = {".pdf"}


def _timeout_for(file_name: str) -> int:
    suffix = Path(file_name).suffix.lower()
    return OCR_TIMEOUT_PDF if suffix in _PDF_SUFFIXES else OCR_TIMEOUT_IMAGE


def _error_message(e: Exception) -> str:
    msg = str(e)
    if msg:
        return msg
    return f"{type(e).__name__}: {type(e).__doc__ or 'no details'}"


async def _process_task(task: dict, on_update: Callable[[str], None]):
    task_id = task["task_id"]
    file_path = task["file_path"]
    file_name = task["file_name"]
    upload_dir = Path(file_path).parent
    output_dir = RESULTS_DIR / task_id
    timeout = _timeout_for(file_name)

    logger.info("Processing task %s: %s (timeout=%ds)", task_id, file_name, timeout)
    on_update(task_id)

    try:
        pipeline_result = await asyncio.wait_for(
            asyncio.to_thread(parse, file_path),
            timeout=timeout,
        )
        pipeline_result.save(output_dir=str(output_dir))

        result_data = {
            "markdown": pipeline_result.markdown_result or "",
            "json": pipeline_result.json_result,
            "file_name": file_name,
        }

        store.complete_task(task_id, result_data)
        logger.info("Task %s completed", task_id)
    except asyncio.TimeoutError:
        error = f"OCR timeout after {timeout}s for {file_name}"
        logger.error("Task %s timed out: %s", task_id, error)
        store.fail_task(task_id, error)
    except Exception as e:
        error = _error_message(e)
        logger.exception("Task %s failed: %s", task_id, error)
        store.fail_task(task_id, error)
    finally:
        on_update(task_id)
        shutil.rmtree(upload_dir, ignore_errors=True)


async def worker_loop(on_update: Callable[[str], None] = lambda _: None):
    logger.info("OCR worker started (max_concurrent=%d)", MAX_CONCURRENT)
    sem = asyncio.Semaphore(MAX_CONCURRENT)

    while True:
        task = await asyncio.to_thread(store.claim_next_pending)
        if task:
            async with sem:
                await _process_task(task, on_update)
        else:
            await asyncio.sleep(1)
