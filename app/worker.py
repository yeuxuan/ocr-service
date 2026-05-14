import asyncio
import os
import signal
import shutil
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable

from . import store

logger = logging.getLogger("ocr-worker")

RESULTS_DIR = Path(__file__).parent.parent / "data" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OCR_TIMEOUT_IMAGE = int(os.environ.get("OCR_TIMEOUT_IMAGE", "300"))
OCR_TIMEOUT_PDF = int(os.environ.get("OCR_TIMEOUT_PDF", "1800"))
MAX_CONCURRENT = int(os.environ.get("OCR_MAX_CONCURRENT", "2"))

_PDF_SUFFIXES = {".pdf"}


def _timeout_for(file_name: str) -> int:
    suffix = Path(file_name).suffix.lower()
    return OCR_TIMEOUT_PDF if suffix in _PDF_SUFFIXES else OCR_TIMEOUT_IMAGE


def _error_message(e: Exception) -> str:
    msg = str(e)
    if msg:
        return msg
    return f"{type(e).__name__}: {type(e).__doc__ or 'no details'}"


def _parse_in_process(file_path: str, output_dir: str, timeout: int) -> dict:
    """在独立 worker 进程中执行，SIGALRM 实现进程内超时。"""
    def _on_alarm(signum, frame):
        raise TimeoutError(f"OCR timeout after {timeout}s")

    signal.signal(signal.SIGALRM, _on_alarm)
    signal.alarm(timeout)
    try:
        from glmocr import parse

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        result = parse(file_path)
        result.save(output_dir=output_dir)
        return {
            "markdown": result.markdown_result or "",
            "json": result.json_result,
        }
    finally:
        signal.alarm(0)


async def _process_task(
    task: dict,
    on_update: Callable[[str], None],
    pool: ProcessPoolExecutor,
    sem: asyncio.Semaphore,
):
    task_id = task["task_id"]
    file_path = task["file_path"]
    file_name = task["file_name"]
    upload_dir = Path(file_path).parent
    output_dir = RESULTS_DIR / task_id
    timeout = _timeout_for(file_name)

    logger.info("Processing task %s: %s (timeout=%ds)", task_id, file_name, timeout)
    on_update(task_id)

    try:
        loop = asyncio.get_running_loop()
        result_data = await asyncio.wait_for(
            loop.run_in_executor(
                pool, _parse_in_process, file_path, str(output_dir), timeout
            ),
            timeout=timeout + 60,
        )
        result_data["file_name"] = file_name

        store.complete_task(task_id, result_data)
        logger.info("Task %s completed", task_id)
    except (asyncio.TimeoutError, TimeoutError) as e:
        error = f"OCR timeout after {timeout}s for {file_name}"
        logger.error("Task %s timed out: %s", task_id, error)
        store.fail_task(task_id, error)
    except Exception as e:
        error = _error_message(e)
        logger.exception("Task %s failed: %s", task_id, error)
        store.fail_task(task_id, error)
    finally:
        sem.release()
        on_update(task_id)
        shutil.rmtree(upload_dir, ignore_errors=True)


async def worker_loop(on_update: Callable[[str], None] = lambda _: None):
    logger.info("OCR worker started (max_concurrent=%d, process pool)", MAX_CONCURRENT)
    pool = ProcessPoolExecutor(max_workers=MAX_CONCURRENT)
    sem = asyncio.Semaphore(MAX_CONCURRENT)

    try:
        while True:
            await sem.acquire()
            task = await asyncio.to_thread(store.claim_next_pending)
            if task:
                asyncio.create_task(_process_task(task, on_update, pool, sem))
            else:
                sem.release()
                await asyncio.sleep(1)
    finally:
        for proc in pool._processes.values():
            proc.terminate()
        pool.shutdown(wait=False)
