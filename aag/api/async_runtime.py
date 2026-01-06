import asyncio
import logging
import threading
from typing import Optional

from aag.api.services.chat_service import ChatService
from aag.api.services.engine_service import EngineService

logger = logging.getLogger(__name__)

# Global state for the shared asyncio runtime
background_loop: Optional[asyncio.AbstractEventLoop] = None
chat_service: Optional[ChatService] = None

_loop_thread: Optional[threading.Thread] = None
_initialized = threading.Event()


def _run_loop(loop: asyncio.AbstractEventLoop):
    """Run the asyncio loop forever inside a dedicated thread."""
    asyncio.set_event_loop(loop)
    loop.run_forever()


async def _initialize_components():
    """
    Initialize shared components on the background loop.

    - Create ChatService (and thus EngineService/AAGEngine/Scheduler)
    - Warm up ComputingEngine connections once
    """
    global chat_service

    chat_service = ChatService()
    try:
        engine = chat_service.engine_service.get_engine()
        scheduler = getattr(engine, "scheduler", None)
        computing_engine = getattr(scheduler, "computing_engine", None)

        if computing_engine and not getattr(computing_engine, "_initialized", False):
            await computing_engine.initialize()
    except Exception:
        logger.exception("Failed to initialize async runtime components")
        raise

    _initialized.set()
    logger.info("Async runtime initialized")


async def _shutdown_components():
    """Tear down shared components on the background loop."""
    try:
        engine_service = EngineService.get_instance()
        if engine_service.is_initialized():
            await engine_service.shutdown()
    except Exception:
        logger.exception("Error while shutting down async runtime components")


def start_async_runtime():
    """Start the background event loop and initialize shared services."""
    global background_loop, _loop_thread

    if background_loop and _loop_thread and _loop_thread.is_alive():
        return background_loop

    background_loop = asyncio.new_event_loop()
    _loop_thread = threading.Thread(
        target=_run_loop,
        args=(background_loop,),
        name="aag-async-runtime",
        daemon=True,
    )
    _loop_thread.start()

    fut = asyncio.run_coroutine_threadsafe(_initialize_components(), background_loop)
    fut.result()  # Propagate initialization failures immediately
    return background_loop


def stop_async_runtime():
    """Gracefully stop the background loop and release resources."""
    global background_loop, _loop_thread, chat_service

    if not background_loop:
        return

    try:
        fut = asyncio.run_coroutine_threadsafe(_shutdown_components(), background_loop)
        fut.result(timeout=10)
    except Exception:
        logger.exception("Error occurred during async runtime shutdown")

    if background_loop.is_running():
        background_loop.call_soon_threadsafe(background_loop.stop)

    if _loop_thread:
        _loop_thread.join(timeout=5)

    if not background_loop.is_closed():
        background_loop.close()

    chat_service = None
    _initialized.clear()
    background_loop = None
    _loop_thread = None
    logger.info("Async runtime stopped")


def get_background_loop() -> asyncio.AbstractEventLoop:
    """Return the shared event loop (must call start_async_runtime first)."""
    if not background_loop:
        raise RuntimeError("Async runtime not started")
    return background_loop


def get_chat_service() -> ChatService:
    """Return the shared ChatService instance."""
    if not chat_service or not _initialized.is_set():
        raise RuntimeError("Async runtime not initialized")
    return chat_service
