import logging
import sys
from typing import Any, Dict

try:
    import structlog

    HAS_STRUCTLOG = True
except ImportError:
    HAS_STRUCTLOG = False


def setup_logging(level: str = "INFO", structured: bool = True) -> Any:
    """Configures structured or standard logging for satsim."""
    log_level = getattr(logging, level.upper(), logging.INFO)

    if structured and HAS_STRUCTLOG:
        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.processors.add_log_level,
                structlog.processors.StackInfoRenderer(),
                structlog.dev.set_exc_info,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.JSONRenderer(),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(log_level),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=True,
        )
        return structlog.get_logger("satsim")
    else:
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
        )
        return logging.getLogger("satsim")


def get_logger(name: str = "satsim") -> Any:
    if HAS_STRUCTLOG:
        return structlog.get_logger(name)
    return logging.getLogger(name)
