"""Application package initialization."""

import logging
import warnings

from app.src.core.config import settings

# ── logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=settings.log_level.upper(),
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)

# Quieten noisy third-party loggers even when our own level is DEBUG.
for _noisy in ("httpcore", "httpx", "urllib3", "influxdb"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

# langchain_core attempts to import pydantic v1 compatibility shims which were
# removed in Python 3.14+. The library emits this warning and continues
# correctly; suppress only this known warning pattern globally for the app.
warnings.filterwarnings(
    "ignore",
    message="Core Pydantic V1 functionality",
    category=UserWarning,
    module=r"langchain_core.*",
)
