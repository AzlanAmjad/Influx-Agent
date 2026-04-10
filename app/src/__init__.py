"""Application package initialization."""

import warnings

# langchain_core attempts to import pydantic v1 compatibility shims which were
# removed in Python 3.14+. The library emits this warning and continues
# correctly; suppress only this known warning pattern globally for the app.
warnings.filterwarnings(
    "ignore",
    message="Core Pydantic V1 functionality",
    category=UserWarning,
    module=r"langchain_core.*",
)
