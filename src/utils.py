# src/utils.py
from dataclasses import is_dataclass
from collections.abc import Mapping, Iterable
import json
import warnings
import importlib
from typing import Any


def silence_warnings() -> None:
    ignore_patterns = [
        "The dataloader, [^,]*, does not have many workers",
        "`LightningModule.configure_optimizers` returned `None`, this fit will run with no optimizer",
        "GPU available but not used.",
        "torch.qr is deprecated in favor of torch.linalg.qr and will be removed in a future PyTorch release."
    ]

    for ptn in ignore_patterns:
        warnings.filterwarnings("ignore", message=ptn, category=UserWarning)


class SilenceWarnings:
    """Callback class - only import pytorch_lightning when actually needed"""
    
    def __init__(self):
        # Import only when needed to avoid circular imports
        from pytorch_lightning import Callback
        self._callback_base = Callback
    
    def on_init_start(self, trainer) -> None:
        silence_warnings()


def import_from(module: str, name: str) -> Any:
    module_ = importlib.import_module(module)
    return getattr(module_, name)


def stringify(obj):
    if isinstance(obj, str):
        return f"'{obj}'"

    if not is_dataclass(obj) and not isinstance(obj, (Mapping, Iterable)):
        return str(obj)

    # dicts, lists, and tuples will re-assign this
    start, end = f'{type(obj).__name__}(', ')'

    if isinstance(obj, dict):
        start, end = '{}'

    elif isinstance(obj, list):
        start, end = '[]'

    elif isinstance(obj, tuple):
        start = '('

    return f'{start}...{end}'
