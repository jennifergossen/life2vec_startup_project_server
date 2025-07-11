# src/dataloaders/types.py
"""
Type definitions for both human life2vec and startup life2vec
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, List, NewType, Optional, TypeVar

JSONSerializable = NewType("JSONSerializable", object)

if TYPE_CHECKING:
    from src.dataloaders.tasks.base import Task

_TaskT = TypeVar("_TaskT", bound="Task")

# Original human types (keep for compatibility)
@dataclass
class PersonDocument:
    """Dataclass for defining the complete person document in a structured fashion"""
    person_id: int
    sentences: List[List[str]]
    abspos: List[int]
    age: List[float]
    segment: Optional[List[int]] = None
    background: Optional["Background"] = None
    timecut_pos: Optional[int] = None
    shuffled: bool = False

@dataclass
class Background:
    """Defines the background information about a person"""
    gender: str
    birth_month: int
    birth_year: int

    @staticmethod
    def get_sentence(x: Optional["Background"]) -> List[str]:
        """Return sequence of tokens corresponding to this person."""
        if x is None:
            return 4 * ["[UNK]"]
        else:
            return [
                x.gender,
                str(x.birth_month),
                str(x.birth_year),
            ]

# New startup types
@dataclass
class StartupDocument:
    """Dataclass for defining the complete startup document"""
    startup_id: str  # uuid as string
    sentences: List[List[str]]
    abspos: List[int]
    age: List[float]  # Company age in years
    segment: Optional[List[int]] = None
    background: Optional["StartupBackground"] = None
    timecut_pos: Optional[int] = None
    shuffled: bool = False

@dataclass
class StartupBackground:
    """Defines the background information about a startup"""
    country: str
    founding_year: int
    founding_month: int
    status: str = "operating"

    @staticmethod
    def get_sentence(x: Optional["StartupBackground"]) -> List[str]:
        """Return sequence of tokens corresponding to this startup."""
        if x is None:
            return 4 * ["[UNK]"]
        else:
            return [
                x.country,
                str(x.founding_month),
                str(x.founding_year),
                x.status
            ]

class EncodedDocument(Generic[_TaskT]):
    """Generic class for encoded documents."""
    pass
