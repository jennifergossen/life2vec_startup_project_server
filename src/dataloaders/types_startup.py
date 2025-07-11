# src/dataloaders/types_startup.py
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, List, NewType, Optional, TypeVar

from .types import PersonDocument, Background, EncodedDocument

if TYPE_CHECKING:
    from src.dataloaders.tasks.base import Task

_TaskT = TypeVar("_TaskT", bound="Task")


@dataclass
class StartupBackground:
    """Defines the background information about a startup"""
    
    # Basic location info
    country: str
    state_code: Optional[str] = None
    region: Optional[str] = None
    city: Optional[str] = None
    
    # Founding info
    founding_year: int = 2000
    founding_month: int = 1
    
    # Business info
    roles: Optional[str] = None
    short_description: Optional[str] = None
    category_list: Optional[str] = None
    category_groups_list: Optional[str] = None
    status: Optional[str] = None
    
    # Funding info (static totals/bins)
    num_funding_rounds_binned: Optional[str] = None
    total_funding_usd_binned: Optional[str] = None
    last_funding_on: Optional[str] = None
    
    # Other static info
    closed_on: Optional[str] = None
    employee_count: Optional[str] = None
    num_exits_binned: Optional[str] = None
    company_age_years: Optional[int] = None

    @staticmethod
    def get_sentence(x: Optional["StartupBackground"]) -> List[str]:
        """Return sequence of tokens corresponding to this startup. Similar to Background.get_sentence
        but adapted for startup-specific information.
        """
        if x is None:
            return 8 * ["[UNK]"]  # More fields than human background
        else:
            return [
                x.country if x.country else "[UNK]",
                str(x.founding_year),
                str(x.founding_month),
                x.status if x.status else "[UNK]",
                x.category_list if x.category_list else "[UNK]",
                x.total_funding_usd_binned if x.total_funding_usd_binned else "[UNK]",
                x.num_funding_rounds_binned if x.num_funding_rounds_binned else "[UNK]",
                x.region if x.region else "[UNK]",
            ]


@dataclass
class StartupDocument:
    """Dataclass for defining the complete startup document in a structured fashion"""

    startup_id: str  # Changed from person_id to startup_id
    sentences: List[List[str]]
    abspos: List[int]
    age: List[float]  # Age in years since founding
    segment: Optional[List[int]] = None
    background: Optional[StartupBackground] = None
    timecut_pos: Optional[int] = None
    shuffled: bool = False
