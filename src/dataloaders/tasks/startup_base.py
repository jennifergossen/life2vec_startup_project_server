# src/dataloaders/tasks/startup_base.py
"""
Startup-specific task base that uses uuid instead of USER_ID
"""

import random
from dataclasses import asdict, dataclass
from itertools import accumulate
from typing import TYPE_CHECKING, Callable, Dict, List, TypeVar

import numpy as np
import pandas as pd
import torch
from functools import partial

from src.dataloaders.augment import (
    add_noise2time,
    align_document,
    make_timecut,
    resample_document,
    shuffle_sentences,
    drop_tokens,
)
from src.dataloaders.types import StartupBackground, StartupDocument

if TYPE_CHECKING:
    from src.dataloaders.datamodule import L2VDataModule
    from src.dataloaders.types import EncodedDocument

def collate_encoded_documents(
    batch: List["EncodedDocument"],
) -> Dict[str, torch.Tensor]:
    dicts = [asdict(x) for x in batch]
    return torch.utils.data.default_collate(dicts)

_TaskT = TypeVar("_TaskT", bound="StartupTask")

def startup_preprocessor(task: _TaskT, x: StartupDocument, is_train: bool) -> "EncodedDocument[_TaskT]":
    x = task.augment_document(x, is_train=is_train)
    x = task.clip_document(x)
    return task.encode_document(x)

@dataclass
class StartupTask:
    """
    Base class for processing StartupDocument objects for ML tasks
    Uses uuid instead of USER_ID throughout
    """

    # General
    name: str
    max_length: int
    no_sep: bool = False

    # Augmentation (same as original but for startups)
    p_sequence_timecut: float = 0.0
    p_sequence_resample: float = 0.0
    p_sequence_abspos_noise: float = 0.0
    p_sequence_hide_background: float = 0.0
    p_sentence_drop_tokens: float = 0.0
    shuffle_within_sentences: bool = True

    def register(self, datamodule: "L2VDataModule") -> None:
        self.datamodule = datamodule

    def get_preprocessor(
        self: _TaskT, is_train: bool
    ) -> Callable[[StartupDocument], "EncodedDocument[_TaskT]"]:
        return partial(startup_preprocessor, self, is_train=is_train)

    def augment_document(
        self, document: StartupDocument, is_train: bool
    ) -> StartupDocument:
        """Augment startup documents (similar to person documents)"""

        if self.shuffle_within_sentences:
            document.sentences = [
                random.sample(x, k=len(x)) for x in document.sentences
            ]

        # Cut document before threshold
        document = align_document(document)

        if is_train:
            # AUGMENTATION WITH NOISE
            p = np.random.uniform(low=0.0, high=1.0, size=[5])

            # 1. TIMECUT
            if p[0] < self.p_sequence_timecut:
                document = make_timecut(document)
            # 2. RESAMPLE DOCUMENT
            if p[1] < self.p_sequence_resample:
                document = resample_document(document)
            # 3. ADD NOISE TO ABSPOS
            if p[2] < self.p_sequence_abspos_noise:
                document = add_noise2time(document)
            # 4. HIDE BACKGROUND
            if p[3] < self.p_sequence_hide_background:
                document.background = None
            # 5. DROP TOKENS
            if p[4] < self.p_sentence_drop_tokens:
                document = drop_tokens(document, p=self.p_sentence_drop_tokens)

        # ADD SEGMENT
        from itertools import cycle, islice
        segment_pattern = [2, 3, 1]  # Background is segment 1
        document.segment = list(
            islice(cycle(segment_pattern), len(document.sentences)))

        return document

    def clip_document(self, document: StartupDocument) -> StartupDocument:
        """Clip startup documents to max length"""
        assert document.segment is not None

        sep_size = 0 if self.no_sep else 1
        prefix_length = len(StartupBackground.get_sentence(
            document.background)) + 1 + sep_size
        max_sequence_length = self.max_length - prefix_length

        lengths = [len(s) + sep_size for s in document.sentences]
        clip_idx = None
        for i, x in enumerate(accumulate(reversed(lengths))):
            if x > max_sequence_length:
                clip_idx = i
                break

        if clip_idx is not None:
            document.sentences = document.sentences[-clip_idx:]
            document.abspos = document.abspos[-clip_idx:]
            document.age = document.age[-clip_idx:]
            document.segment = document.segment[-clip_idx:]

        return document

    def encode_document(
        self: _TaskT, document: StartupDocument
    ) -> "EncodedDocument[_TaskT]":
        raise NotImplementedError

    def get_document(self, startup_sentences: pd.DataFrame) -> StartupDocument:
        """Create StartupDocument from sentences - uses uuid instead of USER_ID"""
        
        # Get startup_id (uuid) from index
        startup_id = startup_sentences.name  # This is the index value
        sentences = [x.split(" ") for x in startup_sentences.SENTENCE]
        abspos = (startup_sentences.RECORD_DATE + 1).tolist()
        age = startup_sentences.AGE.tolist()

        after_threshold = startup_sentences.AFTER_THRESHOLD
        try:
            timecut_pos = next(i for i, k in enumerate(after_threshold) if k)
        except StopIteration:
            timecut_pos = len(after_threshold)

        # Get startup background info
        country = startup_sentences.get('country_code', 'US').iloc[0] if 'country_code' in startup_sentences.columns else 'US'
        status = startup_sentences.get('status', 'operating').iloc[0] if 'status' in startup_sentences.columns else 'operating'
        
        # Get founding date
        if 'founded_on' in startup_sentences.columns:
            founded_on = startup_sentences.founded_on.iloc[0]
            if pd.notna(founded_on):
                founding_year = founded_on.year
                founding_month = founded_on.month
            else:
                founding_year = 2000
                founding_month = 1
        else:
            founding_year = 2000
            founding_month = 1

        background = StartupBackground(
            country=country,
            founding_year=founding_year,
            founding_month=founding_month,
            status=status
        )

        return StartupDocument(
            startup_id=str(startup_id),  # Convert uuid to string
            sentences=sentences,
            abspos=abspos,
            age=age,
            timecut_pos=timecut_pos,
            background=background,
        )
