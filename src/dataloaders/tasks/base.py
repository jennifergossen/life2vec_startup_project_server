# src/dataloaders/tasks/base.py
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
    return torch.utils.data.default_collate(dicts)  # type: ignore


_TaskT = TypeVar("_TaskT", bound="Task")


def preprocessor(task: _TaskT, x: StartupDocument, is_train: bool) -> "EncodedDocument[_TaskT]":
    x = task.augment_document(x, is_train=is_train)
    x = task.clip_document(x)
    return task.encode_document(x)


@dataclass
class Task:
    """
    Base class for processing :class:`src.data.types.StartupDocument` objects
    for various ML tasks. Includes default implementations for clipping and augmenting
    documents.
    In order to implement a new task, we have to implement the :meth:`encode_sequence`,
    which defines how the documents should be encoded for a specific task, ie. what
    input the forward method of the model expects for the task in question.

    Also defines a default implementation for pulling :class:`StartupDocument` objects
    out of the sentence data provided by the :class:`src.data.Corpus`. Task
    implementations can extend this method and save task-specific information in the
    :attr:`task_info` field

    :param name: The name of the task.
    :param max_length: The maximum length of the encoded documents
    :param no_sep: If True, don't include the [SEP] token between sentences.

    :param p_timecut:
    :param p_resample:
    :param p_abspos_noise:
    :param p_hide_background:
    :param p_shuffle_sentences:

    :param shuffle_within_sentences:


    .. todo::
        Confirm this is what no_sep is actually for since this has always been False as
        far as I (Søren) know

    """

    # General
    name: str
    max_length: int
    # It True, then don't include the [SEP] token between sentences.
    no_sep: bool = False

    # Augmentation
    p_sequence_timecut: float = 0.0
    p_sequence_resample: float = 0.0
    p_sequence_abspos_noise: float = 0.0
    p_sequence_hide_background: float = 0.0
    p_sentence_drop_tokens: float = 0.0

    shuffle_within_sentences: bool = True

    # Task specific
    ...

    def register(self, datamodule: "L2VDataModule") -> None:
        self.datamodule = datamodule

    def get_preprocessor(
        self: _TaskT, is_train: bool
    ) -> Callable[[StartupDocument], "EncodedDocument[_TaskT]"]:

        return partial(preprocessor, self, is_train=is_train)

    def augment_document(
        self, document: StartupDocument, is_train: bool
    ) -> StartupDocument:

        if self.shuffle_within_sentences:
            # TODO: Maybe we should only do this for training?
            document.sentences = [
                random.sample(x, k=len(x)) for x in document.sentences
            ]

        # Cut document before 1st January 2016
        document = align_document(document)

        if is_train:

            # AUGMENTATION WITH NOISE
            p = np.random.uniform(low=0.0, high=1.0, size=[5])

            # Should be in the exact order
            # 1. TIMECUT (returns cut document)
            if p[0] < self.p_sequence_timecut:
                document = make_timecut(document)  # random timecut
            # 2. RESAMPLE DOCUMENT
            if p[1] < self.p_sequence_resample:
                document = resample_document(document)
            # 2. ADD NOISE TO ABSPOS
            if p[2] < self.p_sequence_abspos_noise:
                document = add_noise2time(document)
            # 3. HIDE BACKGROUND
            if p[3] < self.p_sequence_hide_background:
                document.background = None
            # 4. DROP TOKENS FROM THE SEQUENCE
            if p[4] < self.p_sentence_drop_tokens:
                document = drop_tokens(document, p=self.p_sentence_drop_tokens)

        # 5. ADD SEGMENT
        from itertools import cycle, islice

        segment_pattern = [2, 3, 1]  # Background is segment always 1
        document.segment = list(
            islice(cycle(segment_pattern), len(document.sentences)))

        return document

    def clip_document(self, document: StartupDocument) -> StartupDocument:

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

    def get_document(self, person_sentences: pd.DataFrame) -> StartupDocument:

        person_id = person_sentences.name
        sentences = [x.split(" ") for x in person_sentences.SENTENCE]
        abspos = (person_sentences.RECORD_DATE + 1).tolist()
        age = person_sentences.AGE.tolist()

        after_threshold = person_sentences.AFTER_THRESHOLD
        try:
            timecut_pos = next(i for i, k in enumerate(after_threshold) if k)
        except StopIteration:
            timecut_pos = len(after_threshold)

        birthday = person_sentences.BIRTHDAY.iloc[0]
        sex = person_sentences.SEX.iloc[0]

        background = StartupBackground(
            gender=sex,
            birth_month=birthday.month,
            birth_year=birthday.year,
        )

        return StartupDocument(
            person_id=person_id,
            sentences=sentences,
            abspos=abspos,
            age=age,
            timecut_pos=timecut_pos,
            background=background,
        )
