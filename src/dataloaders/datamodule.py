import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import dask
import dask.dataframe as dd
import pandas as pd
import numpy as np
import pytorch_lightning as pl
import torch
from pandas.tseries.offsets import MonthEnd
from torch.utils.data import DataLoader, Dataset

from .tasks.base import Task, collate_encoded_documents
from .dataset import DocumentDataset, ShardedDocumentDataset
from .decorators import save_parquet, save_pickle
from .ops import concat_columns_dask, concat_sorted
from .populations.base import Population
from .serialize import DATA_ROOT, ValidationError, _jsonify
from .sources.base import Field, TokenSource
from .vocabulary import Vocabulary

log = logging.getLogger(__name__)


def compute_age(date: pd.Series, birthday: pd.Series) -> pd.Series:
    age = date.dt.year - birthday.year
    return age


# Not quite true, but it is not trivial to do recursive types i think
JSONSerializable = Any


@dataclass
class Corpus:
    """
    Provides a corpus for the specified population with tokens for the specified
    sources. Splits the data into training, validation and testing partition according
    to the population splits.

    :param sources: List of token sources from which to generate sentences
    :param population: Cohort to generate sentences for.
    :param reference_date: the day from which we can calculate the ABSOLUTE POSITION
    :param threshold: the day at which we want to stop or cut the sequence.
    """

    name: str
    sources: List[TokenSource]
    population: Population
    reference_date: str = "2020-01-01"
    threshold: str = "2026-01-01"

    def __post_init__(self) -> None:
        self._reference_date = pd.to_datetime(self.reference_date)
        self._threshold = pd.to_datetime(self.threshold)

    @save_parquet(
        DATA_ROOT / "processed/corpus/{self.name}/sentences/{split}",
        on_validation_error="recompute",
        verify_index=False,
    )
    def combined_sentences(self, split: str) -> dd.DataFrame:
        """Combines the sentences from each source."""
        population: pd.DataFrame = self.population.population()
        data_split = getattr(self.population.data_split(), split)
        sentences_parts = [self.sentences(s) for s in self.sources]
        combined_sentences = concat_sorted(
            [sp.loc[lambda x: x.index.isin(data_split)]
             for sp in sentences_parts],
            columns=["RECORD_DATE"],
        ).join(population)

        # Handle birthday if it exists
        if "BIRTHDAY" in combined_sentences.columns:
            combined_sentences["BIRTHDAY"] = combined_sentences["BIRTHDAY"].apply(
                lambda x: pd.to_datetime(x, errors='coerce'), meta=('BIRTHDAY', 'datetime64[ns]'))
            combined_sentences["AGE"] = combined_sentences.apply(
                lambda x: x.RECORD_DATE.year - x.BIRTHDAY.year, axis=1, meta=('AGE', 'int64'))
        else:
            # For startups, we'll calculate age differently
            combined_sentences["AGE"] = 0

        combined_sentences["AFTER_THRESHOLD"] = (
            combined_sentences.RECORD_DATE >= self._threshold
        )

        # Date as days from reference date
        combined_sentences["RECORD_DATE"] = (
            combined_sentences.RECORD_DATE - self._reference_date
        ).dt.days.astype(int)

        # DASK SPECIFIC
        combined_sentences = combined_sentences.reset_index().set_index(
            "USER_ID", sorted=True, npartitions="auto")

        assert isinstance(combined_sentences, dd.DataFrame)
        return combined_sentences

    def sentences(self, source: TokenSource) -> dd.DataFrame:
        """Returns the sentences from source."""
        tokenized = self.tokenized_and_transformed(source)
        field_labels = source.field_labels()

        import pandas.api.types as ptypes

        for field in field_labels:
            is_string = ptypes.is_string_dtype(tokenized[field].dtype)
            is_known_cat = (
                ptypes.is_categorical_dtype(tokenized[field].dtype)
                and tokenized[field].cat.known
            )
            assert is_string or is_known_cat

        cols = ["RECORD_DATE", "SENTENCE"]
        if "AGE" in tokenized.columns:
            cols.append("AGE")

        sentences = tokenized.astype({x: "string" for x in field_labels}).assign(
            SENTENCE=concat_columns_dask(tokenized, columns=list(field_labels))
        )[cols]

        assert isinstance(sentences, dd.DataFrame)
        return sentences

    @save_parquet(
        DATA_ROOT / "interim/corpus/{self.name}/tokenized_and_transformed/{source.name}",
        on_validation_error="recompute",
        verify_index=False,
    )
    def tokenized_and_transformed(self, source: TokenSource) -> dd.DataFrame:
        """Returns the tokenized data for source with transformations applied."""
        fields_to_transform = self.fitted_fields(source)
        tokenized = source.tokenized()
        for field in fields_to_transform:
            tokenized[field.field_label] = field.transform(
                tokenized[field.field_label])
        assert isinstance(tokenized, dd.DataFrame)
        return tokenized

    @save_pickle(
        DATA_ROOT / "interim/corpus/{self.name}/fitted_fields/{source.name}",
        on_validation_error="recompute",
    )
    def fitted_fields(self, source: TokenSource) -> List[Field]:
        """Fits any Field using the fit method on training data."""
        ids = self.population.data_split().train
        tokenized = source.tokenized()
        fields = source.fields
        fields_to_fit = [field for field in fields if isinstance(field, Field)]
        for field in fields_to_fit:
            field.fit(tokenized.loc[lambda x: x.index.isin(ids)])
        return fields_to_fit

    def prepare(self) -> None:
        """Prepares each dataset split"""
        self.combined_sentences("train")
        self.combined_sentences("val")
        self.combined_sentences("test")


@dataclass
class L2VDataModule(pl.LightningDataModule):
    """
    Main life2vec data processing pipeline.
    """

    # Data components
    corpus: Corpus
    vocabulary: Vocabulary
    task: Task

    # Data loading params
    batch_size: int = 8
    num_workers: int = 2
    persistent_workers: bool = False
    pin_memory: bool = False
    subset: bool = False
    subset_id: int = 0

    def __post_init__(self) -> None:
        super().__init__()
        self.task.register(self)
        self.processor = None

    @property
    def dataset_root(self) -> Path:
        """Return the dataset root according to the corpus and task names"""
        return DATA_ROOT / "processed" / "datasets" / self.corpus.name / self.task.name

    def prepare(self) -> None:
        """Calls prepare_data to prepare the data."""
        self.prepare_data()
        self.setup()

    def _arguments(self) -> Dict[str, Any]:
        """Supply the arguments of the corpus and task for validation."""
        return {
            "corpus": _jsonify(self.corpus),
            "vocabulary": _jsonify(self.vocabulary),
            "task": _
