# src/dataloaders/tasks/startup_tasks.py
import logging
import numpy as np
from dataclasses import dataclass
from functools import cached_property
from itertools import chain
from typing import List, Tuple, TypeVar, cast

from ..types import StartupBackground, StartupDocument
from .pretrain import MLM as BaseMLM, MLMEncodedDocument

log = logging.getLogger(__name__)

T = TypeVar("T")

@dataclass
class StartupMLM(BaseMLM):
    """MLM task adapted for startup data"""
    
    def encode_document(self, document: StartupDocument) -> "StartupMLMEncodedDocument":
        """Create appropriate encoding for startup documents"""
        # Get startup background sentence
        prefix_sentence = (
            ["[CLS]"] +
            StartupBackground.get_sentence(document.background) + ["[SEP]"]
        )

        # Apply SOP task
        document, targ_sop = self.sop_task(document)
        
        # Create full sentences
        sentences = [prefix_sentence] + [s + ["[SEP]"] for s in document.sentences]
        sentence_lengths = [len(x) for x in sentences]

        # Expand positions to token level
        def expand(x: List[T]) -> List[T]:
            assert len(x) == len(sentence_lengths)
            return list(
                chain.from_iterable(
                    length * [i] for length, i in zip(sentence_lengths, x)
                )
            )

        # Expand positions
        abspos_expanded = expand([0] + document.abspos)
        age_expanded = expand([0.0] + document.age)
        assert document.segment is not None
        segment_expanded = expand([1] + document.segment)

        # Flatten everything to sequences
        flat_sentences = np.concatenate(sentences)

        # Convert tokens to indices
        token2index = self.datamodule.vocabulary.token2index
        unk_id = token2index["[UNK]"]
        token_ids = np.array([token2index.get(x, unk_id) for x in flat_sentences])
        
        # Apply MLM mask
        masked_sentences, masked_indx, masked_tokens = self.mlm_mask(token_ids.copy())

        # Create final encoding
        length = len(token_ids)
        
        # Initialize arrays
        input_ids = np.zeros((4, self.max_length))
        input_ids[0, :length] = masked_sentences
        input_ids[1, :length] = abspos_expanded
        input_ids[2, :length] = age_expanded
        input_ids[3, :length] = segment_expanded

        padding_mask = np.repeat(False, self.max_length)
        padding_mask[:length] = True

        original_sequence = np.zeros(self.max_length)
        original_sequence[:length] = token_ids

        # Use startup_id instead of person_id
        sequence_id = np.array(document.startup_id)

        return StartupMLMEncodedDocument(
            sequence_id=sequence_id,
            input_ids=input_ids,
            padding_mask=padding_mask,
            target_tokens=masked_tokens,
            target_pos=masked_indx,
            target_sop=targ_sop,
            original_sequence=original_sequence,
        )

    def sop_task(self, document: StartupDocument):
        """Same as the original but with StartupDocument"""
        p = np.random.rand(1)
        if p < 0.05:
            document.sentences.reverse()
            targ_sop = 1
        elif p > 0.95:
            from random import shuffle
            shuffle(document.sentences)
            targ_sop = 2
        else:
            targ_sop = 0

        return document, targ_sop

@dataclass
class StartupMLMEncodedDocument(MLMEncodedDocument):
    """Encoded document for startup MLM task"""
    pass
