"""Contract tests for the canonical constants (CLAUDE.md §3.6, Spec 01 Task 1.1)."""

from __future__ import annotations

from src.utils.constants import (
    LABEL_TO_IDX,
    LABELS,
    LATERALITY_SWAP_IDX,
    LATERALITY_SWAP_PAIRS,
    N_LABELS,
    SUBMISSION_HEADER,
)


def test_twelve_labels_in_canonical_order():
    assert N_LABELS == 12 == len(LABELS)
    assert LABELS == [
        "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
        "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
    ]
    assert len(set(LABELS)) == N_LABELS


def test_submission_header_is_exact():
    # Spec 05 requires this byte string verbatim.
    assert SUBMISSION_HEADER == (
        "StudyInstanceUID,ACL,MCL,Medial Meniscus,Lateral Meniscus,Medial OA,Lateral OA,"
        "PF OA,Effusion,Synovitis,Baker's,Contusion,Fracture"
    )


def test_laterality_swap_pairs_are_consistent():
    assert LATERALITY_SWAP_PAIRS == (
        ("Medial Meniscus", "Lateral Meniscus"),
        ("Medial OA", "Lateral OA"),
    )
    for (a, b), (ia, ib) in zip(LATERALITY_SWAP_PAIRS, LATERALITY_SWAP_IDX, strict=True):
        assert LABEL_TO_IDX[a] == ia and LABEL_TO_IDX[b] == ib

    # The two swapped indices must be distinct or a flip would be a no-op.
    assert all(ia != ib for ia, ib in LATERALITY_SWAP_IDX)


def test_config_labels_match_constants():
    from src.utils.config import load_config

    assert list(load_config().labels) == LABELS
