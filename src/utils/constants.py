"""Canonical constants. Defined ONCE here and imported everywhere (CLAUDE.md §3.6)."""

from __future__ import annotations

# Canonical label order — submission column order, OOF column order, tensor index order.
# Never retype this list anywhere else.
LABELS: list[str] = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]

N_LABELS: int = len(LABELS)

LABEL_TO_IDX: dict[str, int] = {name: i for i, name in enumerate(LABELS)}

# Pairs swapped by a horizontal flip. Consumed by augment.py:laterality_flip (Spec 03) and by
# the inference-time flip TTA (Spec 05). Anything that mirrors a knee image must use this.
LATERALITY_SWAP_PAIRS: tuple[tuple[str, str], ...] = (
    ("Medial Meniscus", "Lateral Meniscus"),
    ("Medial OA", "Lateral OA"),
)

# Same, as tensor indices — the form the flip actually applies.
LATERALITY_SWAP_IDX: tuple[tuple[int, int], ...] = tuple(
    (LABEL_TO_IDX[a], LABEL_TO_IDX[b]) for a, b in LATERALITY_SWAP_PAIRS
)

# Exact submission header (Spec 05 §Constraints). Tested for byte-equality.
SUBMISSION_HEADER: str = ",".join(["StudyInstanceUID", *LABELS])

# Column names in train_series.csv.
PLANES: tuple[str, ...] = ("Sagittal", "Coronal", "Axial")
PLANE_TO_ID: dict[str, int] = {p: i for i, p in enumerate(PLANES)}

__all__ = [
    "LABELS",
    "N_LABELS",
    "LABEL_TO_IDX",
    "LATERALITY_SWAP_PAIRS",
    "LATERALITY_SWAP_IDX",
    "SUBMISSION_HEADER",
    "PLANES",
    "PLANE_TO_ID",
]
