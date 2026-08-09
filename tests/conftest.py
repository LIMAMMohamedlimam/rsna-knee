"""Synthetic dataset builders.

The real competition data is 570 GB and unavailable in CI, so every test in this repo runs
against a synthetic stand-in that reproduces the structural facts from CLAUDE.md §4:
labels present only for a subset (NaN elsewhere, never 0), multilingual reports, template
duplicates, patients with repeat studies, and multiple scanner fingerprints.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.utils.constants import LABELS

# Deliberately spans the strict and the loose branch of the fold-balance test:
# with 450 labeled studies, rates <= 0.10 land under the 50-positive threshold.
PREVALENCES = [0.30, 0.22, 0.35, 0.16, 0.26, 0.09, 0.21, 0.33, 0.13, 0.19, 0.15, 0.05]

REPORT_TEMPLATES = {
    "en": (
        "MRI LEFT KNEE. TECHNIQUE: Multiplanar multisequence imaging. "
        "FINDINGS: The anterior cruciate ligament is {acl}. The medial meniscus shows {mm}. "
        "Moderate joint effusion is {eff}. IMPRESSION: {imp}."
    ),
    "fr": (
        "IRM DU GENOU DROIT. TECHNIQUE: Sequences multiplanaires. "
        "RESULTATS: Le ligament croise anterieur est {acl}. Le menisque interne presente {mm}. "
        "Un epanchement articulaire est {eff}. CONCLUSION: {imp}."
    ),
    "es": (
        "RESONANCIA DE RODILLA. TECNICA: Secuencias multiplanares. "
        "HALLAZGOS: El ligamento cruzado anterior esta {acl}. El menisco medial muestra {mm}. "
        "Derrame articular {eff}. CONCLUSION: {imp}."
    ),
}
FILLERS = {
    "en": {"acl": ["intact", "completely torn"], "mm": ["a posterior horn tear", "normal signal"],
           "eff": ["present", "absent"], "imp": ["Acute anterior cruciate ligament rupture",
                                                 "No acute internal derangement of the knee"]},
    "fr": {"acl": ["intact", "rompu"], "mm": ["une fissure de la corne posterieure", "un signal normal"],
           "eff": ["present", "absent"], "imp": ["Rupture aigue du ligament croise anterieur",
                                                 "Pas de lesion aigue du genou"]},
    "es": {"acl": ["intacto", "roto"], "mm": ["rotura del cuerno posterior", "senal normal"],
           "eff": ["presente", "ausente"], "imp": ["Rotura aguda del ligamento cruzado anterior",
                                                   "Sin lesion aguda de la rodilla"]},
}

PROTOCOLS = [  # (plane, fluid_sensitive, fat_suppression)
    ("Sagittal", 1, 1),
    ("Coronal", 1, 1),
    ("Axial", 1, 1),
    ("Sagittal", 0, 0),
    ("Coronal", 0, 0),
]
SITES = [
    {"Manufacturer": "SIEMENS", "ManufacturerModelName": "MAGNETOM Aera",
     "MagneticFieldStrength": 1.5, "spacing": 0.31},
    {"Manufacturer": "GE MEDICAL SYSTEMS", "ManufacturerModelName": "SIGNA HDxt",
     "MagneticFieldStrength": 3.0, "spacing": 0.27},
    {"Manufacturer": "Philips", "ManufacturerModelName": "Ingenia",
     "MagneticFieldStrength": 1.5, "spacing": 0.35},
]


@dataclass
class SyntheticData:
    train: pd.DataFrame
    series: pd.DataFrame
    study_meta: pd.DataFrame
    raw_dir: Path | None = None

    def write_raw(self, raw_dir: Path) -> Path:
        raw_dir.mkdir(parents=True, exist_ok=True)
        self.train.to_csv(raw_dir / "train.csv", index=False)
        self.series.to_csv(raw_dir / "train_series.csv", index=False)
        self.raw_dir = raw_dir
        return raw_dir


def make_synthetic(
    n_patients: int = 600,
    labeled_fraction: float = 0.75,
    repeat_patient_fraction: float = 0.15,
    seed: int = 0,
) -> SyntheticData:
    rng = np.random.default_rng(seed)

    patients = [f"PAT{i:05d}" for i in range(n_patients)]
    # Labels are drawn per patient (same knee, same pathology) and copied to their studies —
    # this mirrors reality and keeps group-level stratification consistent with study-level rates.
    patient_labels = {
        pid: (rng.random(len(LABELS)) < np.array(PREVALENCES)).astype(float) for pid in patients
    }
    n_labeled = int(round(n_patients * labeled_fraction))
    labeled_patients = set(patients[:n_labeled])

    n_repeat = int(round(n_patients * repeat_patient_fraction))
    repeat_patients = set(rng.choice(patients, size=n_repeat, replace=False).tolist())

    train_rows, series_rows, meta_rows = [], [], []
    study_counter = 0
    # Index-based site/language assignment: Python's str hash is salted per process, which
    # would make the fixture irreproducible across runs.
    for p_idx, pid in enumerate(patients):
        site = SITES[p_idx % len(SITES)]
        lang = ["en", "fr", "es"][(p_idx // len(SITES)) % 3]
        for _ in range(2 if pid in repeat_patients else 1):
            # No leading zeros: a UID component with them is invalid VR UI and pydicom warns.
            study_uid = f"1.2.826.0.1.{1000000 + study_counter}"
            study_counter += 1
            y = patient_labels[pid]

            row: dict = {
                "StudyInstanceUID": study_uid,
                "PatientSex": "M" if p_idx % 2 else "F",
                "Report": _make_report(lang, y, rng),
            }
            for i, label in enumerate(LABELS):
                row[label] = y[i] if pid in labeled_patients else np.nan
            train_rows.append(row)

            n_protocols = int(rng.integers(3, len(PROTOCOLS) + 1))
            for k in range(n_protocols):
                plane, fluid, fatsat = PROTOCOLS[k]
                series_rows.append({
                    "StudyInstanceUID": study_uid,
                    "SeriesInstanceUID": f"{study_uid}.{k}",
                    "Fluid_Sensitive": fluid,
                    "Fat_Suppression": fatsat,
                    "Anatomical_Plane": plane,
                })

            meta_rows.append({
                "StudyInstanceUID": study_uid,
                "PatientID": pid,
                "language": lang,
                "site_cluster": SITES.index(site),
                "site_cluster_source": "dicom",
            })

    return SyntheticData(
        train=pd.DataFrame(train_rows),
        series=pd.DataFrame(series_rows),
        study_meta=pd.DataFrame(meta_rows),
    )


def _make_report(lang: str, y: np.ndarray, rng: np.random.Generator) -> str:
    """Template report whose text agrees with the ACL / medial-meniscus / effusion labels.

    ~8% are emitted as a bare template so the near-duplicate detector has something to find.
    """
    fillers = FILLERS[lang]
    if rng.random() < 0.08:
        return REPORT_TEMPLATES[lang].format(acl=fillers["acl"][0], mm=fillers["mm"][1],
                                             eff=fillers["eff"][1], imp=fillers["imp"][1])
    acl, mm, eff = int(y[0]), int(y[2]), int(y[7])
    return REPORT_TEMPLATES[lang].format(
        acl=fillers["acl"][acl], mm=fillers["mm"][1 - mm],
        eff=fillers["eff"][1 - eff], imp=fillers["imp"][1 - acl],
    )


def write_synthetic_dicom(path: Path, study_uid: str, series_uid: str, instance: int,
                          patient_id: str, site: dict, rows: int = 256, cols: int = 256) -> Path:
    """A minimal but valid MR DICOM file (Explicit VR LE) for reader/EDA tests."""
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    path.parent.mkdir(parents=True, exist_ok=True)
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = pydicom.uid.MRImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.ImplementationVersionName = "PYDICOM_TEST"

    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = pydicom.uid.MRImageStorage
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.PatientID = patient_id
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = series_uid
    ds.InstanceNumber = instance
    ds.Modality = "MR"
    ds.Manufacturer = site["Manufacturer"]
    ds.ManufacturerModelName = site["ManufacturerModelName"]
    ds.MagneticFieldStrength = site["MagneticFieldStrength"]
    ds.PixelSpacing = [site["spacing"], site["spacing"]]
    ds.SliceThickness = 3.0
    ds.ImagePositionPatient = [0.0, 0.0, float(instance) * 3.0]
    ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    ds.Rows, ds.Columns = rows, cols
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelData = np.full((rows, cols), instance * 10, dtype=np.uint16).tobytes()
    ds.save_as(path, enforce_file_format=True)
    return path


def write_synthetic_dicom_tree(data: SyntheticData, raw_dir: Path, n_studies: int = 6,
                               n_slices: int = 4) -> Path:
    """Materialise train_series/<study>/<series>/*.dcm for the first `n_studies` studies."""
    root = raw_dir / "train_series"
    first = data.series.groupby("StudyInstanceUID", sort=True).head(1).head(n_studies)
    meta = data.study_meta.set_index("StudyInstanceUID")
    for study_uid, series_uid in zip(first["StudyInstanceUID"], first["SeriesInstanceUID"], strict=True):
        site = SITES[int(meta.loc[study_uid, "site_cluster"])]
        for i in range(n_slices):
            write_synthetic_dicom(
                root / str(study_uid) / str(series_uid) / f"{i}.dcm",
                study_uid, series_uid, i, str(meta.loc[study_uid, "PatientID"]), site,
            )
    return root


@pytest.fixture(scope="session")
def synthetic() -> SyntheticData:
    return make_synthetic()


@pytest.fixture(scope="session")
def synthetic_raw(tmp_path_factory, synthetic: SyntheticData) -> SyntheticData:
    """Synthetic data materialised on disk as a raw_dir, with a small DICOM tree."""
    raw_dir = tmp_path_factory.mktemp("rsna_raw")
    synthetic.write_raw(raw_dir)
    write_synthetic_dicom_tree(synthetic, raw_dir)
    return synthetic
