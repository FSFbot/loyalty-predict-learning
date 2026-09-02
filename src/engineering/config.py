from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR/ "raw"

@dataclass(frozen=True)
class KaggleDataset:
    slug: str
    destination: Path

KAGGLE_DATASETS = (
    KaggleDataset(
        slug="teocalvo/teomewhy-loyalty-system",
        destination=RAW_DATA_DIR / "loyalty",
    ),
    KaggleDataset(
        slug="teocalvo/teomewhy-education-platform",
        destination=RAW_DATA_DIR / "education",
    ),
)   