from kaggle import api

from .config import KAGGLE_DATASETS, KaggleDataset

def download_dataset(dataset: KaggleDataset) -> None:
    dataset.destination.mkdir(parents=True, exist_ok=True)
    print(f"Baixando {dataset.slug}")
    print(f"Destino: {dataset.destination}")


    api.dataset_download_files(
        dataset= dataset.slug,
        path=str(dataset.destination),
        unzip=True,
        quiet= False,

    )

def main() -> None:
    for dataset in KAGGLE_DATASETS:
        download_dataset(dataset)

if __name__ == "__main__":
    main()

    