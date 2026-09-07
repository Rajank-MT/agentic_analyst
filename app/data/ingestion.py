from pathlib import Path

import pandas as pd


SUPPORTED_EXTENSIONS = {
    ".csv",
    ".xlsx",
    ".xls",
}


class DataIngestionError(Exception):
    """Raised when a dataset cannot be ingested."""


def load_dataset(file_path: str | Path) -> pd.DataFrame:
    """
    Load a supported dataset into a pandas DataFrame.

    Supported formats:
        - CSV
        - XLSX
        - XLS
    """

    path = Path(file_path)

    if not path.exists():
        raise DataIngestionError(
            f"Dataset does not exist: {path}"
        )

    if not path.is_file():
        raise DataIngestionError(
            f"Dataset path is not a file: {path}"
        )

    extension = path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise DataIngestionError(
            f"Unsupported file type '{extension}'. "
            f"Supported types: {supported}"
        )

    try:
        if extension == ".csv":
            dataframe = pd.read_csv(path)

        elif extension in {".xlsx", ".xls"}:
            dataframe = pd.read_excel(path)

        else:
            # This should never be reached because of the extension check.
            raise DataIngestionError(
                f"No loader configured for: {extension}"
            )

    except Exception as exc:
        raise DataIngestionError(
            f"Failed to load dataset '{path}': {exc}"
        ) from exc

    if dataframe.empty:
        raise DataIngestionError(
            f"Dataset '{path}' contains no rows."
        )

    return dataframe