from pathlib import Path

from app.config.settings import get_settings
from app.data.ingestion import load_dataset
from app.data.profiling import profile_dataset
from app.api.postgres import router as postgres_router

def main() -> None:
    """Run a basic end-to-end data ingestion and profiling test."""

    settings = get_settings()

    print(f"Application: {settings.app_name}")
    print(f"Environment: {settings.environment}")

    dataset_path = Path("datasets/customers.csv")

    dataframe = load_dataset(dataset_path)

    profile = profile_dataset(
        dataframe=dataframe,
        dataset_name=dataset_path.name,
    )

    print("\nDataset Profile")
    print("================")
    print(f"Dataset: {profile.dataset_name}")
    print(f"Rows: {profile.row_count}")
    print(f"Columns: {profile.column_count}")
    print(
        f"Duplicate rows: "
        f"{profile.duplicate_row_count}"
    )

    print("\nColumns")
    print("-------")

    for column in profile.columns:
        print(
            f"{column.name}: "
            f"dtype={column.dtype}, "
            f"type={column.semantic_type.value}, "
            f"nulls={column.null_percentage}%, "
            f"unique={column.unique_count}"
        )


if __name__ == "__main__":
    main()