from dotenv import load_dotenv
import os

import pandas as pd
from sqlalchemy import MetaData, Table, create_engine, inspect, select
from sqlalchemy.engine import URL

load_dotenv()


def create_postgres_engine():
    """Create a PostgreSQL engine using environment variables."""
    required_vars = ["POSTGRES_HOST", "POSTGRES_DB", "POSTGRES_USER", "POSTGRES_PASSWORD"]
    missing = [key for key in required_vars if not os.getenv(key)]
    if missing:
        raise ValueError(f"Missing PostgreSQL configuration: {', '.join(missing)}")

    connection_url = URL.create(
        drivername="postgresql+psycopg",
        username=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        host=os.environ["POSTGRES_HOST"],
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        database=os.environ["POSTGRES_DB"],
        query={"sslmode": os.getenv("POSTGRES_SSLMODE", "require")},
    )
    return create_engine(
        connection_url,
        pool_pre_ping=True,
        pool_size=1,
        max_overflow=0,
        connect_args={"connect_timeout": 10},
    )


def test_postgres_connection() -> bool:
    engine = create_postgres_engine()
    try:
        with engine.connect() as connection:
            connection.exec_driver_sql("SELECT 1")
        return True
    finally:
        engine.dispose()


def list_postgres_tables(schema: str = "public") -> list[str]:
    engine = create_postgres_engine()
    try:
        inspector = inspect(engine)
        if schema not in inspector.get_schema_names():
            raise ValueError("The requested schema does not exist.")
        return inspector.get_table_names(schema=schema)
    finally:
        engine.dispose()


def preview_postgres_table(table_name: str, schema: str = "public", limit: int = 100) -> pd.DataFrame:
    if not 1 <= limit <= 1000:
        raise ValueError("Preview limit must be between 1 and 1000.")
    return _read_postgres_table(table_name, schema, limit)


def load_postgres_table(table_name: str, schema: str = "public", limit: int = 50000) -> pd.DataFrame:
    if not 1 <= limit <= 50000:
        raise ValueError("Analysis limit must be between 1 and 50000.")
    return _read_postgres_table(table_name, schema, limit)


def load_postgres_tables(
    schema: str = "public",
    row_limit: int = 50000,
    max_tables: int = 50,
) -> dict[str, pd.DataFrame]:
    """Load bounded samples from every table in a schema using one engine."""
    if not 1 <= row_limit <= 50000:
        raise ValueError("row_limit must be between 1 and 50000.")
    if not 1 <= max_tables <= 200:
        raise ValueError("max_tables must be between 1 and 200.")
    if not schema or not schema.strip():
        raise ValueError("Schema name cannot be empty.")

    engine = create_postgres_engine()
    try:
        inspector = inspect(engine)
        if schema not in inspector.get_schema_names():
            raise ValueError("The requested schema does not exist.")
        names = sorted(inspector.get_table_names(schema=schema))
        if len(names) > max_tables:
            raise ValueError(
                f"Schema contains {len(names)} tables; limit is {max_tables}. "
                "Increase max_tables or choose a smaller schema."
            )

        metadata = MetaData()
        result: dict[str, pd.DataFrame] = {}
        with engine.connect() as connection:
            for name in names:
                table = Table(name, metadata, schema=schema, autoload_with=connection)
                result[name] = pd.read_sql(select(table).limit(row_limit), connection)
        return result
    finally:
        engine.dispose()


def _read_postgres_table(table_name: str, schema: str, limit: int) -> pd.DataFrame:
    if not table_name or not table_name.strip():
        raise ValueError("Table name cannot be empty.")
    if not schema or not schema.strip():
        raise ValueError("Schema name cannot be empty.")

    engine = create_postgres_engine()
    try:
        inspector = inspect(engine)
        if schema not in inspector.get_schema_names():
            raise ValueError("The requested schema does not exist.")
        if table_name not in inspector.get_table_names(schema=schema):
            raise ValueError("The requested table does not exist.")

        metadata = MetaData()
        with engine.connect() as connection:
            table = Table(table_name, metadata, schema=schema, autoload_with=connection)
            return pd.read_sql(select(table).limit(limit), connection)
    finally:
        engine.dispose()
