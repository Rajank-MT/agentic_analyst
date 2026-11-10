import os
import psycopg
from dotenv import load_dotenv

load_dotenv()

try:
    conn = psycopg.connect(
        host=os.getenv("POSTGRES_HOST"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        sslmode="require",
        connect_timeout=10,
    )

    print("Azure PostgreSQL connection successful!")

    with conn.cursor() as cur:
        cur.execute("SELECT current_database(), version();")
        print(cur.fetchone())

    conn.close()

except Exception as exc:
    print(f"Connection failed: {type(exc).__name__}: {exc}")