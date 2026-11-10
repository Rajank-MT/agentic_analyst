from dotenv import load_dotenv
from sqlalchemy import text

from app.sources.postgres import create_postgres_engine

load_dotenv()

engine = create_postgres_engine()

try:
    with engine.begin() as connection:
        connection.execute(
            text("""
                CREATE TABLE IF NOT EXISTS public.sales (
                    id SERIAL PRIMARY KEY,
                    product VARCHAR(100),
                    region VARCHAR(100),
                    revenue NUMERIC(12, 2),
                    quantity INTEGER
                )
            """)
        )

        connection.execute(
            text("""
                INSERT INTO public.sales
                    (product, region, revenue, quantity)
                SELECT
                    'Laptop', 'South', 75000.00, 2
                WHERE NOT EXISTS (
                    SELECT 1 FROM public.sales
                    WHERE product = 'Laptop'
                      AND region = 'South'
                )
            """)
        )

        connection.execute(
            text("""
                INSERT INTO public.sales
                    (product, region, revenue, quantity)
                SELECT
                    'Phone', 'North', 25000.00, 5
                WHERE NOT EXISTS (
                    SELECT 1 FROM public.sales
                    WHERE product = 'Phone'
                      AND region = 'North'
                )
            """)
        )

    print("Sample sales table is ready.")

finally:
    engine.dispose()