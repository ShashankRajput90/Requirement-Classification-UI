import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

conn = psycopg2.connect(
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT")
)
cur = conn.cursor()

try:
    cur.execute("ALTER TABLE batch_results ADD COLUMN IF NOT EXISTS model_provider VARCHAR(50);")
    cur.execute("ALTER TABLE batch_results ADD COLUMN IF NOT EXISTS model_version VARCHAR(100);")
    cur.execute("ALTER TABLE batch_results ADD COLUMN IF NOT EXISTS input_price NUMERIC(10, 6);")
    cur.execute("ALTER TABLE batch_results ADD COLUMN IF NOT EXISTS output_price NUMERIC(10, 6);")
    conn.commit()
    print("Successfully added tracking columns to batch_results.")
except Exception as e:
    conn.rollback()
    print(f"Error altering table: {e}")
finally:
    cur.close()
    conn.close()
