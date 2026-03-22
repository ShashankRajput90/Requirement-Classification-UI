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
cur.execute("SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'batch_results'")
print('batch_results columns:', cur.fetchall())
cur.execute("SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'api_usage'")
print('api_usage columns:', cur.fetchall())
