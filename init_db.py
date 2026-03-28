import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import os
from dotenv import load_dotenv

def init_database():
    load_dotenv()
    db_name = os.getenv("DB_NAME")
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "5432")

    if not all([db_name, db_user, db_password]):
        print("Error: Missing required database environment variables in .env file.")
        print("Please ensure DB_NAME, DB_USER, and DB_PASSWORD are set.")
        return

    # Connect to the default 'postgres' database
    print(f"Connecting to PostgreSQL server at {db_host}:{db_port}...")
    try:
        conn = psycopg2.connect(
            dbname="postgres",
            user=db_user,
            password=db_password,
            host=db_host,
            port=db_port
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()

        # Check if DB exists
        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db_name,))
        exists = cur.fetchone()

        if not exists:
            print(f"Database '{db_name}' does not exist. Creating...")
            cur.execute(f"CREATE DATABASE {db_name}")
            print(f"Database '{db_name}' created successfully!")
        else:
            print(f"Database '{db_name}' already exists.")

        cur.close()
        conn.close()
    except Exception as e:
        print(f"Failed to connect or create database: {e}")
        print("Please check if PostgreSQL is running and credentials are correct.")
        return

    # Now we can safely import app and let it create tables within Flask's context
    print("Initializing tables via Flask-SQLAlchemy...")
    try:
        # Import inside function to avoid app.py startup crash if DB was missing
        from app import app, db
        with app.app_context():
            db.create_all()
            print("Successfully initialized all database tables.")
    except Exception as e:
        print(f"Failed to initialize tables: {e}")

if __name__ == "__main__":
    init_database()
