"""Database connection. One SQLite file next to the project."""
from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, sessionmaker

DATABASE_URL = "sqlite:///./px4logs.db"

# check_same_thread=False FastAPI may use the connection from another thread
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})


@event.listens_for(engine, "connect")
def _enable_foreign_keys(dbapi_conn, _):
    # SQLite ignores FOREIGN KEY constraints unless this is on (per connection)
    dbapi_conn.execute("PRAGMA foreign_keys = ON")


SessionLocal = sessionmaker(bind=engine)


class Base(DeclarativeBase):
    pass


def init_db() -> None:
    from app import models  # noqa: F401  registers the tables on Base
    Base.metadata.create_all(engine)
