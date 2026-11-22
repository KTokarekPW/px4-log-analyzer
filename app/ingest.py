"""Parse a .ulg and store it in the DB.  Usage: python -m app.ingest logs/sample.ulg"""
import hashlib
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import SessionLocal, init_db
from app.models import Flight, LogMessage, Parameter, Topic
from app.parser import parse_ulog


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ingest(session: Session, path: str | Path, original_name: str | None = None) -> Flight:
    path = Path(path).resolve()
    digest = sha256_of(path)

    existing = session.scalar(select(Flight).where(Flight.file_sha256 == digest))
    if existing:
        return existing

    p = parse_ulog(path)
    if original_name:               # uploads are stored as <sha>.ulg, keep the user's name
        p.meta["filename"] = original_name
    flight = Flight(file_path=str(path), file_sha256=digest, **p.meta)
    flight.parameters = [
        Parameter(name=k, type="INT32" if isinstance(v, int) else "FLOAT", value=float(v))
        for k, v in p.params.items()
    ]
    flight.messages = [LogMessage(**m) for m in p.messages]
    flight.topics = [Topic(**t) for t in p.topics]

    session.add(flight)   # children are added through the relationships
    session.commit()      # one transaction: all rows or none
    return flight


if __name__ == "__main__":
    import sys

    init_db()
    with SessionLocal() as s:
        for arg in sys.argv[1:]:
            f = ingest(s, arg)
            print(f"flight id={f.id}  {f.filename}  {f.duration_s:.1f}s  "
                  f"params={len(f.parameters)} msgs={len(f.messages)} topics={len(f.topics)}")
