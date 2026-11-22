"""REST API.  Run: uvicorn app.main:app --reload   then open http://localhost:8000/docs"""
import hashlib
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Query, Response, UploadFile
from pyulog import ULog
from pyulog.core import ULogException
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db import SessionLocal, init_db
from app.ingest import ingest
from app.models import Flight, LogMessage, Parameter, Topic
from app.schemas import (FlightDetail, FlightSummary, MessageOut, ParameterOut,
                         TopicData, TopicOut)

UPLOAD_DIR = Path("logs/uploads")
MAX_UPLOAD_MB = 200


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()                       # runs once at server start
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(title="PX4 Log Analyzer", lifespan=lifespan)

from fastapi.responses import RedirectResponse

@app.get("/", include_in_schema=False)

def root():
    return RedirectResponse("/docs")

def get_db():
    """Dependency: one DB session per request, always closed afterwards."""
    with SessionLocal() as session:
        yield session


def get_flight(flight_id: int, db: Session = Depends(get_db)) -> Flight:
    flight = db.get(Flight, flight_id)
    if flight is None:
        raise HTTPException(404, f"Flight {flight_id} not found")
    return flight


# ---------- flights ----------

@app.post("/flights", response_model=FlightSummary, status_code=201)
def upload_flight(file: UploadFile, response: Response, db: Session = Depends(get_db)):
    if not (file.filename or "").lower().endswith(".ulg"):
        raise HTTPException(400, "Expected a .ulg file")

    content = file.file.read(MAX_UPLOAD_MB * 1024 * 1024 + 1)
    if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File larger than {MAX_UPLOAD_MB} MB")

    # store under its hash: same file uploaded twice -> same path, no copies
    digest = hashlib.sha256(content).hexdigest()
    existing = db.scalar(select(Flight).where(Flight.file_sha256 == digest))
    if existing:
        response.status_code = 200          # already have it: 200 instead of 201 Created
        return existing

    path = UPLOAD_DIR / f"{digest}.ulg"
    is_new_file = not path.exists()
    if is_new_file:
        path.write_bytes(content)

    try:
        flight = ingest(db, path, original_name=file.filename)
    except ULogException as e:
        if is_new_file:
            path.unlink(missing_ok=True)
        raise HTTPException(400, f"Not a valid ULog file: {e}")
    return flight


@app.get("/flights", response_model=list[FlightSummary])
def list_flights(limit: int = Query(50, le=500), offset: int = 0, db: Session = Depends(get_db)):
    stmt = select(Flight).order_by(Flight.uploaded_at.desc()).limit(limit).offset(offset)
    return db.scalars(stmt).all()


@app.get("/flights/{flight_id}", response_model=FlightDetail)
def flight_detail(flight: Flight = Depends(get_flight), db: Session = Depends(get_db)):
    def count(model, *where):
        return db.scalar(select(func.count()).select_from(model)
                         .where(model.flight_id == flight.id, *where))

    columns = {c.name: getattr(flight, c.name) for c in Flight.__table__.columns}
    return FlightDetail(
        **columns,                  # extra keys (file_path, sha) are ignored by Pydantic
        n_params=count(Parameter),
        n_messages=count(LogMessage),
        n_errors=count(LogMessage, LogMessage.level.in_(["ERROR", "CRITICAL"])),
    )


@app.delete("/flights/{flight_id}", status_code=204)
def delete_flight(flight: Flight = Depends(get_flight), db: Session = Depends(get_db)):
    path = Path(flight.file_path)
    db.delete(flight)               # cascade removes params/messages/topics
    db.commit()
    if path.parent.resolve() == UPLOAD_DIR.resolve():   # never delete files we didn't upload
        path.unlink(missing_ok=True)


# ---------- children of a flight ----------

@app.get("/flights/{flight_id}/params", response_model=list[ParameterOut])
def flight_params(
    flight: Flight = Depends(get_flight),
    db: Session = Depends(get_db),
    search: str | None = Query(None, description="Prefix filter, e.g. BAT1_ or MC_"),
):
    stmt = select(Parameter).where(Parameter.flight_id == flight.id)
    if search:
        stmt = stmt.where(Parameter.name.startswith(search.upper()))
    return db.scalars(stmt.order_by(Parameter.name)).all()


@app.get("/flights/{flight_id}/messages", response_model=list[MessageOut])
def flight_messages(
    flight: Flight = Depends(get_flight),
    db: Session = Depends(get_db),
    level: str | None = Query(None, description="INFO, WARNING, ERROR, ..."),
):
    stmt = select(LogMessage).where(LogMessage.flight_id == flight.id)
    if level:
        stmt = stmt.where(LogMessage.level == level.upper())
    return db.scalars(stmt.order_by(LogMessage.t_s)).all()


@app.get("/flights/{flight_id}/topics", response_model=list[TopicOut])
def flight_topics(flight: Flight = Depends(get_flight), db: Session = Depends(get_db)):
    stmt = select(Topic).where(Topic.flight_id == flight.id).order_by(Topic.name, Topic.multi_id)
    return db.scalars(stmt).all()


@app.get("/flights/{flight_id}/topics/{topic_name}", response_model=TopicData)
def topic_data(
    topic_name: str,
    flight: Flight = Depends(get_flight),
    db: Session = Depends(get_db),
    multi_id: int = 0,
    fields: str | None = Query(None, description="Comma separated, e.g. q[0],q[1]. Empty = all"),
    max_points: int = Query(2000, ge=10, le=100_000),
):
    """Time series are not in the DB: read just this topic from the .ulg on demand."""
    known = db.scalar(select(Topic).where(
        Topic.flight_id == flight.id, Topic.name == topic_name, Topic.multi_id == multi_id))
    if known is None:
        raise HTTPException(404, f"Topic {topic_name}[{multi_id}] not in flight {flight.id}")

    ulog = ULog(flight.file_path, [topic_name])
    data = ulog.get_dataset(topic_name, multi_id).data

    wanted = [f.strip() for f in fields.split(",")] if fields else [k for k in data if k != "timestamp"]
    missing = [f for f in wanted if f not in data]
    if missing:
        raise HTTPException(400, f"Unknown fields: {missing}")

    n = len(data["timestamp"])
    step = max(1, -(-n // max_points))   # ceil(n / max_points): plain decimation
    t_s = (data["timestamp"][::step] - ulog.start_timestamp) / 1e6

    def to_json(arr):
        arr = arr[::step].astype(float)
        return [None if np.isnan(x) else float(x) for x in arr]   # NaN is not valid JSON

    return TopicData(
        name=topic_name, multi_id=multi_id, total_rows=n, step=step, fields=wanted,
        data={"t_s": t_s.tolist(), **{f: to_json(data[f]) for f in wanted}},
    )
