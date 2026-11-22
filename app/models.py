"""Tables. One Flight row per .ulg file, the rest hang off it via flight_id."""
from datetime import UTC, datetime

from sqlalchemy import ForeignKey, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


class Flight(Base):
    __tablename__ = "flights"

    id: Mapped[int] = mapped_column(primary_key=True)
    filename: Mapped[str]
    file_path: Mapped[str]                       # Where the .ulg lives on disk
    file_sha256: Mapped[str] = mapped_column(String(64), unique=True)  # Blocks duplicate uploads
    uploaded_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(UTC))

    hw: Mapped[str | None]
    sw_git: Mapped[str | None]
    sw_branch: Mapped[str | None]
    sys_uuid: Mapped[str | None]
    start_utc: Mapped[str | None]
    duration_s: Mapped[float]
    dropouts: Mapped[int]
    nav_states: Mapped[str | None]
    max_alt_m: Mapped[float | None]
    max_hspeed_ms: Mapped[float | None]
    batt_v_start: Mapped[float | None]
    batt_v_min: Mapped[float | None]
    batt_used_pct: Mapped[float | None]
    max_current_a: Mapped[float | None]

    #Deleting a Flight deletes its params/messages/topics too
    parameters: Mapped[list["Parameter"]] = relationship(back_populates="flight", cascade="all, delete-orphan")
    messages: Mapped[list["LogMessage"]] = relationship(back_populates="flight", cascade="all, delete-orphan")
    topics: Mapped[list["Topic"]] = relationship(back_populates="flight", cascade="all, delete-orphan")


class Parameter(Base):
    __tablename__ = "parameters"
    __table_args__ = (UniqueConstraint("flight_id", "name"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    flight_id: Mapped[int] = mapped_column(ForeignKey("flights.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(index=True)
    type: Mapped[str]       # "INT32" or "FLOAT"
    value: Mapped[float]    # int32 fits exactly in float64, recast if int

    flight: Mapped[Flight] = relationship(back_populates="parameters")


class LogMessage(Base):
    __tablename__ = "log_messages"

    id: Mapped[int] = mapped_column(primary_key=True)
    flight_id: Mapped[int] = mapped_column(ForeignKey("flights.id", ondelete="CASCADE"), index=True)
    t_s: Mapped[float]
    level: Mapped[str]
    text: Mapped[str]

    flight: Mapped[Flight] = relationship(back_populates="messages")


class Topic(Base):
    __tablename__ = "topics"
    __table_args__ = (UniqueConstraint("flight_id", "name", "multi_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    flight_id: Mapped[int] = mapped_column(ForeignKey("flights.id", ondelete="CASCADE"), index=True)
    name: Mapped[str]
    multi_id: Mapped[int]
    rows: Mapped[int]
    fields: Mapped[int]

    flight: Mapped[Flight] = relationship(back_populates="topics")
