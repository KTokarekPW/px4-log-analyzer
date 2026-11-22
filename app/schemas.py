"""Pydantic models = shape of JSON going out of the API (separate from DB tables)."""
from datetime import datetime

from pydantic import BaseModel, ConfigDict, model_validator


class ORMModel(BaseModel):
    # allows FlightSummary.model_validate(flight_orm_object)
    model_config = ConfigDict(from_attributes=True)


class FlightSummary(ORMModel):
    id: int
    filename: str
    uploaded_at: datetime
    start_utc: str | None
    duration_s: float
    hw: str | None
    max_alt_m: float | None


class FlightDetail(FlightSummary):
    sw_git: str | None
    sw_branch: str | None
    sys_uuid: str | None
    dropouts: int
    nav_states: str | None
    max_hspeed_ms: float | None
    batt_v_start: float | None
    batt_v_min: float | None
    batt_used_pct: float | None
    max_current_a: float | None
    n_params: int
    n_messages: int
    n_errors: int


class ParameterOut(ORMModel):
    name: str
    type: str
    value: int | float

    @model_validator(mode="after")
    def _int_back(self):
        # DB keeps everything as REAL; give INT32 params back as int
        if self.type == "INT32":
            self.value = int(self.value)
        return self


class MessageOut(ORMModel):
    t_s: float
    level: str
    text: str


class TopicOut(ORMModel):
    name: str
    multi_id: int
    rows: int
    fields: int


class TopicData(BaseModel):
    name: str
    multi_id: int
    total_rows: int
    step: int                                # every Nth sample returned
    fields: list[str]
    data: dict[str, list[float | None]]      # column -> values, always includes t_s
