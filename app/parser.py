"""ULog -> plain Python dicts ready for DB"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pyulog import ULog

# PX4 vehicle_status.nav_state (subset)
NAV_STATES = {
    0: "MANUAL", 1: "ALTCTL", 2: "POSCTL", 3: "AUTO_MISSION", 4: "AUTO_LOITER",
    5: "AUTO_RTL", 10: "ACRO", 12: "DESCEND", 13: "TERMINATION", 14: "OFFBOARD",
    15: "STAB", 17: "AUTO_TAKEOFF", 18: "AUTO_LAND", 20: "AUTO_PRECLAND",
}


@dataclass
class ParsedLog:
    meta: dict
    params: dict[str, int | float] #Px4 logs all Ulog messages as float32 or int
    messages: list[dict]
    topics: list[dict]


def _get(ulog: ULog, name: str, multi_id: int = 0): #multi_id lets us choose which ekf, gps we are getting
    try:
        return ulog.get_dataset(name, multi_id).data
    except (KeyError, IndexError, ValueError):
        return None


def parse_ulog(path: str | Path) -> ParsedLog:
    ulog = ULog(str(path))
    info = ulog.msg_info_dict

    meta = {
        "filename": Path(path).name,
        "hw": info.get("ver_hw"),
        "sw_git": info.get("ver_sw"),
        "sw_branch": info.get("ver_sw_branch"),
        "sys_uuid": info.get("sys_uuid"), #In future planning to group based on flight controller uuid
        "duration_s": (ulog.last_timestamp - ulog.start_timestamp) / 1e6,
        "dropouts": len(ulog.dropouts),
    }

    # boot time UTC -> real start datetime (0 when no GPS time was available)
    boot_utc = info.get("boot_time_utc_us", 0)
    meta["start_utc"] = (
        pd.Timestamp(boot_utc + ulog.start_timestamp, unit="us", tz="UTC").isoformat()
        if boot_utc else None
    )

    vs = _get(ulog, "vehicle_status")
    if vs is not None:
        meta["nav_states"] = ",".join(
            NAV_STATES.get(int(s), str(int(s))) for s in np.unique(vs["nav_state"])
        )

    lp = _get(ulog, "vehicle_local_position")
    if lp is not None:
        meta["max_alt_m"] = float(-lp["z"].min())
        meta["max_hspeed_ms"] = float(np.hypot(lp["vx"], lp["vy"]).max())

    bat = _get(ulog, "battery_status")
    if bat is not None:
        meta["batt_v_start"] = float(bat["voltage_v"][0])
        meta["batt_v_min"] = float(bat["voltage_v"].min())
        meta["batt_used_pct"] = float((bat["remaining"][0] - bat["remaining"][-1]) * 100)
        meta["max_current_a"] = float(bat["current_a"].max())

    messages = [
        {"t_s": (m.timestamp - ulog.start_timestamp) / 1e6,
         "level": m.log_level_str(),
         "text": m.message.strip()}
        for m in ulog.logged_messages
    ]

    topics = [
        {"name": d.name, "multi_id": d.multi_id, "rows": len(d.data["timestamp"]),
         "fields": len(d.data)}
        for d in ulog.data_list
    ]

    return ParsedLog(meta=meta, params=dict(ulog.initial_parameters),
                     messages=messages, topics=topics)


if __name__ == "__main__":
    import sys
    p = parse_ulog(sys.argv[1])
    for k, v in p.meta.items():
        print(f"{k:16} {v}")
    print(f"{'params':16} {len(p.params)}")
    print(f"{'topics':16} {len(p.topics)}")
    for m in p.messages:
        if m["level"] in ("WARNING", "ERROR"):
            print(f"  {m['t_s']:7.2f}s {m['level']:8} {m['text']}")
