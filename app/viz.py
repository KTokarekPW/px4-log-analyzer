"""Extract everything the 3D viewer needs from a .ulg into one compact JSON-able dict.

Usage (offline): python -m app.viz logs/sample.ulg -o viz.json
Times are seconds from log start. Positions are NED in metres (PX4 local frame).
"""
from pathlib import Path

import numpy as np
from pyulog import ULog

from app.parser import NAV_STATES

LEVEL_ORDER = {"EMERGENCY": 0, "ALERT": 1, "CRITICAL": 2, "ERROR": 3,
               "WARNING": 4, "NOTICE": 5, "INFO": 6, "DEBUG": 7}


def _ds(ulog: ULog, name: str, multi_id: int = 0):
    try:
        return ulog.get_dataset(name, multi_id).data
    except IndexError:
        return None


def _r(arr, nd=3) -> list:
    """numpy -> rounded list, NaN -> None (JSON has no NaN)."""
    a = np.round(np.asarray(arr, dtype=float), nd)
    return [None if np.isnan(x) else float(x) for x in a]


def _primary_rows(ulog: ULog, topic: str, t0: int):
    """Merge rows of a multi-instance estimator topic, keeping only the rows
    from whichever EKF2 instance was primary at that moment."""
    instances = {d.multi_id: d.data for d in ulog.data_list if d.name == topic}
    if not instances:
        return None
    sel = _ds(ulog, "estimator_selector_status")
    if sel is None or len(instances) == 1:
        data = instances[min(instances)]
        return data, np.full(len(data["timestamp"]), min(instances))

    sel_t, sel_p = sel["timestamp"], sel["primary_instance"]
    parts = []
    for inst, data in instances.items():
        idx = np.clip(np.searchsorted(sel_t, data["timestamp"], side="right") - 1, 0, None)
        keep = sel_p[idx] == inst
        parts.append((inst, data, keep))
    keys = parts[0][1].keys()
    merged = {k: np.concatenate([d[k][m] for _, d, m in parts]) for k in keys}
    inst_col = np.concatenate([np.full(m.sum(), i) for i, _, m in parts])
    order = np.argsort(merged["timestamp"])
    return {k: v[order] for k, v in merged.items()}, inst_col[order]


def _segments(t: np.ndarray, values: np.ndarray, t_end: float, label=lambda v: v):
    """Run-length encode a state signal into [{t0, t1, value}]."""
    if len(t) == 0:
        return []
    segs, start, cur = [], t[0], values[0]
    for ti, v in zip(t[1:], values[1:]):
        if v != cur:
            segs.append({"t0": round(float(start), 3), "t1": round(float(ti), 3), "value": label(cur)})
            start, cur = ti, v
    segs.append({"t0": round(float(start), 3), "t1": round(float(t_end), 3), "value": label(cur)})
    return segs


def build_viz(path: str | Path) -> dict:
    ulog = ULog(str(path))
    t0 = ulog.start_timestamp
    rel = lambda ts: (ts.astype(np.int64) - t0) / 1e6
    duration = (ulog.last_timestamp - t0) / 1e6
    info = ulog.msg_info_dict
    params = ulog.initial_parameters

    out: dict = {
        "meta": {
            "filename": Path(path).name,
            "hw": info.get("ver_hw"),
            "sw_git": info.get("ver_sw"),
            "duration_s": round(duration, 3),
        }
    }

    # --- position + EKF accuracy (selected estimator output, ~10 Hz) ---
    lp = _ds(ulog, "vehicle_local_position")
    if lp is None:
        raise ValueError("Log has no vehicle_local_position, nothing to visualise")
    out["pos"] = {
        "t": _r(rel(lp["timestamp"])),
        "n": _r(lp["x"]), "e": _r(lp["y"]), "d": _r(lp["z"]),
        "vn": _r(lp["vx"]), "ve": _r(lp["vy"]), "vd": _r(lp["vz"]),
        "eph": _r(lp["eph"], 4), "epv": _r(lp["epv"], 4),
    }

    # --- attitude quaternion body(FRD) -> NED, ~20-50 Hz ---
    att = _ds(ulog, "vehicle_attitude")
    if att is not None:
        out["att"] = {"t": _r(rel(att["timestamp"]))}
        for i, k in enumerate("wxyz"):
            out["att"][k] = _r(att[f"q[{i}]"], 5)

    # --- per-state 1-sigma from the primary EKF2 covariance diagonal ---
    es = _primary_rows(ulog, "estimator_states", t0)
    if es is not None:
        d, inst = es
        sd = lambda i: np.sqrt(np.abs(d[f"covariances[{i}]"]))
        n_states = int(d["n_states"][0])
        if n_states >= 25:
            # EKF2 >= v1.15 error-state layout: rot(0-2) vel(3-5) pos(6-8)
            rot, vel, pos = (0, 1, 2), (3, 4, 5), (6, 7, 8)
        else:
            # older layout: quat(0-3) vel(4-6) pos(7-9); no direct angle sigma
            rot, vel, pos = None, (4, 5, 6), (7, 8, 9)
        out["ekf"] = {
            "t": _r(rel(d["timestamp"])),
            "instance": [int(i) for i in inst],
            "sn": _r(sd(pos[0]), 4), "se": _r(sd(pos[1]), 4), "sd": _r(sd(pos[2]), 4),
            "svn": _r(sd(vel[0]), 4), "sve": _r(sd(vel[1]), 4), "svd": _r(sd(vel[2]), 4),
        }
        if rot:
            out["ekf"].update({
                "sroll": _r(np.degrees(sd(rot[0])), 3),
                "spitch": _r(np.degrees(sd(rot[1])), 3),
                "syaw": _r(np.degrees(sd(rot[2])), 3),
            })

    # --- innovation test ratios (>1 = measurement rejected by EKF) ---
    tr = _primary_rows(ulog, "estimator_innovation_test_ratios", t0)
    if tr is not None:
        d, _ = tr
        out["ratios"] = {
            "t": _r(rel(d["timestamp"])),
            "mag": _r(np.nanmax([d[f"mag_field[{i}]"] for i in range(3)], axis=0)),
            "heading": _r(d["heading"]),
            "gps_hpos": _r(np.nanmax([d["gps_hpos[0]"], d["gps_hpos[1]"]], axis=0)),
            "gps_vpos": _r(d["gps_vpos"]),
            "baro": _r(d["baro_vpos"]),
        }

    # --- motors + rotor geometry from control allocation params ---
    am = _ds(ulog, "actuator_motors")
    n_rot = int(params.get("CA_ROTOR_COUNT", 4))
    out["rotors"] = [
        {"px": params.get(f"CA_ROTOR{i}_PX", 0.0), "py": params.get(f"CA_ROTOR{i}_PY", 0.0),
         "km": params.get(f"CA_ROTOR{i}_KM", 0.05)}
        for i in range(n_rot)
    ]
    if am is not None:
        out["motors"] = {"t": _r(rel(am["timestamp"])),
                         "u": [_r(am[f"control[{i}]"]) for i in range(n_rot)]}

    bat = _ds(ulog, "battery_status")
    if bat is not None:
        out["batt"] = {"t": _r(rel(bat["timestamp"])), "v": _r(bat["voltage_v"], 2)}

    # --- flight modes and armed state as time segments ---
    vs = _ds(ulog, "vehicle_status")
    if vs is not None:
        t = rel(vs["timestamp"])
        out["modes"] = _segments(t, vs["nav_state"], duration,
                                 lambda v: NAV_STATES.get(int(v), f"NAV_{int(v)}"))
        out["armed"] = [s for s in _segments(t, vs["arming_state"] == 2, duration, bool)
                        if s["value"]]

    out["messages"] = [
        {"t": round((m.timestamp - t0) / 1e6, 3), "level": m.log_level_str(),
         "text": m.message.strip()}
        for m in ulog.logged_messages
    ]
    return out


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("ulg")
    ap.add_argument("-o", "--out", default="viz.json")
    a = ap.parse_args()
    data = build_viz(a.ulg)
    Path(a.out).write_text(json.dumps(data, separators=(",", ":")))
    print(f"{a.out}: {Path(a.out).stat().st_size / 1024:.0f} kB, "
          f"{len(data['pos']['t'])} pos, {len(data.get('att', {}).get('t', []))} att, "
          f"{len(data.get('ekf', {}).get('t', []))} ekf, {len(data['messages'])} msgs")
