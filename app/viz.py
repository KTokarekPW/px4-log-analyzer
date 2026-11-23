"""Extract everything the 3D viewer needs from a .ulg into one compact JSON-able dict."""
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
    out["channels"] = build_channels(ulog)
    return out


# ======================================================================
# Channel catalog: every EKF2-related signal the viewer can plot on demand
# ======================================================================

# EKF2 >= v1.15 error-state covariance diagonal (24 entries), name, unit, rad->deg
COV25 = [("roll", "deg", 1), ("pitch", "deg", 1), ("yaw", "deg", 1),
         ("vel N", "m/s", 0), ("vel E", "m/s", 0), ("vel D", "m/s", 0),
         ("pos N", "m", 0), ("pos E", "m", 0), ("pos D", "m", 0),
         ("gyro bias X", "deg/s", 1), ("gyro bias Y", "deg/s", 1), ("gyro bias Z", "deg/s", 1),
         ("accel bias X", "m/s²", 0), ("accel bias Y", "m/s²", 0), ("accel bias Z", "m/s²", 0),
         ("mag earth N", "G", 0), ("mag earth E", "G", 0), ("mag earth D", "G", 0),
         ("mag body X", "G", 0), ("mag body Y", "G", 0), ("mag body Z", "G", 0),
         ("wind N", "m/s", 0), ("wind E", "m/s", 0), ("terrain", "m", 0)]
COV24_OLD = [("q0", "-", 0), ("q1", "-", 0), ("q2", "-", 0), ("q3", "-", 0),
             ("vel N", "m/s", 0), ("vel E", "m/s", 0), ("vel D", "m/s", 0),
             ("pos N", "m", 0), ("pos E", "m", 0), ("pos D", "m", 0),
             ("dAng bias X", "rad", 0), ("dAng bias Y", "rad", 0), ("dAng bias Z", "rad", 0),
             ("dVel bias X", "m/s", 0), ("dVel bias Y", "m/s", 0), ("dVel bias Z", "m/s", 0),
             ("mag earth N", "G", 0), ("mag earth E", "G", 0), ("mag earth D", "G", 0),
             ("mag body X", "G", 0), ("mag body Y", "G", 0), ("mag body Z", "G", 0),
             ("wind N", "m/s", 0), ("wind E", "m/s", 0)]
# nominal state vector for n_states == 25 (index, name, unit, rad->deg); q, vel, pos skipped
STATES25 = [(10, "gyro bias X", "deg/s", 1), (11, "gyro bias Y", "deg/s", 1), (12, "gyro bias Z", "deg/s", 1),
            (13, "accel bias X", "m/s²", 0), (14, "accel bias Y", "m/s²", 0), (15, "accel bias Z", "m/s²", 0),
            (16, "mag earth N", "G", 0), (17, "mag earth E", "G", 0), (18, "mag earth D", "G", 0),
            (19, "mag body X", "G", 0), (20, "mag body Y", "G", 0), (21, "mag body Z", "G", 0),
            (22, "wind N", "m/s", 0), (23, "wind E", "m/s", 0), (24, "terrain", "m", 0)]

DEFAULT_CHANNELS = {"cov.pos_hv", "cov.yaw", "ratio.main",
                    "flag.cs_gnss_pos", "flag.cs_mag_hdg", "flag.cs_mag_3d", "flag.cs_mag_field_disturbed"}


def _sig(v: np.ndarray, digits: int = 5) -> list:
    """Round to ~`digits` significant digits of the channel's magnitude."""
    finite = np.abs(v[np.isfinite(v)])
    mag = finite.max() if finite.size and finite.max() > 0 else 1.0
    nd = int(np.clip(digits - 1 - np.floor(np.log10(mag)), 0, 9))
    return _r(v, nd)


def _innov_unit(name: str) -> tuple[str, bool]:
    """(unit, convert rad->deg) guessed from an innovation field name."""
    if "heading" in name or "beta" in name:
        return "deg", True
    if "mag" in name:
        return "G", False
    if "flow" in name:
        return "rad/s", False
    if "vel" in name or "airspeed" in name or "drag" in name:
        return "m/s", False
    if "gravity" in name:
        return "m/s²", False
    return "m", False


class _Catalog:
    def __init__(self, t0: int):
        self.t0, self.times, self.list = t0, {}, []

    def tkey(self, key: str, ts: np.ndarray) -> str:
        if key not in self.times:
            self.times[key] = _r((ts.astype(np.int64) - self.t0) / 1e6)
        return key

    def line(self, cid, group, label, unit, tk, series: list[tuple[str, np.ndarray]]):
        out = []
        for name, v in series:
            v = np.asarray(v, dtype=float)
            if np.isfinite(v).any() and np.nanmax(np.abs(v)) > 0:
                out.append({"label": name, "v": _sig(v)})
        if out:
            self.list.append({"id": cid, "group": group, "label": label, "unit": unit,
                              "kind": "line", "tk": tk, "series": out,
                              "default": cid in DEFAULT_CHANNELS})

    def flag(self, cid, group, label, tk, v):
        v = np.nan_to_num(np.asarray(v, dtype=float)) != 0
        if v.any():                                   # skip flags that were never set
            self.list.append({"id": cid, "group": group, "label": label, "unit": "",
                              "kind": "bool", "tk": tk,
                              "series": [{"label": label, "v": v.astype(int).tolist()}],
                              "default": cid in DEFAULT_CHANNELS})


def build_channels(ulog: ULog) -> dict:
    t0 = ulog.start_timestamp
    cat = _Catalog(t0)
    deg = np.degrees

    # --- covariance diagonal -> 1-sigma per state, plus states themselves
    es = _primary_rows(ulog, "estimator_states", t0)
    if es is not None:
        d, _ = es
        tk = cat.tkey("es", d["timestamp"])
        n_states = int(d["n_states"][0])
        layout = COV25 if n_states >= 25 else COV24_OLD
        sd = lambda i: np.sqrt(np.abs(d[f"covariances[{i}]"]))
        G = "EKF σ (covariance)"
        if n_states >= 25:
            cat.line("cov.pos_hv", G, "position σ horizontal / vertical", "m", tk,
                     [("σ_h", np.hypot(sd(6), sd(7))), ("σ_v", sd(8))])
            cat.line("cov.yaw", G, "yaw σ", "deg", tk, [("σ_yaw", deg(sd(2)))])
            cat.line("cov.tilt", G, "roll / pitch σ", "deg", tk,
                     [("σ_roll", deg(sd(0))), ("σ_pitch", deg(sd(1)))])
        for i, (name, unit, to_deg) in enumerate(layout):
            if f"covariances[{i}]" in d:
                v = deg(sd(i)) if to_deg else sd(i)
                cat.line(f"cov.{i}", G, f"{name} σ", unit, tk, [(name, v)])
        if n_states >= 25:
            G = "EKF states"
            for i, name, unit, to_deg in STATES25:
                v = d[f"states[{i}]"]
                cat.line(f"state.{i}", G, name, unit, tk, [(name, deg(v) if to_deg else v)])
            cat.line("state.magI_norm", G, "earth field strength |B|", "G", tk,
                     [("|B| est", np.sqrt(sum(d[f"states[{i}]"] ** 2 for i in (16, 17, 18))))])

    # --- innovation test ratios (>1 = rejected) and raw innovations
    tr = _primary_rows(ulog, "estimator_innovation_test_ratios", t0)
    if tr is not None:
        d, _ = tr
        tk = cat.tkey("tr", d["timestamp"])
        mx = lambda *k: np.nanmax([d[x] for x in k], axis=0)
        cat.line("ratio.main", "Innovation test ratios", "mag / gps hpos / baro", "ratio", tk,
                 [("mag", mx("mag_field[0]", "mag_field[1]", "mag_field[2]")),
                  ("gps", mx("gps_hpos[0]", "gps_hpos[1]")), ("baro", d["baro_vpos"])])
        for k in d:
            if not k.startswith("timestamp"):
                cat.line(f"ratio.{k}", "Innovation test ratios", k, "ratio", tk, [(k, d[k])])
    inn = _primary_rows(ulog, "estimator_innovations", t0)
    if inn is not None:
        d, _ = inn
        tk = cat.tkey("inn", d["timestamp"])
        var = _primary_rows(ulog, "estimator_innovation_variances", t0)
        for k in d:
            if k.startswith("timestamp"):
                continue
            unit, to_deg = _innov_unit(k)
            v = deg(d[k]) if to_deg else d[k]
            series = [(k, v)]
            if var is not None and k in var[0] and len(var[0][k]) == len(v):
                s = np.sqrt(np.abs(var[0][k]))
                series += [("±σ", deg(s) if to_deg else s)]
            cat.line(f"inn.{k}", "Innovations", k, unit, tk, series)

    # --- per-source aiding: test ratio, fused, rejected
    for name in sorted({x.name for x in ulog.data_list if x.name.startswith("estimator_aid_src_")}):
        src = name.removeprefix("estimator_aid_src_")
        rows = _primary_rows(ulog, name, t0)
        if rows is None:
            continue
        d, _ = rows
        tk = cat.tkey(f"aid.{src}", d["timestamp"])
        trk = [k for k in d if k == "test_ratio" or k.startswith("test_ratio[")]
        if trk:
            cat.line(f"aid.{src}.ratio", "Aid sources", f"{src} test ratio", "ratio", tk,
                     [(src, np.nanmax([d[k] for k in trk], axis=0))])
        cat.flag(f"aid.{src}.fused", "Aid sources", f"{src} fused", tk, d["fused"])
        cat.flag(f"aid.{src}.rejected", "Aid sources", f"{src} rejected", tk, d["innovation_rejected"])

    # --- status flags: control status, faults, rejections
    fl = _primary_rows(ulog, "estimator_status_flags", t0)
    if fl is not None:
        d, _ = fl
        tk = cat.tkey("flags", d["timestamp"])
        groups = {"cs_": "Control status (cs_)", "fs_": "Fault status (fs_)", "reject_": "Rejections (reject_)"}
        for k in d:
            for pre, grp in groups.items():
                if k.startswith(pre):
                    cat.flag(f"flag.{k}", grp, k, tk, d[k])

    ev = _primary_rows(ulog, "estimator_event_flags", t0)
    if ev is not None:
        d, _ = ev
        tk = cat.tkey("events", d["timestamp"])
        for k in d:
            if not k.startswith("timestamp") and k != "information_event_changes":
                cat.flag(f"event.{k}", "Events", k, tk, d[k])

    # --- estimator_status: accuracies, test ratios, mag field vs WMM, resets
    st = _primary_rows(ulog, "estimator_status", t0)
    if st is not None:
        d, _ = st
        tk = cat.tkey("status", d["timestamp"])
        G = "Estimator status"
        cat.line("status.acc", G, "position accuracy h / v", "m", tk,
                 [("horiz", d["pos_horiz_accuracy"]), ("vert", d["pos_vert_accuracy"])])
        if "mag_strength_gs" in d:
            cat.line("status.mag_strength", G, "mag strength measured / WMM", "G", tk,
                     [("meas", d["mag_strength_gs"]), ("WMM", d["mag_strength_ref_gs"])])
            cat.line("status.mag_incl", G, "mag inclination measured / WMM", "deg", tk,
                     [("meas", d["mag_inclination_deg"]), ("WMM", d["mag_inclination_ref_deg"])])
        for k in ("hdg_test_ratio", "vel_test_ratio", "pos_test_ratio", "hgt_test_ratio",
                  "tas_test_ratio", "hagl_test_ratio", "beta_test_ratio"):
            if k in d:
                cat.line(f"status.{k}", G, k, "ratio", tk, [(k, d[k])])
        if "time_slip" in d:
            cat.line("status.time_slip", G, "time slip", "s", tk, [("slip", d["time_slip"])])
        trk = [f"output_tracking_error[{i}]" for i in range(3)]
        if all(k in d for k in trk):
            cat.line("status.track", G, "output tracking error", "-", tk,
                     [("ang", d[trk[0]]), ("vel", d[trk[1]]), ("pos", d[trk[2]])])
        for k in d:
            if k.startswith("reset_count_"):
                cat.line(f"status.{k}", G, k, "count", tk, [(k, d[k])])
            elif k.startswith("pre_flt_fail_"):
                cat.flag(f"status.{k}", G, k, tk, d[k])

    # --- yaw: EKF vs EKF-GSF emergency yaw estimator
    gsf = _primary_rows(ulog, "yaw_estimator_status", t0)
    att = _ds(ulog, "vehicle_attitude")
    if gsf is not None:
        d, _ = gsf
        tk = cat.tkey("gsf", d["timestamp"])
        series = [("GSF", deg(d["yaw_composite"]))]
        if att is not None:
            ta = att["timestamp"]
            w, x, y, z = (np.interp(d["timestamp"], ta, att[f"q[{i}]"]) for i in range(4))
            series.insert(0, ("EKF", deg(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))))
        cat.line("yaw.compare", "Yaw", "yaw EKF / GSF", "deg", tk, series)
        cat.line("yaw.gsf_sigma", "Yaw", "GSF yaw σ", "deg", tk,
                 [("σ", deg(np.sqrt(np.abs(d["yaw_variance"]))))])
        if "yaw_composite_valid" in d:
            cat.flag("yaw.gsf_valid", "Yaw", "GSF yaw valid", tk, d["yaw_composite_valid"])

    # --- multi-EKF selector
    sel = _ds(ulog, "estimator_selector_status")
    if sel is not None:
        tk = cat.tkey("sel", sel["timestamp"])
        n = int(np.max(sel["instances_available"])) if "instances_available" in sel else 2
        cat.line("sel.primary", "Selector", "primary instance", "-", tk,
                 [("primary", sel["primary_instance"].astype(float) + 1e-9)])
        cat.line("sel.combined", "Selector", "combined test ratio per instance", "ratio", tk,
                 [(f"#{i}", sel[f"combined_test_ratio[{i}]"]) for i in range(n)])
        cat.line("sel.relative", "Selector", "relative test ratio per instance", "-", tk,
                 [(f"#{i}", sel[f"relative_test_ratio[{i}]"]) for i in range(n)])

    # --- GPS quality
    gps = _ds(ulog, "vehicle_gps_position")
    if gps is not None:
        tk = cat.tkey("gps", gps["timestamp"])
        G = "GPS"
        cat.line("gps.sats", G, "satellites used", "-", tk, [("sats", gps["satellites_used"])])
        cat.line("gps.ephv", G, "eph / epv", "m", tk, [("eph", gps["eph"]), ("epv", gps["epv"])])
        cat.line("gps.dop", G, "hdop / vdop", "-", tk, [("hdop", gps["hdop"]), ("vdop", gps["vdop"])])
        cat.line("gps.svar", G, "speed accuracy", "m/s", tk, [("s_var", gps["s_variance_m_s"])])
        for k, lbl in (("noise_per_ms", "noise per ms"), ("jamming_indicator", "jamming indicator"),
                       ("automatic_gain_control", "AGC"), ("fix_type", "fix type")):
            if k in gps:
                cat.line(f"gps.{k}", G, lbl, "-", tk, [(lbl, gps[k])])

    # --- IMU health / vibration (all instances)
    for dd in (x for x in ulog.data_list if x.name == "vehicle_imu_status"):
        d, i = dd.data, dd.multi_id
        tk = cat.tkey(f"imu{i}", d["timestamp"])
        G = "IMU / vibration"
        cat.line(f"imu{i}.vib", G, f"IMU{i} vibration accel / gyro", "-", tk,
                 [("accel", d["accel_vibration_metric"]), ("gyro", d["gyro_vibration_metric"])])
        cat.line(f"imu{i}.coning", G, f"IMU{i} coning", "-", tk, [("coning", d["delta_angle_coning_metric"])])
        clip = sum(d[f"accel_clipping[{k}]"] for k in range(3))
        cat.line(f"imu{i}.clip", G, f"IMU{i} accel clipping", "count", tk, [("clip", clip)])
        cat.line(f"imu{i}.temp", G, f"IMU{i} temperature", "°C", tk, [("temp", d["temperature_accel"])])

    # --- magnetometer raw field strength
    vm = _ds(ulog, "vehicle_magnetometer")
    if vm is not None:
        tk = cat.tkey("mag", vm["timestamp"])
        b = [vm[f"magnetometer_ga[{k}]"] for k in range(3)]
        cat.line("mag.norm", "Magnetometer", "|B| measured", "G", tk, [("|B|", np.sqrt(sum(x ** 2 for x in b)))])
        cat.line("mag.xyz", "Magnetometer", "field X / Y / Z", "G", tk, [("X", b[0]), ("Y", b[1]), ("Z", b[2])])

    return {"times": cat.times, "list": cat.list}


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