import streamlit as st
import tempfile
import os
from pymavlink import mavutil
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF
from io import BytesIO
from typing import Dict, List, Tuple, Any

# -----------------------------
# Constants and mappings
# -----------------------------
ERR_SUBSYS_CODES = {
    0: "Main",
    1: "Radio",
    2: "Compass",
    3: "Optical Flow",
    4: "GPS",
    5: "Battery",
    6: "Flight Mode",
    8: "EKF",
}

ERR_ERROR_CODES = {
    0: "Unspecified",
    1: "Inconsistent",
    2: "Missing",
    3: "Too Large",
    4: "Too Small",
    5: "Timeout",
}

EV_ID_MAP = {
    11: "Landing complete",
    15: "Arm",
    17: "Disarm",
    18: "Failsafe",
    28: "Flight mode change",
    56: "Auto mission started",
    57: "Auto mission complete",
    62: "EKF failsafe triggered",
}

# -----------------------------
# Page config & header
# -----------------------------
st.set_page_config(page_title="Jamie D Flight Analysis Tool", layout="wide")
st.title("Jamie D Flight Analysis Tool")
st.markdown(
    """
Upload your ArduPilot `.BIN` log file for analysis. This tool helps assess flight health and detect potential issues.
    """
)

# -----------------------------
# Helpers
# -----------------------------

def _safe_time_s(msg: Any) -> float | None:
    """Return message time in seconds if available, else None."""
    timeus = getattr(msg, "TimeUS", None)
    if timeus is not None:
        return round(timeus / 1e6, 2)
    time_s = getattr(msg, "TimeS", None)
    if time_s is not None:
        try:
            return float(time_s)
        except Exception:
            return None
    return None


@st.cache_data(show_spinner=False)
def parse_log(file_bytes: bytes) -> Dict[str, Any]:
    """Parse a BIN log (bytes) and return structured data for the app."""
    # Temp file for pymavlink
    with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as tf:
        tf.write(file_bytes)
        temp_filename = tf.name

    error_msgs: List[str] = []
    event_msgs: List[str] = []
    altitude_data: List[Dict[str, Any]] = []
    battery_by_pack: Dict[int, List[Dict[str, Any]]] = {}
    attitude_data: List[Dict[str, Any]] = []
    esc_data: List[Dict[str, Any]] = []
    vibe_data: List[Dict[str, Any]] = []
    rc_data: List[Dict[str, Any]] = []
    rcout_data: List[Dict[str, Any]] = []
    mode_data: List[Tuple[float, str]] = []

    try:
        mlog = mavutil.mavlink_connection(temp_filename)
        count = 0
        while True:
            msg = mlog.recv_match()
            if msg is None:
                break
            count += 1

            t = _safe_time_s(msg)
            if t is None:
                continue

            msg_type = msg.get_type()

            if msg_type == "ERR":
                subsys = getattr(msg, "Subsys", -1)
                ecode = getattr(msg, "ECode", -1)
                subsys_str = ERR_SUBSYS_CODES.get(subsys, f"Unknown Subsys {subsys}")
                ecode_str = ERR_ERROR_CODES.get(ecode, f"Unknown ECode {ecode}")
                error_msgs.append(f"ERR at {t}s: {subsys_str} - {ecode_str}")

            elif msg_type == "EV":
                eid = getattr(msg, "Id", -1)
                event_str = EV_ID_MAP.get(eid, f"Unknown Event ID {eid}")
                event_msgs.append(f"EV at {t}s: {event_str}")
                if eid == 28:  # mode change marker
                    mode_data.append((t, event_str))

            elif msg_type == "GPS":
                alt_cm = getattr(msg, "Alt", None)
                if alt_cm is not None:
                    altitude_data.append({"Time (s)": t, "Alt_m": alt_cm / 100.0})

            elif msg_type == "BAT":
                pack = getattr(msg, "SNum", 0)  # battery index/serial number
                entry = {
                    "Time (s)": t,
                    "Volt": getattr(msg, "Volt", None),
                    "VoltR": getattr(msg, "VoltR", None),
                    "Curr": getattr(msg, "Curr", None),
                    "CurrTot": getattr(msg, "CurrTot", None),
                    "Temp": getattr(msg, "Temp", None),
                    "RemPct": getattr(msg, "RemPct", None),
                    "SNum": pack,
                }
                battery_by_pack.setdefault(int(pack), []).append(entry)

            elif msg_type == "ATT":
                attitude_data.append(
                    {
                        "Time (s)": t,
                        "DesRoll": getattr(msg, "DesRoll", None),
                        "Roll": getattr(msg, "Roll", None),
                    }
                )

            elif msg_type == "ESC":
                esc_data.append({"Time (s)": t, "Temp": getattr(msg, "Temp", None)})

            elif msg_type == "VIBE":
                vibe_data.append(
                    {
                        "Time (s)": t,
                        "VibeX": getattr(msg, "VibeX", None),
                        "VibeY": getattr(msg, "VibeY", None),
                        "VibeZ": getattr(msg, "VibeZ", None),
                    }
                )

            elif msg_type == "RCIN":
                rc_data.append(
                    {
                        "Time (s)": t,
                        "C1": getattr(msg, "C1", None),
                        "C2": getattr(msg, "C2", None),
                        "C3": getattr(msg, "C3", None),
                        "C4": getattr(msg, "C4", None),
                    }
                )

            elif msg_type == "RCOU":
                # Capture as many channels as present (1..16)
                fields = {f"C{i}": getattr(msg, f"C{i}", None) for i in range(1, 17)}
                fields["Time (s)"] = t
                rcout_data.append(fields)

        # Convert to DataFrames where useful
        altitude_df = pd.DataFrame(altitude_data)
        attitude_df = pd.DataFrame(attitude_data)
        esc_df = pd.DataFrame(esc_data)
        vibe_df = pd.DataFrame(vibe_data)
        rc_df = pd.DataFrame(rc_data)
        rcout_df = pd.DataFrame(rcout_data)
        battery_dfs = {k: pd.DataFrame(v) for k, v in battery_by_pack.items()}

        return {
            "errors": error_msgs,
            "events": event_msgs,
            "modes": mode_data,
            "altitude_df": altitude_df,
            "attitude_df": attitude_df,
            "esc_df": esc_df,
            "vibe_df": vibe_df,
            "rc_df": rc_df,
            "rcout_df": rcout_df,
            "battery_dfs": battery_dfs,
        }

    finally:
        if os.path.exists(temp_filename):
            try:
                os.remove(temp_filename)
            except Exception:
                pass


def plot_chart(df: pd.DataFrame, title: str, ylabel: str, keys: List[str]):
    if df is None or df.empty:
        st.info(f"No data available for {title}")
        return None
    available = [k for k in keys if k in df.columns]
    if not available:
        st.info(f"No matching fields for {title}")
        return None
    plot_df = df[["Time (s)"] + available].dropna(how="all", subset=available)
    if plot_df.empty:
        st.info(f"No valid samples for {title}")
        return None
    fig, ax = plt.subplots(figsize=(10, 4))
    for key in available:
        ax.plot(plot_df["Time (s)"], plot_df[key], label=key, linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Flight Duration (s)")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
    st.download_button(
        label=f"📥 Download {title} CSV",
        data=plot_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{title.replace(' ', '_').lower()}.csv",
        mime="text/csv",
    )
    return plot_df


# -----------------------------
# Sidebar: uploader & options
# -----------------------------
with st.sidebar:
    uploaded_file = st.file_uploader("Upload a .BIN file", type=["bin"])
    st.caption("Tip: nothing you do locally will change your deployed app until you push to GitHub.")

if uploaded_file is None:
    st.info("👆 Upload an ArduPilot .BIN log to begin.")
    st.stop()

st.success("File uploaded successfully!")

# Parse with caching to avoid re-work on reruns
with st.spinner("Parsing log…"):
    parsed = parse_log(uploaded_file.getbuffer().tobytes())

# -----------------------------
# Errors & events
# -----------------------------
if parsed["errors"]:
    with st.expander("❌ Error Messages", expanded=True):
        for err in parsed["errors"]:
            st.code(err)

if parsed["events"]:
    with st.expander("⚠️ System Events", expanded=True):
        for ev in parsed["events"]:
            st.code(ev)

# -----------------------------
# Charts
# -----------------------------
with st.expander("🪫 Battery Metrics"):
    if not parsed["battery_dfs"]:
        st.info("No battery messages found.")
    else:
        for pack, bdf in sorted(parsed["battery_dfs"].items()):
            st.subheader(f"Battery {pack} Metrics")
            plot_chart(bdf, f"Battery {pack} Metrics", "Value", ["Volt", "Curr", "Temp", "RemPct"])  # include RemPct if present

with st.expander("🛰️ GPS Altitude"):
    plot_chart(parsed["altitude_df"], "GPS Altitude Over Time", "Altitude (m)", ["Alt_m"])  

with st.expander("🧭 Attitude (Roll)"):
    plot_chart(parsed["attitude_df"], "Desired vs Actual Roll", "Degrees", ["DesRoll", "Roll"])  

with st.expander("🧊 ESC Temperature"):
    plot_chart(parsed["esc_df"], "ESC Temperature Over Time", "Temp (°C)", ["Temp"])  

with st.expander("🔧 Vibration Metrics"):
    plot_chart(parsed["vibe_df"], "Vibration Metrics", "Vibe", ["VibeX", "VibeY", "VibeZ"])  

with st.expander("🎮 RC Inputs"):
    plot_chart(parsed["rc_df"], "RC Input (C1–C4)", "PWM", ["C1", "C2", "C3", "C4"])  

with st.expander("⚡ Motor Outputs"):
    # Auto-detect available RCOU channels
    rcout_cols = [c for c in parsed["rcout_df"].columns if c.startswith("C")]
    # Prefer common groupings; fallback to whatever is available
    preferred = [f"C{i}" for i in range(1, 13)]
    keys = [c for c in preferred if c in rcout_cols] or rcout_cols
    plot_chart(parsed["rcout_df"], "Motor Output (RCOU)", "PWM", keys)

# -----------------------------
# Mode timeline (event markers)
# -----------------------------
if parsed["modes"]:
    with st.expander("🧭 Flight Modes Timeline"):
        for t, mode in parsed["modes"]:
            st.write(f"{t}s: {mode}")

# -----------------------------
# PDF report export
# -----------------------------
with st.expander("📄 Export Summary"):
    if st.button("Generate PDF Report"):
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(0, 10, txt="Jamie D Flight Log Summary", ln=True, align="C")
        pdf.ln(5)

        pdf.set_font("Arial", size=10)
        # Errors
        pdf.cell(0, 8, txt="Error Messages", ln=True)
        if parsed["errors"]:
            for err in parsed["errors"]:
                pdf.multi_cell(0, 6, txt=err)
        else:
            pdf.multi_cell(0, 6, txt="(none)")
        pdf.ln(3)

        # Events
        pdf.cell(0, 8, txt="System Events", ln=True)
        if parsed["events"]:
            for ev in parsed["events"]:
                pdf.multi_cell(0, 6, txt=ev)
        else:
            pdf.multi_cell(0, 6, txt="(none)")

        # Emit as bytes (FPDF 1.x safe)
        pdf_bytes = pdf.output(dest="S").encode("latin-1")
        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_bytes,
            file_name="flight_log_summary.pdf",
            mime="application/pdf",
        )

# Footer tip
st.caption("Refactor notes: uses cached parsing, safer plotting, battery pack detection (SNum), and fixed PDF export.")


         

           
