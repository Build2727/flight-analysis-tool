import streamlit as st
import tempfile
import os
from pymavlink import mavutil
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF
from io import BytesIO

# Subsystem and error code mappings
ERR_SUBSYS_CODES = {
    0: "Main",
    1: "Radio",
    2: "Compass",
    3: "Optical Flow",
    4: "GPS",
    5: "Battery",
    6: "Flight Mode",
    8: "EKF"
}

ERR_ERROR_CODES = {
    0: "Unspecified",
    1: "Inconsistent",
    2: "Missing",
    3: "Too Large",
    4: "Too Small",
    5: "Timeout"
}

EV_ID_MAP = {
    11: "Landing complete",
    15: "Arm",
    17: "Disarm",
    18: "Failsafe",
    28: "Flight mode change",
    56: "Auto mission started",
    57: "Auto mission complete",
    62: "EKF failsafe triggered"
}

st.set_page_config(page_title="Jamie D Flight Analysis Tool", layout="wide")
st.title("Jamie D Flight Analysis Tool")
st.markdown("""
Upload your ArduPilot `.BIN` log file for analysis. This tool will help you assess flight health and detect potential issues.
""")

uploaded_file = st.file_uploader("Upload a .BIN file", type=["bin"])

if uploaded_file:
    st.success("File uploaded successfully!")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_filename = temp_file.name

    try:
        mlog = mavutil.mavlink_connection(temp_filename)

        error_msgs = []
        event_msgs = []
        altitude_data = []
        battery_data_1 = []
        battery_data_2 = []
        attitude_data = []
        esc_data = []
        vibe_data = []
        mode_data = []
        rc_data = []
        rcout_data = []

        bat_row = 0

        while True:
            msg = mlog.recv_match()
            if msg is None:
                break
            msg_type = msg.get_type()

            timeus = getattr(msg, "TimeUS", -1)
            time_s = round(timeus / 1e6, 2)

            if msg_type == "ERR":
                subsys = getattr(msg, "Subsys", -1)
                ecode = getattr(msg, "ECode", -1)
                subsys_str = ERR_SUBSYS_CODES.get(subsys, f"Unknown Subsys {subsys}")
                ecode_str = ERR_ERROR_CODES.get(ecode, f"Unknown ECode {ecode}")
                decoded = f"ERR at {time_s}s: {subsys_str} - {ecode_str}"
                error_msgs.append(decoded)

            elif msg_type == "EV":
                eid = getattr(msg, "Id", -1)
                event_str = EV_ID_MAP.get(eid, f"Unknown Event ID {eid}")
                decoded = f"EV at {time_s}s: {event_str}"
                event_msgs.append(decoded)
                if eid == 28:
                    mode_data.append((time_s, event_str))

            elif msg_type == "GPS":
                alt = getattr(msg, "Alt", None)
                if alt is not None:
                    altitude_data.append({"Time (s)": time_s, "Alt": alt})

            elif msg_type == "BAT":
                entry = {
                    "Time (s)": time_s,
                    "Volt": getattr(msg, "Volt", None),
                    "VoltR": getattr(msg, "VoltR", None),
                    "Curr": getattr(msg, "Curr", None),
                    "CurrTot": getattr(msg, "CurrTot", None),
                    "Temp": getattr(msg, "Temp", None),
                    "RemPct": getattr(msg, "RemPct", None),
                    "SNum": getattr(msg, "SNum", None)
                }
                if bat_row % 2 == 0:
                    battery_data_1.append(entry)
                else:
                    battery_data_2.append(entry)
                bat_row += 1

            elif msg_type == "ATT":
                desroll = getattr(msg, "DesRoll", None)
                roll = getattr(msg, "Roll", None)
                attitude_data.append({
                    "Time (s)": time_s,
                    "DesRoll": desroll,
                    "Roll": roll
                })

            elif msg_type == "ESC":
                temp = getattr(msg, "Temp", None)
                esc_data.append({
                    "Time (s)": time_s,
                    "Temp": temp
                })

            elif msg_type == "VIBE":
                vibe_data.append({
                    "Time (s)": time_s,
                    "VibeX": getattr(msg, "VibeX", None),
                    "VibeY": getattr(msg, "VibeY", None),
                    "VibeZ": getattr(msg, "VibeZ", None)
                })

            elif msg_type == "RCIN":
                rc_data.append({
                    "Time (s)": time_s,
                    "C1": getattr(msg, "C1", None),
                    "C2": getattr(msg, "C2", None),
                    "C3": getattr(msg, "C3", None),
                    "C4": getattr(msg, "C4", None)
                })

            elif msg_type == "RCOU":
                rcout_data.append({
                    "Time (s)": time_s,
                    "C9": getattr(msg, "C9", None),
                    "C10": getattr(msg, "C10", None),
                    "C11": getattr(msg, "C11", None),
                    "C12": getattr(msg, "C12", None)
                })

        if error_msgs:
            st.markdown("### ❌ Error Messages")
            for err in error_msgs:
                st.code(err)

        if event_msgs:
            st.markdown("### ⚠️ System Events")
            for ev in event_msgs:
                st.code(ev)

        def plot_chart(data, title, ylabel, keys):
            df = pd.DataFrame(data)
            if df.empty:
                st.info(f"No data available for {title}")
                return None, None
            fig, ax = plt.subplots(figsize=(10, 4))
            for key in keys:
                ax.plot(df["Time (s)"], df[key], label=key, linewidth=0.7)
            ax.set_title(title)
            ax.set_xlabel("Flight Duration (s)")
            ax.set_ylabel(ylabel)
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
            st.download_button(
                label=f"📥 Download {title} Data as CSV",
                data=df.to_csv(index=False).encode('utf-8'),
                file_name=f"{title.replace(' ', '_').lower()}.csv",
                mime="text/csv"
            )
            return title, df

        # Collect all chart data
        charts = [
            (battery_data_1, "Battery 1 Metrics", "Value", ["Volt", "Curr", "Temp"]),
            (battery_data_2, "Battery 2 Metrics", "Value", ["Volt", "Curr", "Temp"]),
            (esc_data, "ESC Temperature Over Time", "Temp (°C)", ["Temp"]),
            (altitude_data, "GPS Altitude Over Time", "Altitude (cm)", ["Alt"]),
            (vibe_data, "Vibration Metrics", "Vibe", ["VibeX", "VibeY", "VibeZ"]),
            (attitude_data, "Desired vs Actual Roll", "Degrees", ["DesRoll", "Roll"]),
            (rc_data, "RC Input (C1-C4)", "PWM", ["C1", "C2", "C3", "C4"]),
            (rcout_data, "Motor Output (C9-C12)", "PWM", ["C9", "C10", "C11", "C12"])
        ]

        all_chart_data = []
        for data, title, ylabel, keys in charts:
            result = plot_chart(data, title, ylabel, keys)
            if result[0]:
                all_chart_data.append(result)

        if mode_data:
            st.markdown("### 🧭 Flight Modes Timeline")
            for t, mode in mode_data:
                st.write(f"{t}s: {mode}")

        # PDF Export
        if st.button("📄 Export Summary as PDF"):
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.add_page()
            pdf.set_font("Arial", size=12)

            pdf.cell(200, 10, txt="Jamie D Flight Log Summary", ln=True, align="C")
            pdf.ln(10)

            pdf.set_font("Arial", size=10)
            pdf.cell(200, 10, txt="Error Messages", ln=True)
            for err in error_msgs:
                pdf.multi_cell(0, 8, txt=err)

            pdf.ln(5)
            pdf.cell(200, 10, txt="System Events", ln=True)
            for ev in event_msgs:
                pdf.multi_cell(0, 8, txt=ev)

            pdf_output = BytesIO()
            pdf.output(pdf_output)
            st.download_button(
                label="📄 Download PDF Report",
                data=pdf_output.getvalue(),
                file_name="flight_log_summary.pdf",
                mime="application/pdf"
            )

    except Exception as e:
        st.error(f"Error processing log file: {e}")

    finally:
        if os.path.exists(temp_filename):
            try:
                os.remove(temp_filename)
            except Exception:
                pass

         

           
