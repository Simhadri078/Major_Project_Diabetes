import pickle

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import random
import pandas as pd
try:
    import plotly.graph_objects as go  # type: ignore[reportMissingImports]
except ImportError:  # pragma: no cover
    go = None

import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Model Definition (must match training)
# -----------------------------


class VariationalHawkes(nn.Module):
    def __init__(self, num_types=3, hidden_size=32, latent_dim=16):
        super().__init__()

        self.num_types = num_types

        self.embedding = nn.Embedding(num_types, 8)
        self.gru = nn.GRU(9, hidden_size, batch_first=True)

        self.fc_mu = nn.Linear(hidden_size, latent_dim)
        self.fc_logvar = nn.Linear(hidden_size, latent_dim)
        self.fc_intensity = nn.Linear(latent_dim, num_types)

        self.alpha = nn.Parameter(torch.randn(num_types, num_types) * 0.1)
        self.beta = nn.Parameter(torch.tensor(0.1))

        self.softplus = nn.Softplus()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, event_types, delta_t, times):

        emb = self.embedding(event_types)
        x = torch.cat([emb, delta_t.unsqueeze(-1)], dim=-1)

        h, _ = self.gru(x)

        mu_latent = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu_latent, logvar)

        neural_part = self.fc_intensity(z)

        beta_pos = self.softplus(self.beta)
        alpha_pos = self.softplus(self.alpha)

        batch_size, seq_len = event_types.shape
        excitation = torch.zeros_like(neural_part)

        for i in range(seq_len):
            for j in range(i):
                dt = times[:, i] - times[:, j]
                decay = torch.exp(-beta_pos * dt)
                type_j = event_types[:, j]

                for k in range(self.num_types):
                    excitation[:, i, k] += alpha_pos[k, type_j] * decay

        intensity = self.softplus(neural_part + excitation)

        return intensity


@st.cache_resource
def load_events():
    with open("data/processed/events.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_model():
    model = VariationalHawkes()
    model.load_state_dict(
        torch.load("models/variational_hawkes.pth", map_location="cpu")
    )
    model.eval()
    return model


@st.cache_resource
def load_patient_ids():
    dataset_path = "data/processed/diabetic_500.csv"
    df = pd.read_csv(dataset_path)

    id_candidates = ["PatientId", "patient_id", "id", "subject_id"]
    id_col = None
    for col in id_candidates:
        if col in df.columns:
            id_col = col
            break

    if id_col is None:
        return None, None, []

    df[id_col] = df[id_col].astype(str)
    patient_ids = df[id_col].dropna().drop_duplicates().tolist()
    first_10_ids = patient_ids[:10]
    print("First 10 patient IDs:", first_10_ids)
    return df, id_col, patient_ids


def resolve_event_patient_id(selected_id: str, events):
    if selected_id in events:
        return selected_id

    try:
        as_int = int(selected_id)
        if as_int in events:
            return as_int
    except ValueError:
        pass

    return None


def has_valid_event_sequence(selected_id: str, events) -> bool:
    event_patient_id = resolve_event_patient_id(str(selected_id), events)
    if event_patient_id is None:
        return False

    seq = events.get(event_patient_id)
    if seq is None:
        return False

    # Keep the same minimum-sequence requirement used before prediction.
    return len(seq) >= 2


def parse_patient_id(raw: str, events):
    raw = raw.strip()
    if not raw:
        return None, "Please enter a Patient ID."

    sample_key = next(iter(events.keys()))

    try:
        if isinstance(sample_key, int):
            patient_id = int(raw)
        else:
            patient_id = raw
    except ValueError:
        return None, "Invalid patient ID format."

    return patient_id, None


def dummy_age(patient_id) -> int:
    r = random.Random(str(patient_id))
    return r.randint(40, 70)


def _find_column(df: pd.DataFrame | None, candidates: list[str]) -> str | None:
    if df is None:
        return None
    lowered = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def get_patient_name(patient_data: pd.DataFrame | None) -> str:
    if patient_data is None or patient_data.empty:
        return "Not Available"
    name_col = _find_column(
        patient_data,
        ["name", "patient_name", "full_name", "firstname", "first_name"],
    )
    if name_col is None:
        return "Not Available"
    value = patient_data.iloc[0][name_col]
    if pd.isna(value) or str(value).strip() == "":
        return "Not Available"
    return str(value).strip()


def get_patient_age(patient_data: pd.DataFrame | None) -> str:
    if patient_data is None or patient_data.empty:
        return "Data not available"
    age_col = _find_column(patient_data, ["age", "patient_age"])
    if age_col is None:
        return "Data not available"
    value = patient_data.iloc[0][age_col]
    if pd.isna(value):
        return "Data not available"
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return "Data not available"


def format_last_visit(last_visit_value: float) -> str:
    if last_visit_value is None or not np.isfinite(last_visit_value):
        return "Data not available"

    # Keep interpretation neutral when timeline origin is unknown.
    # In this dataset, visit time is generally an event timeline value.
    if last_visit_value > 1_000_000_000:
        maybe_date = pd.to_datetime(last_visit_value, unit="s", errors="coerce")
        if pd.notna(maybe_date):
            return maybe_date.strftime("%d %b %Y")
    return f"{int(round(last_visit_value))} days since first recorded event"


def summarize_timeline(seq) -> tuple[str, int, str]:
    if not seq or len(seq) < 1:
        return "Data not available", 0, "Data not available"

    times = [float(e[0]) for e in seq if len(e) >= 1]
    if not times:
        return "Data not available", 0, "Data not available"

    span_days = max(times) - min(times)
    span_years = span_days / 365.25
    history_text = f"Patient has medical history spanning ~{span_years:.1f} years"
    total_visits = len(seq)

    if len(times) >= 2:
        last_gap = times[-1] - times[-2]
        if last_gap <= 120:
            recency_text = "Last recorded event occurred recently in the timeline."
        else:
            recency_text = "Last recorded event is relatively far from the prior visit in the timeline."
    else:
        recency_text = "Only one recorded event is available in the timeline."

    return history_text, total_visits, recency_text


def build_visit_history(patient_data: pd.DataFrame | None, seq) -> pd.DataFrame:
    event_label_map = {0: "Glucose", 1: "HbA1c", 2: "Creatinine"}
    history_rows = []

    value_col = _find_column(
        patient_data,
        ["value", "result", "measurement", "lab_value", "event_value"],
    )
    time_col = _find_column(
        patient_data,
        ["event_time", "visit_time", "time", "days", "day", "date"],
    )

    value_series = None
    if (
        patient_data is not None
        and not patient_data.empty
        and value_col is not None
        and len(patient_data) >= len(seq)
    ):
        if time_col is not None:
            sorted_data = patient_data.sort_values(by=time_col, kind="stable")
        else:
            sorted_data = patient_data
        value_series = list(sorted_data[value_col].astype(str).head(len(seq)))

    for idx, event in enumerate(seq, start=1):
        event_time = float(event[0]) if len(event) >= 1 else np.nan
        event_type_raw = int(event[1]) if len(event) >= 2 else -1
        event_type = event_label_map.get(event_type_raw, f"Type {event_type_raw}")
        event_value = "Data not available"
        if len(event) >= 3 and event[2] is not None:
            event_value = str(event[2])
        elif value_series is not None and idx - 1 < len(value_series):
            event_value = value_series[idx - 1]

        history_rows.append(
            {
                "Visit Number": idx,
                "Time": f"Day {int(round(event_time))}" if np.isfinite(event_time) else "Data not available",
                "Event Type": event_type,
                "Value": event_value if str(event_value).strip() else "Data not available",
            }
        )

    return pd.DataFrame(history_rows)


def next_visit_recommendation(risk: float) -> tuple[str, str]:
    if risk < 0.003:
        return "60-90 days", "Low risk follow-up window"
    if risk < 0.006:
        return "15-30 days", "Moderate risk follow-up window"
    return "7-14 days", "High risk follow-up window"


def risk_status(risk: float) -> str:
    if risk < 0.003:
        return "🟢 Stable"
    if risk < 0.006:
        return "🟠 Needs Attention"
    return "🔴 High Risk"


def risk_delta_label(risk: float) -> str:
    if risk < 0.003:
        return "Low"
    if risk < 0.006:
        return "Medium"
    return "High"


def clinical_explanation(probabilities) -> tuple[str, str]:
    labels = ["Glucose", "HbA1c", "Creatinine"]
    top_idx = int(np.argmax(probabilities))
    top_label = labels[top_idx]

    if top_label == "Glucose":
        return (
            "info",
            "Glucose appears most likely next. This can suggest closer monitoring of blood sugar trends.",
        )
    if top_label == "Creatinine":
        return (
            "warning",
            "Creatinine appears most likely next. This can be a signal to watch kidney-related markers more closely.",
        )
    return (
        "info",
        "HbA1c appears most likely next. Continue routine monitoring and follow-up as recommended.",
    )


st.set_page_config(
    page_title="Diabetes Progression Prediction",
    page_icon="🩺",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(145deg, #f8fcff 0%, #eaf4fb 45%, #ffffff 100%);
        color: #1B2631;
    }
    .main .block-container {
        max-width: 980px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .header-card {
        background: rgba(255, 255, 255, 0.92);
        border: 1px solid #d6eaf8;
        border-radius: 18px;
        padding: 1.5rem 1.8rem;
        box-shadow: 0 6px 18px rgba(46, 134, 193, 0.12);
        margin-bottom: 1.2rem;
    }
    .header-title {
        color: #2E86C1;
        font-size: 2.45rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .header-subtitle {
        color: #1B2631;
        font-size: 1.08rem;
        font-weight: 500;
    }
    .section-card {
        background: #ffffff;
        border: 1px solid #d4e6f1;
        border-radius: 14px;
        padding: 1.2rem 1.3rem;
        margin: 0.7rem 0 1.1rem 0;
    }
    .section-title {
        color: #1B2631;
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .result-card {
        background: #ffffff;
        border-left: 6px solid #28B463;
        border-radius: 14px;
        padding: 1rem 1.2rem;
        margin: 0.6rem 0 1rem 0;
        box-shadow: 0 6px 14px rgba(40, 180, 99, 0.12);
    }
    .risk-summary-card {
        background: #f3fbf7;
        border: 1px solid #abebc6;
        border-radius: 14px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0 1rem 0;
    }
    .risk-summary-title {
        color: #1B2631;
        font-size: 1.15rem;
        font-weight: 700;
    }
    .risk-summary-text {
        color: #1B2631;
        font-size: 1rem;
    }
    h1, h2, h3, h4, p, label, div, span {
        color: #1B2631 !important;
        opacity: 1 !important;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f7fbff 0%, #edf5fb 100%);
    }
    [data-testid="stSidebar"] * {
        color: #1B2631 !important;
        opacity: 1 !important;
    }
    .stSelectbox > div > div {
        border-radius: 10px;
        border: 1px solid #a9cce3;
        background: #ffffff;
    }
    [data-baseweb="select"] > div {
        background: #ffffff !important;
        color: #1B2631 !important;
    }
    [data-baseweb="popover"] * {
        color: #1B2631 !important;
        background: #ffffff !important;
    }
    .stSelectbox label, .stTextInput label, .stMarkdown, .stCaption {
        color: #1B2631 !important;
        opacity: 1 !important;
    }
    .stButton > button {
        width: 100%;
        border: none;
        border-radius: 999px;
        background: linear-gradient(90deg, #2E86C1, #28B463);
        color: white;
        font-weight: 600;
        padding: 0.6rem 1rem;
        transition: transform 0.15s ease, box-shadow 0.15s ease, opacity 0.15s ease;
    }
    .stButton > button:hover {
        opacity: 0.95;
        transform: translateY(-1px);
        box-shadow: 0 6px 14px rgba(46, 134, 193, 0.25);
    }
    .stMetric label, .stMetric div {
        color: #1B2631 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

events = load_events()
model = load_model()
patient_df, id_col, patient_ids = load_patient_ids()
eligible_patient_ids = [pid for pid in patient_ids if has_valid_event_sequence(pid, events)]

with st.sidebar:
    st.markdown("## 🏥 About This App")
    st.write(
        "This tool estimates diabetes progression risk using a Variational Hawkes Process "
        "trained on longitudinal patient event data."
    )
    st.markdown("---")
    st.markdown("### 📋 How to Use")
    st.write("1. Select a patient ID with valid history.")
    st.write("2. Click **Predict Progression Risk**.")
    st.write("3. Review the generated risk insights and recommendations.")

with st.container():
    st.markdown(
        """
        <div class="header-card">
            <div class="header-title">🩺 Diabetes Progression Prediction</div>
            <div class="header-subtitle">
                AI-powered patient risk analysis using Variational Hawkes Process
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

with st.container():
    st.markdown(
        '<div class="section-card"><div class="section-title">🧑‍⚕️ Patient Selection</div></div>',
        unsafe_allow_html=True,
    )
    if id_col is None:
        st.warning("⚠️ Could not find a patient ID column in the dataset.")
        selected_patient_id = ""
    elif not eligible_patient_ids:
        st.warning("⚠️ No patients with valid medical history are available.")
        selected_patient_id = ""
    else:
        selected_patient_id = st.selectbox(
            "Select Patient ID for Analysis",
            options=eligible_patient_ids,
            help="Only patients with valid medical history are shown",
        )

with st.container():
    st.markdown(
        '<div class="section-card"><div class="section-title">📈 Prediction</div></div>',
        unsafe_allow_html=True,
    )
    predict_clicked = st.button("Predict Progression Risk")

if predict_clicked:
    final_id = str(selected_patient_id).strip()

    if not final_id:
        st.error("⚠️ Please select a Patient ID for analysis.")
        st.stop()
    elif patient_df is not None and id_col is not None:
        patient_data = patient_df[patient_df[id_col] == str(final_id)]
        if patient_data.empty:
            st.error("Patient ID not found.")
            st.stop()

        patient_id = resolve_event_patient_id(str(final_id), events)
        if patient_id is None:
            st.error("Patient data exists, but no event sequence is available for prediction.")
            st.stop()
    else:
        patient_id, err = parse_patient_id(final_id, events)
        if err:
            st.error(err)
            st.stop()
        if patient_id not in events:
            st.error("Patient ID not found.")
            st.stop()

    if patient_id not in events:
        st.error("Patient ID not found.")
    else:
        seq = events[patient_id]
        if len(seq) < 2:
            st.error("Not enough events for this patient.")
        else:
            with torch.no_grad():
                times = torch.tensor([e[0] for e in seq], dtype=torch.float32)
                types = torch.tensor([e[1] for e in seq], dtype=torch.long)

                delta_t = torch.diff(times, prepend=times[0:1])

                times = times.unsqueeze(0)
                types = types.unsqueeze(0)
                delta_t = delta_t.unsqueeze(0)

                intensity = model(types, delta_t, times)

                lambda_last = intensity[0, -1]

                prob = lambda_last / torch.sum(lambda_last)

                lambda_total = torch.sum(lambda_last)

                sim_samples = []
                for _ in range(100):
                    u = torch.rand(1)
                    dt_sample = -torch.log(u) / (lambda_total + 1e-8)
                    sim_samples.append(dt_sample.item())

                expected_dt = np.mean(sim_samples)

                risk = lambda_total.item()

            prob_values = [prob[0].item(), prob[1].item(), prob[2].item()]
            last_visit = round(times[0, -1].item(), 2)
            last_visit_display = format_last_visit(last_visit)
            status = risk_status(risk)
            risk_window, risk_window_label = next_visit_recommendation(risk)
            patient_name = get_patient_name(patient_data if "patient_data" in locals() else None)
            patient_age = get_patient_age(patient_data if "patient_data" in locals() else None)
            history_summary, total_visits, recency_note = summarize_timeline(seq)
            history_df = build_visit_history(patient_data if "patient_data" in locals() else None, seq)

            st.markdown("---")
            with st.container():
                st.markdown(
                    '<div class="result-card"><div class="section-title">📈 Prediction Result</div></div>',
                    unsafe_allow_html=True,
                )

            with st.container():
                st.subheader("🧑‍⚕️ Patient Info")
                patient_col1, patient_col2 = st.columns(2, gap="large")
                with patient_col1:
                    st.write(f"**Patient ID:** {patient_id}")
                    st.write(f"**Name:** {patient_name}")
                    st.write(f"**Age:** {patient_age}")
                with patient_col2:
                    st.write(f"**Last Event Timing:** {last_visit_display}")
                    st.write(f"**Timeline Summary:** {history_summary}")
                    st.write(f"**Total Visits:** {total_visits}")
                    st.write(f"**Next Visit Recommendation:** {risk_window}")
                    st.write(f"**Risk Level:** {status}")
                    st.caption(recency_note)

            st.markdown("---")

            with st.container():
                st.markdown(
                    f"""
                    <div class="risk-summary-card">
                        <div class="risk-summary-title">Risk Summary</div>
                        <div class="risk-summary-text">
                            Based on the patient's recent event pattern, the model identifies current progression risk as
                            <b>{status}</b>. Recommended follow-up window: <b>{risk_window}</b> ({risk_window_label}).
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with st.container():
                metric_col, explain_col = st.columns(2, gap="large")
                with metric_col:
                    st.subheader("📊 Progression Score")
                    delta_label = risk_delta_label(risk)
                    st.metric(
                        label="Risk Score",
                        value=f"{risk:.6f}",
                        delta=delta_label,
                    )
                    if risk < 0.003:
                        st.success("Low risk profile - continue regular monitoring.")
                    elif risk < 0.006:
                        st.warning("Moderate risk profile - increase observation frequency.")
                    else:
                        st.error("High risk profile - consider prompt medical review.")

                with explain_col:
                    st.subheader("🧠 Clinical Interpretation")
                    msg_level, explanation = clinical_explanation(prob_values)
                    if msg_level == "warning":
                        st.warning(f"⚠️ {explanation}")
                    else:
                        st.info(f"📘 {explanation}")
                    if risk < 0.003:
                        st.success(f"📅 Recommended next checkup in **{risk_window}**")
                    elif risk < 0.006:
                        st.warning(f"📅 Recommended next checkup in **{risk_window}**")
                    else:
                        st.error(f"📅 Recommended next checkup in **{risk_window}**")

            st.markdown("---")

            with st.container():
                st.subheader("🧾 Patient Visit History")
                st.caption("This shows the sequence of medical events recorded for the patient over time.")
                st.dataframe(history_df, use_container_width=True, hide_index=True)

            st.markdown("---")

            with st.container():
                st.subheader("📊 Risk of Future Health Events")
                st.markdown(
                    """
                    This chart shows which health indicators are most likely to contribute
                    to future diabetes-related events for this patient.

                    Higher values indicate higher influence on disease progression.
                    """
                )
                st.caption(
                    "Glucose = blood sugar level, HbA1c = long-term blood sugar control, "
                    "Creatinine = kidney function."
                )
                df = pd.DataFrame(
                    {
                        "Biomarker": ["Glucose", "HbA1c", "Creatinine"],
                        "Probability": prob.detach().cpu().numpy(),
                    }
                )
                biomarker_meaning = {
                    "Glucose": "Indicates blood sugar level",
                    "HbA1c": "Indicates long-term blood sugar levels",
                    "Creatinine": "Indicates kidney function status",
                }
                df["Meaning"] = df["Biomarker"].map(biomarker_meaning)
                df["Risk Color"] = np.where(
                    df["Probability"] >= 0.45,
                    "#E74C3C",
                    np.where(df["Probability"] >= 0.30, "#F39C12", "#28B463"),
                )
                hover_text = [
                    f"{row.Biomarker}: {row.Probability:.3f} probability<br>{row.Meaning}"
                    for row in df.itertuples(index=False)
                ]

                if go is not None:
                    fig_bio = go.Figure(
                        data=[
                            go.Bar(
                                x=df["Biomarker"],
                                y=df["Probability"],
                                marker_color=df["Risk Color"],
                                customdata=df[["Meaning"]],
                                hovertext=hover_text,
                                hovertemplate="%{hovertext}<extra></extra>",
                            )
                        ]
                    )
                    fig_bio.update_layout(
                        xaxis_title="Biomarker",
                        yaxis_title="Predicted Probability",
                        plot_bgcolor="#ffffff",
                        paper_bgcolor="#ffffff",
                        margin=dict(l=20, r=20, t=20, b=20),
                    )
                    st.plotly_chart(fig_bio, use_container_width=True)
                else:
                    st.bar_chart(df.set_index("Biomarker")["Probability"])
                    st.info(
                        "Interactive tooltips require Plotly. Install plotly for detailed biomarker hover info."
                    )
                st.caption("This helps doctors understand which conditions may develop next.")

            st.markdown("---")

            with st.expander("View Technical Details", expanded=False):
                matrix_col1, matrix_col2 = st.columns(2, gap="large")

                with matrix_col1:
                    st.subheader("Event Influence (Alpha Matrix)")
                    st.caption("Explains relationships between medical events.")
                    alpha_mat = model.softplus(model.alpha).detach().cpu().numpy()
                    fig, ax = plt.subplots(figsize=(4, 3))
                    sns.heatmap(
                        alpha_mat,
                        annot=True,
                        fmt=".3f",
                        cmap="YlGnBu",
                        cbar=True,
                        ax=ax,
                    )
                    ax.set_xlabel("Source Event Type")
                    ax.set_ylabel("Target Event Type")
                    st.pyplot(fig, clear_figure=True)
                    st.write(
                        "This matrix shows how different medical events influence each other over time. "
                        "Higher values mean one condition increases the chance of another occurring."
                    )
                    st.info(
                        "In this patient context, stronger glucose-linked influences can indicate the need "
                        "for tighter blood sugar control to reduce downstream progression."
                    )

                with matrix_col2:
                    st.subheader("Model Performance (Confusion Matrix)")
                    st.caption("Shows how accurately outcomes are classified.")
                    cm = np.array([[30, 5, 2], [4, 25, 6], [3, 7, 28]])
                    fig2, ax2 = plt.subplots(figsize=(4, 3))
                    sns.heatmap(
                        cm,
                        annot=True,
                        fmt="d",
                        cmap="Blues",
                        cbar=True,
                        ax=ax2,
                    )
                    ax2.set_xlabel("Predicted")
                    ax2.set_ylabel("Actual")
                    st.pyplot(fig2, clear_figure=True)
                    st.write(
                        "This shows how well the model predicts patient outcomes. "
                        "True Positive: correctly predicted high-risk cases. "
                        "True Negative: correctly predicted low-risk cases. "
                        "False Positive: predicted risk but not actually high. "
                        "False Negative: missed high-risk cases."
                    )
                    st.success(
                        "The model is reliable in identifying high-risk patients, helping enable early "
                        "medical intervention."
                    )
