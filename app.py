# description: Qubit Lab, an interactive quantum computing playground. Build circuits,
#              watch amplitudes and Bloch spheres change, simulate measurements,
#              and see Grover's algorithm find a hidden item faster than guessing.
# author: Nick Faiella (built with help from Claude)
# date: October 14, 2026
#
# Run locally with:  streamlit run app.py

import math

import altair as alt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import quantum
from quantum import QuantumCircuit

st.set_page_config(page_title="Qubit Lab", page_icon="⚛️", layout="wide")

INK = "#1C1F3A"
MUTED = "#6B6F8E"
WIRE = "#9AA0C0"
ACCENT = "#6A3FCF"

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600;700&family=STIX+Two+Text:ital@0;1&display=swap');
    html, body, [class*="css"], .stMarkdown { font-family: 'IBM Plex Sans', system-ui, sans-serif; }
    .ket {
        font-family: 'STIX Two Text', 'Cambria Math', Georgia, serif;
        font-size: 1.45rem; line-height: 1.5; padding: 0.6rem 0 0.2rem 0;
        overflow-wrap: anywhere;
    }
    .circuit { overflow-x: auto; padding: 0.4rem 0 0.8rem 0; }
    .lede { font-size: 1.1rem; color: #4A4E6B; max-width: 46em; margin-top: -0.6rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- drawing helpers ----------

def circuit_svg(n_qubits, steps):
    """Draw the circuit as an SVG picture: one wire per qubit, one column per gate."""
    col_width, row_height, left = 58, 52, 52
    width = left + max(len(steps), 4) * col_width + 30
    height = n_qubits * row_height + 14
    y = lambda q: 30 + q * row_height
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
             f'font-family="IBM Plex Sans, sans-serif">']
    for q in range(n_qubits):
        parts.append(f'<text x="8" y="{y(q) + 5}" font-size="14" fill="{MUTED}">q{q}</text>')
        parts.append(f'<line x1="{left - 12}" y1="{y(q)}" x2="{width - 10}" y2="{y(q)}" '
                     f'stroke="{WIRE}" stroke-width="1.5"/>')
    for i, step in enumerate(steps):
        x = left + i * col_width + col_width / 2
        name, t, c = step["gate"], step["target"], step["control"]
        if name in quantum.TWO_QUBIT_GATES:
            parts.append(f'<line x1="{x}" y1="{y(c)}" x2="{x}" y2="{y(t)}" stroke="{INK}" stroke-width="2"/>')
            if name == "CNOT":
                parts.append(f'<circle cx="{x}" cy="{y(c)}" r="5" fill="{INK}"/>')
                parts.append(f'<circle cx="{x}" cy="{y(t)}" r="12" fill="white" stroke="{INK}" stroke-width="2"/>')
                parts.append(f'<line x1="{x - 12}" y1="{y(t)}" x2="{x + 12}" y2="{y(t)}" stroke="{INK}" stroke-width="2"/>')
                parts.append(f'<line x1="{x}" y1="{y(t) - 12}" x2="{x}" y2="{y(t) + 12}" stroke="{INK}" stroke-width="2"/>')
            elif name == "CZ":
                parts.append(f'<circle cx="{x}" cy="{y(c)}" r="5" fill="{INK}"/>')
                parts.append(f'<circle cx="{x}" cy="{y(t)}" r="5" fill="{INK}"/>')
            else:  # SWAP draws an X on both wires
                for q in (c, t):
                    parts.append(f'<path d="M{x - 7} {y(q) - 7} L{x + 7} {y(q) + 7} M{x + 7} {y(q) - 7} '
                                 f'L{x - 7} {y(q) + 7}" stroke="{INK}" stroke-width="2"/>')
        else:
            label = name
            parts.append(f'<rect x="{x - 19}" y="{y(t) - 18}" width="38" height="36" rx="4" '
                         f'fill="{ACCENT}" />')
            size = 15 if len(label) == 1 else 12
            parts.append(f'<text x="{x}" y="{y(t) + 5}" text-anchor="middle" font-size="{size}" '
                         f'font-weight="600" fill="white">{label}</text>')
            if step["angle"] is not None:
                degrees = round(math.degrees(step["angle"]))
                parts.append(f'<text x="{x}" y="{y(t) + 31}" text-anchor="middle" font-size="10" '
                             f'fill="{MUTED}">{degrees}°</text>')
    parts.append("</svg>")
    return "".join(parts)


def amplitude_chart(qc):
    """Bar height = probability. Bar color = phase (the angle of the complex amplitude)."""
    df = pd.DataFrame({
        "State": [f"|{label}⟩" for label in qc.labels()],
        "Probability": qc.probabilities(),
        "Phase": np.degrees(np.angle(qc.state)),
        "Amplitude": [f"{a.real:+.3f} {a.imag:+.3f}i" for a in qc.state],
    })
    df.loc[df["Probability"] < 1e-12, "Phase"] = 0
    return alt.Chart(df).mark_bar().encode(
        x=alt.X("State:N", sort=None, title=None, axis=alt.Axis(labelAngle=0, labelFontSize=13)),
        y=alt.Y("Probability:Q", scale=alt.Scale(domain=[0, 1]), title="Probability"),
        color=alt.Color("Phase:Q", scale=alt.Scale(scheme="sinebow", domain=[-180, 180]),
                        legend=alt.Legend(title="Phase (°)", values=[-180, -90, 0, 90, 180])),
        tooltip=["State", alt.Tooltip("Probability:Q", format=".3f"), "Amplitude",
                 alt.Tooltip("Phase:Q", format=".0f")],
    ).properties(height=280)


def bloch_figure(vector, title):
    """A 3D Bloch sphere with an arrow showing one qubit's state."""
    fig = go.Figure()
    t = np.linspace(0, 2 * math.pi, 80)
    for xs, ys, zs in [(np.cos(t), np.sin(t), 0 * t), (np.cos(t), 0 * t, np.sin(t)), (0 * t, np.cos(t), np.sin(t))]:
        fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines",
                                   line=dict(color="#C9CDE3", width=2), hoverinfo="skip"))
    for (a, b), label in [(((0, 0, 1.25), (0, 0, -1.25)), ("|0⟩", "|1⟩"))]:
        fig.add_trace(go.Scatter3d(x=[a[0], b[0]], y=[a[1], b[1]], z=[a[2], b[2]], mode="text",
                                   text=list(label), textfont=dict(size=14, color=INK), hoverinfo="skip"))
    x, y, z = vector
    length = math.sqrt(x * x + y * y + z * z)
    fig.add_trace(go.Scatter3d(x=[0, x], y=[0, y], z=[0, z], mode="lines",
                               line=dict(color=ACCENT, width=7), hoverinfo="skip"))
    if length > 0.05:
        fig.add_trace(go.Cone(x=[x], y=[y], z=[z], u=[x], v=[y], w=[z], sizemode="absolute",
                              sizeref=0.25, anchor="tip", showscale=False,
                              colorscale=[[0, ACCENT], [1, ACCENT]], hoverinfo="skip"))
    else:
        fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode="markers",
                                   marker=dict(size=6, color=ACCENT), hoverinfo="skip"))
    axis = dict(visible=False, range=[-1.3, 1.3])
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=14, color=INK)),
        showlegend=False, height=290, margin=dict(l=0, r=0, t=34, b=0),
        scene=dict(xaxis=axis, yaxis=axis, zaxis=axis, aspectmode="cube",
                   camera=dict(eye=dict(x=1.25, y=1.0, z=0.55))),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ---------- app state ----------
if "steps" not in st.session_state:
    st.session_state.steps = []
if "n_qubits" not in st.session_state:
    st.session_state.n_qubits = 2

PRESETS = {
    "Coin flip": (1, [("H", 0, None)]),
    "Bell pair": (2, [("H", 0, None), ("CNOT", 1, 0)]),
    "GHZ state": (3, [("H", 0, None), ("CNOT", 1, 0), ("CNOT", 2, 1)]),
}


def load_preset(name):
    n, gates = PRESETS[name]
    st.session_state.n_qubits = n
    st.session_state.steps = [{"gate": g, "target": t, "control": c, "angle": None} for g, t, c in gates]


# ---------- page ----------
st.title("Qubit Lab")
st.markdown('<p class="lede">A quantum computer simulator built from scratch in Python. '
            'Build a circuit and watch superposition, interference, and entanglement happen.</p>',
            unsafe_allow_html=True)

builder_tab, grover_tab = st.tabs(["Circuit builder", "Grover's search"])

with builder_tab:
    controls, results = st.columns([1, 2], gap="large")

    with controls:
        n = st.slider("Qubits", 1, 4, key="n_qubits")
        # drop gates that use a qubit that no longer exists
        st.session_state.steps = [s for s in st.session_state.steps
                                  if s["target"] < n and (s["control"] is None or s["control"] < n)]

        st.write("Start from an example")
        preset_cols = st.columns(len(PRESETS))
        for col, name in zip(preset_cols, PRESETS):
            col.button(name, on_click=load_preset, args=(name,), use_container_width=True)

        st.divider()
        gate_options = ["H", "X", "Y", "Z", "S", "T", "RX", "RY", "RZ"]
        if n >= 2:
            gate_options += quantum.TWO_QUBIT_GATES
        gate = st.selectbox("Gate", gate_options, help="Pick a gate to add to the end of the circuit.")
        st.caption(quantum.GATE_INFO[gate])

        control = None
        angle = None
        if gate in quantum.TWO_QUBIT_GATES:
            c1, c2 = st.columns(2)
            control = c1.selectbox("Control qubit", range(n), format_func=lambda q: f"q{q}")
            target = c2.selectbox("Target qubit", [q for q in range(n) if q != control],
                                  format_func=lambda q: f"q{q}")
        else:
            target = st.selectbox("Qubit", range(n), format_func=lambda q: f"q{q}")
        if gate in quantum.ROTATION_GATES:
            angle = math.radians(st.slider("Angle (degrees)", -180, 180, 90, step=15))

        add_col, undo_col, clear_col = st.columns(3)
        if add_col.button("Add gate", type="primary", use_container_width=True):
            st.session_state.steps.append({"gate": gate, "target": target, "control": control, "angle": angle})
        if undo_col.button("Undo", use_container_width=True, disabled=not st.session_state.steps):
            st.session_state.steps.pop()
        if clear_col.button("Clear", use_container_width=True, disabled=not st.session_state.steps):
            st.session_state.steps = []

    # build the circuit fresh from the saved list of gates
    qc = QuantumCircuit(n)
    for s in st.session_state.steps:
        qc.gate(s["gate"], s["target"], control=s["control"], angle=s["angle"])

    with results:
        st.markdown(f'<div class="circuit">{circuit_svg(n, qc.steps)}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="ket">|ψ⟩ = {qc.ket_string()}</div>', unsafe_allow_html=True)
        st.altair_chart(amplitude_chart(qc), use_container_width=True)
        st.caption("Bar height is the chance of measuring that result. Color is the phase, which "
                   "doesn't change the odds by itself but decides how paths interfere.")

    st.subheader("Each qubit on the Bloch sphere")
    st.caption("Top is |0⟩, bottom is |1⟩, the equator is an even superposition. An arrow that "
               "shrinks toward the center means the qubit is entangled with another one.")
    sphere_cols = st.columns(n)
    for q in range(n):
        vector = qc.bloch_vector(q)
        sphere_cols[q].plotly_chart(bloch_figure(vector, f"q{q}"), use_container_width=True,
                                    config={"displayModeBar": False}, key=f"bloch_{q}")

    st.subheader("Measure it")
    m1, m2 = st.columns([1, 2], gap="large")
    with m1:
        shots = st.select_slider("Number of runs", options=[10, 100, 1000, 10000], value=1000)
        st.button("Run measurement", key="measure")
        st.caption("Measuring forces each qubit to pick 0 or 1. Run it again and the counts "
                   "change a little, but they always follow the probabilities above.")
    with m2:
        counts = qc.measure(shots=shots)
        count_df = pd.DataFrame({"Result": list(counts.keys()), "Count": list(counts.values())})
        st.altair_chart(
            alt.Chart(count_df).mark_bar(color=INK).encode(
                x=alt.X("Result:N", sort=None, title=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y("Count:Q", title=f"Count out of {shots:,}"),
                tooltip=["Result", "Count"],
            ).properties(height=220),
            use_container_width=True,
        )

with grover_tab:
    st.markdown('<p class="lede">Hide one item among many. A regular computer has to check items '
                'one by one, about N/2 tries on average. Grover\'s algorithm finds it in about √N '
                'rounds by boosting the right answer\'s amplitude a little each round.</p>',
                unsafe_allow_html=True)
    g1, g2 = st.columns([1, 2], gap="large")
    with g1:
        gq = st.slider("Qubits", 2, 6, 4, key="grover_qubits")
        size = 2 ** gq
        best = quantum.grover_optimal_iterations(gq)
        labels = [format(i, f"0{gq}b") for i in range(size)]
        marked = st.selectbox("Hidden item", range(size), index=size - 3,
                              format_func=lambda i: f"|{labels[i]}⟩ (item {i})")
        rounds = st.slider("Grover rounds", 0, best * 3, best)
        probs, history = quantum.grover_search(gq, marked, rounds)
        st.metric("Chance of finding the hidden item", f"{history[-1] * 100:.1f}%")
        st.caption(f"{size} items. Guessing one by one takes about {size / 2:.0f} tries on average. "
                   f"Grover's best is {best} round{'s' if best != 1 else ''}. Go past it and the "
                   "amplitude overshoots, so the chance drops again.")
    with g2:
        _, full_history = quantum.grover_search(gq, marked, best * 3)
        hist_df = pd.DataFrame({"Round": range(len(full_history)), "Chance": full_history})
        line = alt.Chart(hist_df).mark_line(color=ACCENT, strokeWidth=3, point=alt.OverlayMarkDef(color=ACCENT, size=40)).encode(
            x=alt.X("Round:Q", axis=alt.Axis(tickMinStep=1)),
            y=alt.Y("Chance:Q", scale=alt.Scale(domain=[0, 1]), title="Chance of hidden item"),
        )
        here = alt.Chart(hist_df[hist_df["Round"] == rounds]).mark_point(
            color=INK, size=140, filled=True).encode(x="Round:Q", y="Chance:Q")
        st.altair_chart((line + here).properties(height=230), use_container_width=True)

        dist_df = pd.DataFrame({"Item": [f"|{l}⟩" for l in labels], "Probability": probs,
                                "Hidden": ["Hidden item" if i == marked else "Other" for i in range(size)]})
        st.altair_chart(
            alt.Chart(dist_df).mark_bar().encode(
                x=alt.X("Item:N", sort=None, title=None,
                        axis=alt.Axis(labelAngle=-90 if size > 16 else 0, labelFontSize=10)),
                y=alt.Y("Probability:Q", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("Hidden:N", scale=alt.Scale(domain=["Hidden item", "Other"],
                                                            range=[ACCENT, "#C9CDE3"]), legend=None),
                tooltip=["Item", alt.Tooltip("Probability:Q", format=".3f")],
            ).properties(height=220),
            use_container_width=True,
        )

st.caption("Every number here comes from quantum.py, a simulator written from scratch with NumPy. No quantum libraries used.")
