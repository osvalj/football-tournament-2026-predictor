"""
Football Tournament 2026 Predictor — Dashboard
Simulación Monte Carlo con 100.000 iteraciones

Copyright (c) 2026 Osvaldo Hernández. Todos los derechos reservados.

Este software y su código fuente son propiedad de Osvaldo Hernández.
Se permite su visualización con fines educativos y de portfolio,
pero queda prohibida su reproducción, distribución o uso comercial
sin autorización expresa del autor.

Author: Osvaldo Hernández
"""

import json
import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

# ── Configuración de página ───────────────────────────────────
st.set_page_config(
    page_title="Football Tournament 2026 Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Estilos ───────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background-color: #0a0e1a !important;
    color: #f1f5f9 !important;
    font-family: 'DM Sans', sans-serif;
}
[data-testid="stSidebar"] {
    background-color: #111827 !important;
    border-right: 1px solid #1f2937;
}
h1, h2, h3 { font-family: 'Bebas Neue', sans-serif; letter-spacing: 0.05em; }
.metric-card {
    background: #111827; border: 1px solid #1f2937;
    border-radius: 12px; padding: 20px 24px; text-align: center;
}
.metric-value {
    font-family: 'Bebas Neue', sans-serif; font-size: 2.4rem;
    color: #f59e0b; line-height: 1;
}
.metric-label {
    font-size: 0.75rem; color: #6b7280;
    text-transform: uppercase; letter-spacing: 0.1em; margin-top: 4px;
}
.match-card {
    background: #111827; border: 1px solid #1f2937;
    border-radius: 10px; padding: 16px 20px; margin-bottom: 10px;
}
.section-title {
    font-family: 'Bebas Neue', sans-serif; font-size: 1.8rem;
    letter-spacing: 0.08em; border-bottom: 2px solid #f59e0b;
    padding-bottom: 6px; margin-bottom: 20px; color: #f1f5f9;
}
.stTabs [data-baseweb="tab-list"] {
    background-color: #111827; border-radius: 8px; padding: 4px; gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background-color: transparent; color: #6b7280;
    font-family: 'DM Sans', sans-serif; font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background-color: #f59e0b !important; color: #000 !important; border-radius: 6px;
}
</style>
""", unsafe_allow_html=True)

# ── Cargar datos ──────────────────────────────────────────────
BASE_DIR = Path(__file__).parent

@st.cache_data
def load_data():
    with open(BASE_DIR / "champion_probs.json") as f:
        champion = json.load(f)
    with open(BASE_DIR / "match_probs.json") as f:
        matches = json.load(f)
    with open(BASE_DIR / "group_probs.json") as f:
        groups = json.load(f)
    return champion, matches, groups

champion_data, match_data, group_data = load_data()

# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 10px 0 20px 0;">
    <div style="font-family: 'Bebas Neue', sans-serif; font-size: 3rem;
                letter-spacing: 0.1em; line-height: 1; color: #f1f5f9;">
        ⚽ FOOTBALL TOURNAMENT 2026 PREDICTOR
    </div>
    <div style="color: #6b7280; font-size: 0.9rem; margin-top: 6px;">
        Random Forest · 100,000 simulaciones Monte Carlo · Accuracy 65.1% · F1 macro 56.1%
    </div>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Tabs principales ──────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🏆  PROBABILIDADES MUNDIAL",
    "⚽  PARTIDOS POR GRUPO",
    "📊  CLASIFICACIÓN POR GRUPO"
])

# ════════════════════════════════════════════════════════════════
# TAB 1 — Probabilidades de campeón
# ════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-title">FAVORITOS AL TÍTULO</div>', unsafe_allow_html=True)

    sorted_teams = sorted(champion_data.items(), key=lambda x: x[1]["campeon"], reverse=True)

    col1, col2, col3 = st.columns(3)
    for i, col in enumerate([col1, col2, col3]):
        team, probs = sorted_teams[i]
        medal = ["🥇", "🥈", "🥉"][i]
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:1.5rem;">{medal}</div>
                <div class="metric-value">{probs['campeon']}%</div>
                <div style="font-family:'Bebas Neue',sans-serif; font-size:1.2rem; margin-top:4px;">{team}</div>
                <div class="metric-label">Probabilidad de ser campeón</div>
                <div style="margin-top:10px; display:flex; gap:8px; justify-content:center;
                            font-size:0.8rem; color:#6b7280;">
                    <span>Final: {probs['final']}%</span>
                    <span>·</span>
                    <span>Semis: {probs['semifinales']}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    top15  = sorted_teams[:15]
    teams  = [t[0] for t in top15]
    camps  = [t[1]["campeon"] for t in top15]
    finals = [t[1]["final"] for t in top15]
    semis  = [t[1]["semifinales"] for t in top15]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=teams[::-1], x=semis[::-1], name="Semifinales", orientation="h",
        marker_color="rgba(59, 130, 246, 0.27)", marker_line_width=0,
    ))
    fig.add_trace(go.Bar(
        y=teams[::-1], x=finals[::-1], name="Final", orientation="h",
        marker_color="rgba(245, 158, 11, 0.53)", marker_line_width=0,
    ))
    fig.add_trace(go.Bar(
        y=teams[::-1], x=camps[::-1], name="Campeón", orientation="h",
        marker_color="#f59e0b", marker_line_width=0,
        text=[f"{v}%" for v in camps[::-1]], textposition="outside",
        textfont=dict(color="#f1f5f9", size=11),
    ))
    fig.update_layout(
        barmode="overlay",
        plot_bgcolor="#111827", paper_bgcolor="#0a0e1a",
        font=dict(family="DM Sans", color="#f1f5f9"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(title="Probabilidad (%)", gridcolor="#1f2937", tickfont=dict(size=11)),
        yaxis=dict(tickfont=dict(size=12)),
        height=520,
        margin=dict(l=10, r=60, t=40, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Ver todos los equipos"):
        df_all = pd.DataFrame([
            {"Equipo": t, "🏆 Campeón": f"{v['campeon']}%",
             "🥈 Final": f"{v['final']}%", "🥉 Semifinales": f"{v['semifinales']}%"}
            for t, v in sorted_teams
        ])
        st.dataframe(df_all, use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════════
# TAB 2 — Partidos por grupo
# ════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">PROBABILIDADES POR PARTIDO</div>', unsafe_allow_html=True)

    grupo_sel = st.selectbox(
        "Selecciona un grupo",
        options=list(match_data.keys()),
        format_func=lambda g: f"Grupo {g} — {' · '.join(match_data[g]['equipos'])}"
    )

    for p in match_data[grupo_sel]["partidos"]:
        home   = p["home"]
        away   = p["away"]
        p_home = p["home_win"]
        p_draw = p["draw"]
        p_away = p["away_win"]
        winner = home if p_home > p_away and p_home > p_draw else \
                 away if p_away > p_home and p_away > p_draw else "Empate"

        st.markdown(f"""
        <div class="match-card">
            <div style="display:flex; justify-content:space-between;
                        align-items:center; margin-bottom:12px;">
                <div style="text-align:left; flex:1;">
                    <div style="font-family:'Bebas Neue',sans-serif; font-size:1.1rem;">{home}</div>
                    <div style="font-size:1.6rem; font-family:'Bebas Neue',sans-serif; color:#22c55e;">{p_home}%</div>
                </div>
                <div style="text-align:center; flex:0.6;">
                    <div style="color:#6b7280; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.1em;">Empate</div>
                    <div style="font-size:1.4rem; font-family:'Bebas Neue',sans-serif; color:#f59e0b;">{p_draw}%</div>
                </div>
                <div style="text-align:right; flex:1;">
                    <div style="font-family:'Bebas Neue',sans-serif; font-size:1.1rem;">{away}</div>
                    <div style="font-size:1.6rem; font-family:'Bebas Neue',sans-serif; color:#ef4444;">{p_away}%</div>
                </div>
            </div>
            <div style="display:flex; gap:3px; height:6px; border-radius:4px; overflow:hidden;">
                <div style="width:{p_home}%; background:#22c55e; border-radius:4px 0 0 4px;"></div>
                <div style="width:{p_draw}%; background:#f59e0b;"></div>
                <div style="width:{p_away}%; background:#ef4444; border-radius:0 4px 4px 0;"></div>
            </div>
            <div style="display:flex; justify-content:space-between; margin-top:6px;
                        font-size:0.7rem; color:#6b7280;">
                <span>Victoria local</span>
                <span style="color:#f59e0b; font-size:0.7rem;">Favorito: {winner}</span>
                <span>Victoria visitante</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# TAB 3 — Clasificación por grupo
# ════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">PROBABILIDAD DE CLASIFICACIÓN</div>', unsafe_allow_html=True)
    st.caption("Probabilidad de terminar en el top 2 del grupo (5.000 simulaciones)")

    grupos_lista = list(group_data.keys())

    for row_start in range(0, len(grupos_lista), 3):
        cols = st.columns(3)
        for col_idx, grupo in enumerate(grupos_lista[row_start:row_start + 3]):
            equipos   = group_data[grupo]
            sorted_eq = sorted(equipos.items(), key=lambda x: x[1], reverse=True)
            with cols[col_idx]:
                st.markdown(f"**📋 GRUPO {grupo}**")
                df_grupo = pd.DataFrame([
                    {
                        "":       "✅" if i < 2 else "❌",
                        "Equipo": equipo,
                        "Prob.":  f"{pct}%",
                    }
                    for i, (equipo, pct) in enumerate(sorted_eq)
                ])
                st.dataframe(df_grupo, use_container_width=True, hide_index=True, height=178)
        st.markdown("<br>", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style="text-align:center; color:#374151; font-size:0.8rem; padding:10px 0; line-height:1.8;">
    Football Tournament 2026 Predictor · Random Forest · 100,000 simulaciones Monte Carlo ·
    Accuracy 65.1% · F1 macro 56.1%<br>
    Datos: FIFA Rankings, EA FC 26, Transfermarkt<br>
    <span style="color:#4b5563;">© 2026 Osvaldo Hernández · Todos los derechos reservados</span>
</div>
""", unsafe_allow_html=True)
