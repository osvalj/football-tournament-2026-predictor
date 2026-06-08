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
    background-color: #111827; border-radius: 10px;
    padding: 6px 8px; gap: 8px;
}
.stTabs [data-baseweb="tab"] {
    background-color: transparent; color: #6b7280;
    font-family: 'DM Sans', sans-serif; font-weight: 500;
    padding: 10px 20px; font-size: 0.85rem;
    letter-spacing: 0.03em;
}
.stTabs [aria-selected="true"] {
    background-color: #f59e0b !important; color: #000 !important;
    border-radius: 6px; font-weight: 600;
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

# ── Banderas por país ─────────────────────────────────────────
FLAGS = {
    "France": "🇫🇷", "Argentina": "🇦🇷", "Brazil": "🇧🇷",
    "England": "🏴󠁧󠁢󠁥󠁮󠁧󠁿", "Belgium": "🇧🇪", "Spain": "🇪🇸",
    "Portugal": "🇵🇹", "Germany": "🇩🇪", "Croatia": "🇭🇷",
    "Morocco": "🇲🇦", "Netherlands": "🇳🇱", "Uruguay": "🇺🇾",
    "Switzerland": "🇨🇭", "Colombia": "🇨🇴", "Japan": "🇯🇵",
    "Mexico": "🇲🇽", "Korea Republic": "🇰🇷", "South Africa": "🇿🇦",
    "Czechia": "🇨🇿", "Canada": "🇨🇦", "Qatar": "🇶🇦",
    "Bosnia and Herzegovina": "🇧🇦", "Brazil": "🇧🇷",
    "Scotland": "🏴󠁧󠁢󠁳󠁣󠁴󠁿", "Haiti": "🇭🇹", "USA": "🇺🇸",
    "Australia": "🇦🇺", "Paraguay": "🇵🇾", "Turkey": "🇹🇷",
    "Ecuador": "🇪🇨", "Côte d'Ivoire": "🇨🇮", "Curaçao": "🇨🇼",
    "Japan": "🇯🇵", "Tunisia": "🇹🇳", "Sweden": "🇸🇪",
    "Belgium": "🇧🇪", "IR Iran": "🇮🇷", "Egypt": "🇪🇬",
    "New Zealand": "🇳🇿", "Saudi Arabia": "🇸🇦", "Cape Verde": "🇨🇻",
    "Senegal": "🇸🇳", "Iraq": "🇮🇶", "Norway": "🇳🇴",
    "Algeria": "🇩🇿", "Austria": "🇦🇹", "Jordan": "🇯🇴",
    "Portugal": "🇵🇹", "Congo DR": "🇨🇩", "Uzbekistan": "🇺🇿",
    "Ghana": "🇬🇭", "Panama": "🇵🇦",
}

def flag(team):
    return FLAGS.get(team, "🏳️")

def flagged(team):
    return f"{FLAGS.get(team, '🏳️')} {team}"


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

# ── Sidebar — Buscador por país ───────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="font-family:'Bebas Neue',sans-serif; font-size:1.4rem;
                letter-spacing:0.08em; color:#f59e0b; margin-bottom:8px;">
        🔍 BUSCAR POR PAÍS
    </div>
    """, unsafe_allow_html=True)

    todos_equipos_sorted = sorted(champion_data.keys())
    pais_sel = st.selectbox(
        "Selecciona tu selección",
        options=["— Ver todos —"] + todos_equipos_sorted,
        index=0
    )

    if pais_sel != "— Ver todos —":
        probs = champion_data[pais_sel]
        grupo_pais = next(
            (g for g, data in match_data.items() if pais_sel in data["equipos"]), None
        )
        ranking_pos = sorted(
            champion_data.items(), key=lambda x: x[1]["campeon"], reverse=True
        )
        posicion = next((i+1 for i, (t, _) in enumerate(ranking_pos) if t == pais_sel), None)

        st.markdown(f"""
        <div style="background:#111827; border:1px solid #1f2937;
                    border-radius:10px; padding:14px; margin-top:12px;">
            <div style="font-family:'Bebas Neue',sans-serif; font-size:1.2rem;
                        color:#f1f5f9; margin-bottom:10px;">{flagged(pais_sel)}</div>
            <div style="display:flex; flex-direction:column; gap:8px;">
                <div style="display:flex; justify-content:space-between;">
                    <span style="color:#6b7280; font-size:0.85rem;">🏆 Campeón</span>
                    <span style="font-family:'Bebas Neue',sans-serif;
                                 color:#f59e0b; font-size:1rem;">{probs['campeon']}%</span>
                </div>
                <div style="display:flex; justify-content:space-between;">
                    <span style="color:#6b7280; font-size:0.85rem;">🥈 Final</span>
                    <span style="font-family:'Bebas Neue',sans-serif;
                                 color:#f1f5f9; font-size:1rem;">{probs['final']}%</span>
                </div>
                <div style="display:flex; justify-content:space-between;">
                    <span style="color:#6b7280; font-size:0.85rem;">🥉 Semifinales</span>
                    <span style="font-family:'Bebas Neue',sans-serif;
                                 color:#f1f5f9; font-size:1rem;">{probs['semifinales']}%</span>
                </div>
                <div style="display:flex; justify-content:space-between;">
                    <span style="color:#6b7280; font-size:0.85rem;">📊 Ranking favoritos</span>
                    <span style="font-family:'Bebas Neue',sans-serif;
                                 color:#f1f5f9; font-size:1rem;">#{posicion}</span>
                </div>
                <div style="display:flex; justify-content:space-between;">
                    <span style="color:#6b7280; font-size:0.85rem;">📋 Grupo</span>
                    <span style="font-family:'Bebas Neue',sans-serif;
                                 color:#f1f5f9; font-size:1rem;">{grupo_pais}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Partidos del equipo seleccionado
        st.markdown("""
        <div style="font-family:'Bebas Neue',sans-serif; font-size:1rem;
                    color:#f59e0b; margin-top:16px; margin-bottom:8px;">
            PARTIDOS DE GRUPO
        </div>
        """, unsafe_allow_html=True)

        if grupo_pais:
            partidos_pais = [
                p for p in match_data[grupo_pais]["partidos"]
                if p["home"] == pais_sel or p["away"] == pais_sel
            ]
            for p in partidos_pais:
                es_local = p["home"] == pais_sel
                rival    = p["away"] if es_local else p["home"]
                p_equipo = p["home_win"] if es_local else p["away_win"]
                p_draw   = p["draw"]
                p_rival  = p["away_win"] if es_local else p["home_win"]
                color    = "#22c55e" if p_equipo > p_rival and p_equipo > p_draw else                            "#f59e0b" if p_draw > p_equipo and p_draw > p_rival else "#ef4444"
                resultado = "Favorito" if p_equipo > p_rival and p_equipo > p_draw else                             "Empate probable" if p_draw > p_equipo and p_draw > p_rival else                             "Desfavorable"
                st.markdown(f"""
                <div style="background:#111827; border:1px solid #1f2937;
                            border-radius:8px; padding:10px 12px; margin-bottom:6px;">
                    <div style="font-size:0.8rem; color:#6b7280; margin-bottom:4px;">
                        vs {flag(rival)} {rival}
                    </div>
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="font-family:'Bebas Neue',sans-serif;
                                     font-size:1.1rem; color:{color};">{p_equipo}%</span>
                        <span style="font-size:0.75rem; color:{color};">{resultado}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

# ── Tabs principales ──────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🏆  PROBABILIDADES MUNDIAL",
    "⚽  PARTIDOS POR GRUPO",
    "📊  CLASIFICACIÓN POR GRUPO",
    "🧠  SOBRE EL PROYECTO"
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
        team_display = flagged(team)
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:1.5rem;">{medal}</div>
                <div class="metric-value">{probs['campeon']}%</div>
                <div style="font-family:'Bebas Neue',sans-serif; font-size:1.2rem; margin-top:4px;">{team_display}</div>
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
    teams  = [flagged(t[0]) for t in top15]
    camps  = [t[1]["campeon"] for t in top15]
    finals = [t[1]["final"] for t in top15]
    semis  = [t[1]["semifinales"] for t in top15]

    # Tres gráficos separados — uno por ronda
    col_g1, col_g2, col_g3 = st.columns(3)

    graficos = [
        ("🏆 Campeón", camps, "#f59e0b", "#000000"),
        ("🥈 Final", finals, "rgba(245,158,11,0.55)", "#ffffff"),
        ("🥉 Semifinales", semis, "rgba(147,197,253,0.7)", "#000000"),
    ]

    for col, (titulo, valores, color_barra, color_texto) in zip([col_g1, col_g2, col_g3], graficos):
        with col:
            st.markdown(f"<div style='text-align:center; font-family:Bebas Neue,sans-serif; "
                        f"font-size:1.1rem; color:#f1f5f9; margin-bottom:8px;'>{titulo}</div>",
                        unsafe_allow_html=True)
            fig = go.Figure(go.Bar(
                y=teams[::-1],
                x=valores[::-1],
                orientation="h",
                marker_color=color_barra,
                marker_line_width=0,
                text=[f"{v}%" for v in valores[::-1]],
                textposition="inside",
                textfont=dict(color=color_texto, size=10),
                insidetextanchor="middle",
            ))
            fig.update_layout(
                plot_bgcolor="#111827",
                paper_bgcolor="#111827",
                font=dict(family="DM Sans", color="#f1f5f9"),
                xaxis=dict(
                    gridcolor="#1f2937",
                    tickfont=dict(size=9),
                    title="",
                ),
                yaxis=dict(tickfont=dict(size=10)),
                height=460,
                margin=dict(l=5, r=10, t=10, b=30),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)
    with st.expander("Ver todos los equipos"):
        df_all = pd.DataFrame([
            {"Equipo": flagged(t), "🏆 Campeón": f"{v['campeon']}%",
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
                    <div style="font-family:'Bebas Neue',sans-serif; font-size:1.1rem;">{flag(home)} {home}</div>
                    <div style="font-size:1.6rem; font-family:'Bebas Neue',sans-serif; color:#22c55e;">{p_home}%</div>
                </div>
                <div style="text-align:center; flex:0.6;">
                    <div style="color:#6b7280; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.1em;">Empate</div>
                    <div style="font-size:1.4rem; font-family:'Bebas Neue',sans-serif; color:#f59e0b;">{p_draw}%</div>
                </div>
                <div style="text-align:right; flex:1;">
                    <div style="font-family:'Bebas Neue',sans-serif; font-size:1.1rem;">{flag(away)} {away}</div>
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
                        "Equipo": flagged(equipo),
                        "Prob.":  f"{pct}%",
                    }
                    for i, (equipo, pct) in enumerate(sorted_eq)
                ])
                st.dataframe(df_grupo, use_container_width=True, hide_index=True, height=178)
        st.markdown("<br>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# TAB 4 — Sobre el proyecto
# ════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">SOBRE EL PROYECTO</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns([3, 2])

    with col_a:
        st.markdown("""
### ¿Qué es esto?

Este dashboard muestra las predicciones de un modelo de **inteligencia artificial** entrenado para predecir resultados de partidos de fútbol internacional. El modelo analiza datos históricos de más de 10.000 partidos y calcula la probabilidad de que gane el equipo local, el visitante, o que el partido termine en empate.

Con esas probabilidades, simulamos el torneo completo **100.000 veces** para estimar qué equipo tiene más posibilidades de ser campeón.

---

### ¿Cómo funciona? La analogía del meteorólogo

Imagina un meteorólogo que predice el tiempo. No dice "mañana lloverá" con certeza absoluta — dice "hay un 70% de probabilidad de lluvia". Lo sabe porque ha estudiado miles de días anteriores con condiciones similares.

Nuestro modelo hace exactamente lo mismo con el fútbol:

- Estudia miles de partidos históricos
- Identifica patrones: ¿los equipos mejor rankeados ganan más? ¿La forma reciente importa? ¿Los equipos con mejores jugadores tienen ventaja?
- Con esos patrones, calcula probabilidades para partidos nuevos

El modelo no adivina — **aprende de la historia**.

---

### ¿Qué es el Random Forest?

El algoritmo que usamos se llama **Random Forest** (Bosque Aleatorio). Para entenderlo, primero imagina un árbol de decisión:

> *"¿El equipo local está en el top 10 del ranking FIFA?"*
> - Sí → *"¿Ha ganado 4 de sus últimos 5 partidos?"*
>   - Sí → **Predice victoria local**
>   - No → **Predice empate**
> - No → *"¿El visitante tiene ventaja de más de 200 puntos FIFA?"*
>   - Sí → **Predice victoria visitante**

Un solo árbol es como pedir opinión a una sola persona. El Random Forest **consulta a 649 árboles diferentes** — cada uno entrenado con datos ligeramente distintos — y toma la decisión por mayoría. Como pedir opinión a 649 expertos y quedarte con la respuesta más votada.

---

### ¿Qué información usa el modelo?

El modelo analiza **26 variables** por partido:

| Variable | Qué mide |
|----------|----------|
| Ranking FIFA | Posición oficial de cada selección en el mundo |
| Puntos FIFA | Puntuación acumulada en el ranking |
| Ratings EA FC 26 | Calidad de los jugadores (videojuego como proxy) |
| Forma reciente | Resultados de los últimos 5 partidos competitivos |
| Historial directo | Quién gana históricamente cuando se enfrentan |
| Promedio de goles | Goles marcados y recibidos por partido |
| Importancia del torneo | No es lo mismo un amistoso que una final |
| Campo neutral | Si ningún equipo juega en casa |

---

### ¿Qué es la simulación Monte Carlo?

Una vez que el modelo da probabilidades para cada partido, necesitamos simular el torneo completo. Aquí entra **Monte Carlo**.

Imagina que lanzas un dado trucado: la cara 1 sale el 52% de las veces (Francia gana), la cara 2 el 24% (empate), la cara 3 el 24% (Argentina gana). Lanzas el dado para cada partido y avanza quien gana.

Repetimos esto **100.000 veces** con el torneo completo. Al final contamos: *¿cuántas veces ganó Francia?* Si ganó en 10.700 de las 100.000 simulaciones, su probabilidad de ser campeón es del **10.7%**.

Esto captura algo importante: **incluso el favorito puede perder**. Un equipo con 60% de probabilidad de ganar cada partido todavía tiene un 40% de perder. En 6 partidos eliminatorios, el azar acumula. Por eso ningún equipo supera el 11% — el torneo es genuinamente impredecible.

---

### ¿Qué tan bueno es el modelo?

El modelo fue evaluado en **502 partidos reales** jugados entre junio de 2025 y 2026 — partidos que el modelo nunca vio durante el entrenamiento.

| Métrica | Resultado |
|---------|-----------|
| Accuracy | **65.1%** — acierta 2 de cada 3 partidos |
| F1 macro | **56.1%** — rendimiento equilibrado entre las 3 clases |

Para contexto: predecir siempre "gana el local" daría ~47% de accuracy. Predecir al azar daría ~33%. Nuestro modelo supera ambas referencias con claridad.

La clase más difícil de predecir son los **empates** — con solo un 22% de F1. El empate es el resultado más aleatorio del fútbol y el más difícil de capturar con datos estructurales. Esto es consistente con la literatura académica: ningún modelo público supera ese techo con datos abiertos.

---

### ¿Cuáles son las limitaciones?

Este es un proyecto de portfolio con datos públicos. Sus predicciones tienen límites claros:

- **No conoce las alineaciones** del día del partido — si Mbappé está lesionado, el modelo no lo sabe
- **No incorpora contexto táctico** ni análisis de juego real
- **Los ratings de EA FC** son un proxy imperfecto de la calidad real de los jugadores
- **El bracket de la simulación** es una simplificación del cuadro oficial de la FIFA

Las probabilidades son estimaciones estadísticas basadas en patrones históricos. **No son consejos de apuestas.**
        """)

    with col_b:
        st.markdown("""
### Datos utilizados

**Resultados históricos**
49.287 partidos internacionales desde 1872

**Ranking FIFA**
67.472 registros de ranking desde 1992

**EA FC 26 Player Ratings**
16.228 jugadores con valoraciones de habilidad

**Transfermarkt**
47.689 jugadores y datos de selecciones nacionales

---

### Tecnología

- **Python 3.11**
- **scikit-learn** — modelo y pipeline
- **pandas / numpy** — datos
- **XGBoost / LightGBM** — modelos comparados
- **Streamlit** — este dashboard
- **Plotly** — visualizaciones

---

### Resultados del modelo

**Modelos evaluados:** 8

**Mejor modelo:** Random Forest

**Hiperparámetros finales:**
```
n_estimators = 649
max_depth = 4
max_features = 0.5
bootstrap = False
class_weight = balanced
```

**Simulaciones Monte Carlo:** 100.000

---

### Autor

**Osvaldo Hernández**
Data Scientist

© 2026 · Todos los derechos reservados
        """)


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
