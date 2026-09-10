import io
import re
from typing import Dict, List, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="Optimización de Generadores",
    page_icon="⚡",
    layout="wide",
)

GEN_LOAD_RE = re.compile(r"Percent power used GEN\s*([1-4])", re.IGNORECASE)
DEFAULT_THRESHOLD = 20.0
DEFAULT_MIN_HOURS = 5.0


def read_uploaded_file(uploaded_file, selected_sheet: str | None = None) -> Dict[str, pd.DataFrame]:
    """Read CSV or Excel and return {rig/sheet_name: dataframe}."""
    name = uploaded_file.name.lower()
    raw = uploaded_file.getvalue()

    if name.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(raw))
        return {"992": df}

    if name.endswith((".xlsx", ".xlsm", ".xls")):
        xls = pd.ExcelFile(io.BytesIO(raw))
        sheets = xls.sheet_names
        if selected_sheet and selected_sheet in sheets:
            return {selected_sheet: pd.read_excel(io.BytesIO(raw), sheet_name=selected_sheet)}
        result = {}
        for sheet in sheets:
            if sheet.lower() in {"parametros", "parameters", "readme"}:
                continue
            result[sheet] = pd.read_excel(io.BytesIO(raw), sheet_name=sheet)
        return result

    raise ValueError("Formato no soportado. Usa CSV o Excel (.xlsx/.xlsm/.xls).")


def prepare_rig(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Normalize time and detect the four GEN load columns."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    if "Time" not in df.columns:
        raise ValueError("No encuentro la columna 'Time'.")

    gen_cols = []
    for col in df.columns:
        match = GEN_LOAD_RE.search(col)
        if match:
            gen_cols.append((int(match.group(1)), col))
    gen_cols = [col for _, col in sorted(gen_cols)]

    if len(gen_cols) < 1:
        raise ValueError("No encuentro columnas como 'Percent power used GEN 1' ... 'GEN 4'.")

    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    df = df.dropna(subset=["Time"]).sort_values("Time").reset_index(drop=True)

    for col in gen_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return df, gen_cols


def detect_events(
    df: pd.DataFrame,
    gen_cols: List[str],
    threshold: float,
    min_hours: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Detect contiguous periods with >=2 active generators and >=2 low-load generators."""
    work = df.copy()
    values = work[gen_cols]
    work["generadores_activos"] = (values > 0).sum(axis=1)
    work["generadores_baja"] = ((values > 0) & (values <= threshold)).sum(axis=1)
    work["condicion_alerta"] = (work["generadores_activos"] >= 2) & (work["generadores_baja"] >= 2)

    # Estimate normal sampling interval from all timestamps.
    diffs = work["Time"].diff().dropna().dt.total_seconds().div(60)
    interval_min = float(diffs.median()) if not diffs.empty else 20.0
    if interval_min <= 0:
        interval_min = 20.0

    # Build groups of contiguous alert records. A gap substantially larger than the
    # expected sampling interval breaks continuity.
    alert = work[work["condicion_alerta"]].copy()
    if alert.empty:
        return work, pd.DataFrame()

    alert["gap_min"] = alert["Time"].diff().dt.total_seconds().div(60)
    alert["new_group"] = alert["gap_min"].isna() | (alert["gap_min"] > interval_min * 1.5)
    alert["grupo"] = alert["new_group"].cumsum()

    events = []
    for _, g in alert.groupby("grupo"):
        start = g["Time"].min()
        end = g["Time"].max()
        duration_hours = ((end - start).total_seconds() / 3600.0) + (interval_min / 60.0)
        involved = []
        for col in gen_cols:
            # Generator involved if it is active for at least one record in the event.
            if (g[col] > 0).any():
                m = GEN_LOAD_RE.search(col)
                involved.append(f"GEN {m.group(1)}" if m else col)

        avg_loads = {col: g[col].mean() for col in gen_cols if (g[col] > 0).any()}
        avg_all_active = pd.Series(avg_loads).mean() if avg_loads else 0.0

        # Peak number of active and low-load generators during the event.
        max_active = int(g["generadores_activos"].max())
        max_low = int(g["generadores_baja"].max())

        events.append({
            "Inicio": start,
            "Fin": end + pd.Timedelta(minutes=interval_min),
            "Duración (h)": round(duration_hours, 2),
            "GEN involucrados": " + ".join(involved),
            "Carga promedio GEN involucrados (%)": round(float(avg_all_active), 2),
            "Máx. GEN activos": max_active,
            "Máx. GEN baja carga": max_low,
            "Cumple >5h": duration_hours > min_hours,
        })

    events_df = pd.DataFrame(events).sort_values("Inicio").reset_index(drop=True)
    return work, events_df[events_df["Cumple >5h"]].copy()


def make_load_chart(df, gen_cols, threshold, events, rig):

    colors = {
        "GEN 1": "#00B0F0",  # Azul brillante
        "GEN 2": "#FFC000",  # Amarillo
        "GEN 3": "#00B050",  # Verde
        "GEN 4": "#FF4D4D"   # Rojo
    }

    fig = go.Figure()

    labels = {}

    for col in gen_cols:

        m = GEN_LOAD_RE.search(col)

        label = f"GEN {m.group(1)}" if m else col

        labels[col] = label

        fig.add_trace(
            go.Scatter(
                x=df["Time"],
                y=df[col],
                mode="lines",
                name=label,
                line=dict(
                    color=colors[label],
                    width=3
                ),
                hovertemplate=f"%{{x|%d/%m %H:%M}}<br>{label}: %{{y:.1f}}%<extra></extra>",
            )
        )

    fig.add_hline(
    y=threshold,
    line_color="white",
    line_dash="dot",
    line_width=4,
    annotation_text=f"UMBRAL {threshold:.0f}%",
    annotation_font_color="white",
    annotation_font_size=14,
)

    for _, ev in events.iterrows():

        fig.add_vrect(
            x0=ev["Inicio"],
            x1=ev["Fin"],
            fillcolor="rgba(102,204,255,0.10)",
            line_width=0,
            layer="below",
        )

        fig.update_layout(

            title=dict(
                text=f"Carga de los 4 generadores — Rig {rig}",
                font=dict(
                    color="#FFFFFF",
                    size=26
                )
            ),
            
            plot_bgcolor="#102542",
            paper_bgcolor="#102542",
        
            font=dict(
                color="white",
                size=12
            ),

          xaxis_title="Tiempo",
          yaxis_title="Carga (%)",

            fig.add_annotation(
                text="Tiempo",
                x=0.5,
                y=-0.15,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(color="white", size=14)
            )
            
            fig.add_annotation(
                text="Carga (%)",
                x=-0.06,
                y=0.5,
                textangle=-90,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(color="white", size=14)
            )

            xaxis=dict(
                tickfont=dict(
                    color="white",
                    size=12
                ),
                showgrid=True,
                gridcolor="rgba(255,255,255,0.08)",
                color="white"
            ),
            
            yaxis=dict(
                tickfont=dict(
                    color="white",
                    size=12
                ),
                showgrid=True,
                gridcolor="rgba(255,255,255,0.08)",
                color="white",
                range=[0, max(50, threshold + 10)]
            ),

            legend=dict(
                bgcolor="rgba(0,0,0,0)",
                font=dict(
                    color="#FFFFFF",
                    size=13
                )
            ),
          
            hovermode="x unified",
        
            margin=dict(
                l=40,
                r=20,
                t=70,
                b=40
            ),
        
            height=600
        )
    
    return fig


def make_fleet_chart(summary: pd.DataFrame):
    s = summary.sort_values("Eventos >5h", ascending=True)
    fig = go.Figure(go.Bar(
        x=s["Eventos >5h"],
        y=s["Taladro"],
        orientation="h",
        text=s["Eventos >5h"],
        textposition="outside",
        hovertemplate="Taladro %{y}<br>Eventos >5h: %{x}<extra></extra>",
    ))
    fig.update_layout(
        title="Eventos prolongados de operación multigenerador a baja carga",
        xaxis_title="Cantidad de eventos >5 h",
        yaxis_title="Taladro",
        margin=dict(l=40, r=40, t=70, b=40),
        height=390,
    )
    return fig


st.title("⚡ Optimización de Utilización de Generadores")
st.caption("Detección de operación multigenerador a baja carga durante períodos prolongados")

with st.sidebar:
    st.header("Parámetros")
    threshold = st.number_input(
        "Umbral de baja carga (%)",
        min_value=1.0, max_value=100.0, value=DEFAULT_THRESHOLD, step=1.0,
        help="Un generador cuenta como 'baja carga' si está >0% y <= este valor.",
    )
    min_hours = st.number_input(
        "Duración mínima para alerta (h)",
        min_value=0.5, max_value=72.0, value=DEFAULT_MIN_HOURS, step=0.5,
        help="La condición debe mantenerse de forma continua por más de este tiempo.",
    )
    st.markdown("---")
    st.markdown("**Regla:** 2 o más GEN activos + 2 o más GEN a baja carga + duración > umbral.")

uploaded = st.file_uploader(
    "Carga tu archivo de datos",
    type=["xlsx", "xlsm", "xls", "csv"],
    help="Puedes subir el CSV del Rig 992 o el Excel consolidado con una hoja por taladro.",
)

if not uploaded:
    st.info("Empieza cargando el CSV del Rig 992 para probar el modelo.")
    st.stop()

try:
    # For Excel, allow an explicit sheet selection.
    if uploaded.name.lower().endswith((".xlsx", ".xlsm", ".xls")):
        raw = uploaded.getvalue()
        xls = pd.ExcelFile(io.BytesIO(raw))
        valid_sheets = [s for s in xls.sheet_names if s.lower() not in {"parametros", "parameters", "readme"}]
        selected = st.selectbox("Hoja/taladro", valid_sheets)
        rig_data = read_uploaded_file(uploaded, selected_sheet=selected)
    else:
        rig_data = read_uploaded_file(uploaded)
except Exception as e:
    st.error(f"No pude leer el archivo: {e}")
    st.stop()

all_results = []
rig_prepared = {}

for rig, raw_df in rig_data.items():
    try:
        df, gen_cols = prepare_rig(raw_df)
        work, events = detect_events(df, gen_cols, threshold, min_hours)
        rig_prepared[rig] = (df, gen_cols, work, events)

        all_results.append({
            "Taladro": str(rig),
            "Registros": len(df),
            "GEN máx. activos": int(work["generadores_activos"].max()) if len(work) else 0,
            "Eventos >5h": len(events),
            "Horas en eventos >5h": round(events["Duración (h)"].sum(), 2) if not events.empty else 0.0,
            "Mayor evento (h)": round(events["Duración (h)"].max(), 2) if not events.empty else 0.0,
        })
    except Exception as e:
        st.warning(f"Taladro {rig}: se omitió porque no tiene la estructura esperada ({e}).")

if not rig_prepared:
    st.error("No encontré ningún taladro con la estructura esperada.")
    st.stop()

summary = pd.DataFrame(all_results)

st.subheader("Vista de flota")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Taladros cargados", len(summary))
k2.metric("Eventos >5 h", int(summary["Eventos >5h"].sum()))
k3.metric("Horas acumuladas", f"{summary['Horas en eventos >5h'].sum():.1f} h")
k4.metric("Máx. duración", f"{summary['Mayor evento (h)'].max():.1f} h")

c1, c2 = st.columns([1.15, 0.85])
with c1:
    st.plotly_chart(make_fleet_chart(summary), use_container_width=True)
with c2:
    st.markdown("### Criterio aplicado")
    st.markdown(
        f"""
        **No es alerta:** 1 solo generador al 10%, 15% o 20%.  \n\n
        **Condición de revisión:** 2 o más generadores activos y 2 o más con carga ≤ **{threshold:.0f}%**.  \n\n
        **Alerta prolongada:** la condición anterior permanece **> {min_hours:g} h continuas**.  \n\n
        El análisis es independiente de si participan GEN 1+2, GEN 2+4, GEN 1+3, GEN 1+2+4, etc.
        """
    )

st.markdown("---")

rig_options = list(rig_prepared.keys())
rig = st.selectbox("Analizar taladro", rig_options, index=0)
df, gen_cols, work, events = rig_prepared[rig]

m1, m2, m3, m4 = st.columns(4)
m1.metric("Registros", len(df))
m2.metric("GEN máx. activos", int(work["generadores_activos"].max()))
m3.metric("Eventos >5 h", len(events))
m4.metric("Horas en eventos", f"{events['Duración (h)'].sum():.2f} h" if not events.empty else "0.00 h")

st.plotly_chart(make_load_chart(df, gen_cols, threshold, events, str(rig)), use_container_width=True)

st.markdown("### Eventos detectados")
if events.empty:
    st.success(
        f"No se encontraron eventos que cumplan: ≥2 generadores en baja carga (≤{threshold:.0f}%) durante >{min_hours:g} h continuas."
    )
else:
    st.dataframe(events, use_container_width=True, hide_index=True)

# Download event table for the selected rig.
out = io.BytesIO()
with pd.ExcelWriter(out, engine="openpyxl") as writer:
    summary.to_excel(writer, sheet_name="Resumen Flota", index=False)
    events.to_excel(writer, sheet_name=f"Eventos {str(rig)[:20]}", index=False)
    work.to_excel(writer, sheet_name=f"Datos {str(rig)[:19]}", index=False)
out.seek(0)

st.download_button(
    "📥 Descargar análisis en Excel",
    data=out,
    file_name=f"Analisis_Generadores_{rig}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
