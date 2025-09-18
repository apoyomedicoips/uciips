# scripts/uci_build_report.py
from __future__ import annotations
import os
from pathlib import Path
import base64, json, unicodedata
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Rutas
# =========================
ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "report"
ASSETS_DIR = REPORT_DIR / "assets"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# Config vía entorno
# =========================
SHEET_CSV_URL = os.getenv("SHEET_CSV_URL", "").strip()
GSHEETS_CREDENTIALS_B64 = os.getenv("GSHEETS_CREDENTIALS_B64", "").strip()
GSHEET_ID = os.getenv("GSHEET_ID", "").strip()
GSHEET_TAB = os.getenv("GSHEET_TAB", "base")

# =========================
# Utilidades
# =========================
def _accent_fold(s: str) -> str:
    if s is None:
        return ""
    return "".join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c))

def parse_date_series(sr: pd.Series) -> pd.Series:
    s = sr.astype(str).str.strip().replace({"": pd.NA, "NaT": pd.NA, "nan": pd.NA})
    dt = pd.to_datetime(s, dayfirst=True, errors="coerce")
    def _fix(y):
        if pd.isna(y):
            return pd.NaT
        if 1000 <= y.year <= 1100:
            try:
                return y.replace(year=y.year + 1000)
            except Exception:
                return y
        return y
    return dt.apply(_fix)

def to_bool(sr: pd.Series) -> pd.Series:
    s = sr.astype(str).str.strip().str.lower()
    si = {"si","sí","yes","y","1","true","verdadero"}
    no = {"no","n","0","false","falso"}
    out = pd.Series(pd.NA, index=sr.index, dtype="boolean")
    out = out.mask(s.isin(si), True).mask(s.isin(no), False)
    return out

def to_int(sr: pd.Series) -> pd.Series:
    return pd.to_numeric(sr, errors="coerce").astype("Int64")

def median_iqr(x: pd.Series):
    x = pd.to_numeric(x, errors="coerce").dropna()
    if x.empty:
        return None, None, None
    return float(x.median()), float(x.quantile(0.25)), float(x.quantile(0.75))

def safe_pct(num, den) -> float:
    return float(num)/float(den)*100.0 if den else 0.0

def fmt_int(x) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x))):
        return "0"
    try:
        return f"{int(round(float(x))):,}".replace(",", ".")
    except Exception:
        return str(x)

def fmt_pct(x, nd=1) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x))):
        return "0.0%"
    return f"{float(x):.{nd}f}%"

def fmt_float(x, nd=1) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x))):
        return f"{0:.{nd}f}"
    return f"{float(x):.{nd}f}"

def md_table(df: pd.DataFrame, index=False) -> str:
    # Formatea a Markdown; el CSS aplicará estilo
    try:
        return df.to_markdown(index=index)
    except Exception:
        cols = list(df.columns)
        rows = ["| " + " | ".join(map(str, cols)) + " |",
                "|" + "|".join("---" for _ in cols) + "|"]
        for _, r in df.iterrows():
            rows.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in r.tolist()) + " |")
        return "\n".join(rows)

# =========================
# Carga de datos
# =========================
def load_from_csv_url(url: str) -> pd.DataFrame:
    if not url:
        raise RuntimeError("SHEET_CSV_URL vacío. Defínelo en el workflow.")
    return pd.read_csv(url, dtype=str)

def load_from_gsheets_service_account(b64_json: str, sheet_id: str, tab: str="base") -> pd.DataFrame:
    import gspread
    from google.oauth2.service_account import Credentials
    if not b64_json or not sheet_id:
        raise RuntimeError("Faltan credenciales o GSHEET_ID.")
    info = json.loads(base64.b64decode(b64_json).decode("utf-8"))
    scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
    creds = Credentials.from_service_account_info(info, scopes=scopes)
    gc = gspread.authorize(creds)
    ws = gc.open_by_key(sheet_id).worksheet(tab)
    return pd.DataFrame.from_records(ws.get_all_records())

def load_data() -> pd.DataFrame:
    if SHEET_CSV_URL:
        return load_from_csv_url(SHEET_CSV_URL)
    if GSHEETS_CREDENTIALS_B64 and GSHEET_ID:
        return load_from_gsheets_service_account(GSHEETS_CREDENTIALS_B64, GSHEET_ID, GSHEET_TAB)
    raise RuntimeError("Configura SHEET_CSV_URL o variables de Service Account.")

# =========================
# Normalización
# =========================
COLMAP = {
  'Marca temporal':'marca_temporal',
  'Dirección de correo electrónico':'email',
  'Fecha de Nacimiento':'fec_nac',
  'Nombre y Apellido':'nombre',
  'Edad':'edad',
  'Fecha de Ingreso':'fec_ing',
  'Fecha de Egreso':'fec_egr',
  'Registro de Internación':'reg_intern',
  'Prontuario':'prontuario',
  'Médico Tratante':'medico',
  'Condición al Egreso':'cond_egreso',
  'Días de Internación':'los',
  'APACHE II a las 24 h del ingreso':'apache2',
  'SOFA a las 48 h del ingreso':'sofa48',
  'Origen del Paciente':'origen',
  'Tipos de Pacientes':'tipo',
  'KPC/MBL POSITIVO EN PACIENTES':'kpc_mbl',
  'Catéter de Hemodiálisis':'cateter_hd',
  'Vía Venosa Central':'vvc',
  'Ventilación Invasiva':'vi',
  'Líneas Arteriales':'lineas_art',
  'Tubo de drenaje pleural (hechos por UCIA)':'tubo_dren',
  'Traqueostomías (hechos por UCIA)':'traqueo',
  'Uso de CAF':'caf',
  'Electrocardiograma':'ecg',
  'POCUS':'pocus',
  'Doppler transcraneal':'doppler_tc',
  'Fibrobroncoscopia':'fibro',
  'Observaciones':'obs'
}

def canon_outcome(x: str) -> str:
    if not isinstance(x, str):
        return ""
    t = _accent_fold(x).lower().strip().rstrip(":")
    if "obito" in t:
        return "Óbito"
    if "alta" in t:
        return "Alta a piso"
    return x.strip().rstrip(":")

def canon_kpc(x: str) -> str:
    if not isinstance(x, str):
        return ""
    t = _accent_fold(x).lower()
    if "negativo" in t:
        return "Negativo"
    if "pendiente retorno" in t:
        return "Pendiente HR ingreso"
    if "prevalencia" in t:
        return "HR de Prevalencia"
    if "ingreso" in t:
        return "HR de Ingreso"
    if "portador" in t or "plasmido" in t or "mdr" in t:
        return "Conocido portador MDR"
    return x

def canon_servicio(x: str) -> str:
    if not isinstance(x, str):
        return ""
    t = _accent_fold(x).strip()
    repl = {
        "Traumatologia":"Traumatología", "Urologia":"Urología", "Mastologia":"Mastología",
        "IPS INTERIOR":"IPS Interior", "Reanimacion":"Reanimación", "Clinica Medica":"Clínica Médica"
    }
    return repl.get(t, x)

def prepare(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.rename(columns=COLMAP).copy()

    # Fechas
    for col in ["marca_temporal","fec_nac","fec_ing","fec_egr"]:
        if col in df.columns:
            df[col] = parse_date_series(df[col])

    # Números
    for col in ["edad","apache2","sofa48","vvc","cateter_hd","lineas_art","ecg","los","reg_intern","prontuario"]:
        if col in df.columns:
            df[col] = to_int(df[col])

    # Booleans
    for col in ["vi","tubo_dren","traqueo","caf","pocus","doppler_tc","fibro"]:
        if col in df.columns:
            df[col] = to_bool(df[col])

    # Canon
    if "cond_egreso" in df.columns:
        df["cond_egreso"] = df["cond_egreso"].astype(str).map(canon_outcome)
    if "kpc_mbl" in df.columns:
        df["kpc_mbl"] = df["kpc_mbl"].astype(str).map(canon_kpc)
    if "origen" in df.columns:
        df["origen"] = df["origen"].astype(str).map(canon_servicio)
    if "tipo" in df.columns:
        df["tipo"] = df["tipo"].astype(str).str.strip()
    if "medico" in df.columns:
        df["medico"] = df["medico"].astype(str).str.strip()

    # Recalcular LOS si hace falta
    if "fec_ing" in df.columns and "fec_egr" in df.columns:
        los_calc = (df["fec_egr"] - df["fec_ing"]).dt.days
        df["los_calc"] = los_calc.where(los_calc >= 0)
    df["los_final"] = df.get("los").fillna(df.get("los_calc"))

    # Filtrar sin fecha de ingreso
    df = df[df["fec_ing"].notna()].copy()
    return df

# =========================
# Figuras
# =========================
def timeseries_and_census(df: pd.DataFrame):
    # Ajustes de estilo gráfico (sobrios)
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["figure.figsize"] = (9, 4)

    adm = df.groupby(df["fec_ing"].dt.date).size().rename("Admisiones").to_frame()
    dis = df[df["fec_egr"].notna()].groupby(df["fec_egr"].dt.date).size().rename("Egresos").to_frame()

    start = df["fec_ing"].min()
    if df["fec_egr"].notna().any():
        start = min(start, df["fec_egr"].min())
    end = df["fec_ing"].max()
    if df["fec_egr"].notna().any():
        end = max(end, df["fec_egr"].max())
    idx = pd.date_range(start, end, freq="D").date

    ts = pd.DataFrame(index=idx)
    ts["Admisiones"] = adm.reindex(idx).fillna(0).astype(int)
    ts["Egresos"] = dis.reindex(idx).fillna(0).astype(int)

    census = pd.Series(0, index=pd.Index(idx, name="Fecha"), dtype=int)
    for _, r in df.iterrows():
        d0 = r["fec_ing"].date()
        d1 = (r["fec_egr"].date() if pd.notna(r["fec_egr"]) else r["fec_ing"].date())
        for d in pd.date_range(d0, d1, freq="D").date:
            if d in census.index:
                census.loc[d] = int(census.loc[d]) + 1

    fig1 = plt.figure()
    ax = plt.gca()
    ts.plot(ax=ax)
    ax.set_title("Admisiones y Egresos diarios")
    ax.set_xlabel("Fecha"); ax.set_ylabel("Conteo")
    ax.grid(True, axis="y", alpha=.3)
    plt.tight_layout(); plt.savefig(ASSETS_DIR / "timeseries_adm_disc.png"); plt.close(fig1)

    fig2 = plt.figure()
    ax2 = plt.gca()
    census.plot(ax=ax2)
    ax2.set_title("Censo diario UCI (pacientes presentes)")
    ax2.set_xlabel("Fecha"); ax2.set_ylabel("Pacientes")
    ax2.grid(True, axis="y", alpha=.3)
    plt.tight_layout(); plt.savefig(ASSETS_DIR / "census_daily.png"); plt.close(fig2)

    return ts, census

def distribution_plots(df: pd.DataFrame):
    plt.rcParams["figure.figsize"] = (8, 4)

    los = pd.to_numeric(df["los_final"], errors="coerce").dropna()
    if not los.empty:
        fig = plt.figure()
        ax = plt.gca()
        bins = range(0, int(max(1, los.max())) + 2)
        ax.hist(los, bins=bins)
        ax.set_title("Distribución de LOS (días)"); ax.set_xlabel("Días"); ax.set_ylabel("Pacientes")
        ax.grid(True, axis="y", alpha=.3)
        plt.tight_layout(); plt.savefig(ASSETS_DIR / "los_hist.png"); plt.close(fig)

    ap = pd.to_numeric(df["apache2"], errors="coerce").dropna()
    if not ap.empty:
        fig = plt.figure()
        ax = plt.gca()
        ax.boxplot(ap, vert=True, labels=["APACHE II (24 h)"])
        ax.set_title("APACHE II (24 h)"); ax.set_ylabel("Puntaje")
        ax.grid(True, axis="y", alpha=.3)
        plt.tight_layout(); plt.savefig(ASSETS_DIR / "apache_box.png"); plt.close(fig)

    so = pd.to_numeric(df["sofa48"], errors="coerce").dropna()
    if not so.empty:
        fig = plt.figure()
        ax = plt.gca()
        ax.boxplot(so, vert=True, labels=["SOFA 48 h"])
        ax.set_title("SOFA a 48 h"); ax.set_ylabel("Puntaje")
        ax.grid(True, axis="y", alpha=.3)
        plt.tight_layout(); plt.savefig(ASSETS_DIR / "sofa_box.png"); plt.close(fig)

def bar_plots(df: pd.DataFrame):
    plt.rcParams["figure.figsize"] = (8, 4)

    k = df["kpc_mbl"].fillna("").replace("", "No informado").value_counts().sort_values(ascending=False)
    if not k.empty:
        fig = plt.figure()
        ax = plt.gca()
        k.plot(kind="bar", ax=ax)
        ax.set_title("Estado KPC/MBL"); ax.set_ylabel("Pacientes")
        ax.grid(True, axis="y", alpha=.3)
        plt.tight_layout(); plt.savefig(ASSETS_DIR / "kpc_bars.png"); plt.close(fig)

    o = df["origen"].fillna("No informado").value_counts().head(8)
    if not o.empty:
        fig = plt.figure()
        ax = plt.gca()
        o.plot(kind="bar", ax=ax)
        ax.set_title("Casos por origen (Top 8)"); ax.set_ylabel("Pacientes")
        ax.grid(True, axis="y", alpha=.3)
        plt.tight_layout(); plt.savefig(ASSETS_DIR / "casemix_bars.png"); plt.close(fig)

# =========================
# KPIs y tablas
# =========================
def compute_kpis(df: pd.DataFrame) -> dict:
    n = len(df)
    egresos = df["fec_egr"].notna().sum()
    obitos = (df["cond_egreso"] == "Óbito").sum()
    mort_egresos = safe_pct(obitos, egresos)
    mort_admisiones = safe_pct(obitos, n)

    los = pd.to_numeric(df["los_final"], errors="coerce")
    los_med, los_q1, los_q3 = median_iqr(los)
    los_mean = float(los.mean()) if los.notna().any() else None

    ap_med, ap_q1, ap_q3 = median_iqr(df["apache2"])
    so_med, so_q1, so_q3 = median_iqr(df["sofa48"])

    vi_rate = to_bool(df["vi"]).mean(skipna=True) if "vi" in df else np.nan
    vi_rate = float(vi_rate * 100) if pd.notna(vi_rate) else None

    vvc_per100 = safe_pct(df["vvc"].fillna(0).sum(), n)
    hd_per100 = safe_pct(df["cateter_hd"].fillna(0).sum(), n)
    la_per100 = safe_pct(df["lineas_art"].fillna(0).sum(), n)
    ecg_prom_pt = float(df["ecg"].fillna(0).sum()) / n if n else 0.0

    periodo_ini = df["fec_ing"].min()
    periodo_fin = df["fec_egr"].max() if df["fec_egr"].notna().any() else df["fec_ing"].max()

    return {
        "admisiones": n,
        "egresos": int(egresos),
        "obitos": int(obitos),
        "mort_sobre_egresos": mort_egresos,
        "mort_sobre_admisiones": mort_admisiones,
        "los_mediana": los_med, "los_q1": los_q1, "los_q3": los_q3, "los_media": los_mean,
        "apache_med": ap_med, "apache_q1": ap_q1, "apache_q3": ap_q3,
        "sofa_med": so_med, "sofa_q1": so_q1, "sofa_q3": so_q3,
        "vi_rate": vi_rate,
        "vvc_per100": vvc_per100, "hd_per100": hd_per100, "la_per100": la_per100,
        "ecg_prom_pt": ecg_prom_pt,
        "periodo_ini": periodo_ini, "periodo_fin": periodo_fin
    }

def group_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    # Por médico
    g = df.groupby("medico", dropna=False)
    casos_med = g.size().rename("Casos")
    obitos_med = g["cond_egreso"].apply(lambda s: (s == "Óbito").sum()).rename("Óbitos")
    egresos_med = g["fec_egr"].apply(lambda s: s.notna().sum())
    mort_med = (obitos_med / egresos_med.replace(0, np.nan) * 100).rename("Mort.%")
    los_med = g["los_final"].median().rename("LOS_med")
    ap_med = g["apache2"].median().rename("APACHE_med")
    so_med = g["sofa48"].median().rename("SOFA48_med")
    tab_med = pd.concat([casos_med, obitos_med, mort_med, los_med, ap_med, so_med], axis=1)\
                .sort_values(["Óbitos","Casos"], ascending=[False, False])

    # Por origen
    g2 = df.groupby("origen", dropna=False)
    casos_org = g2.size().rename("Casos")
    obitos_org = g2["cond_egreso"].apply(lambda s: (s == "Óbito").sum()).rename("Óbitos")
    egresos_org = g2["fec_egr"].apply(lambda s: s.notna().sum())
    mort_org = (obitos_org / egresos_org.replace(0, np.nan) * 100).rename("Mort.%")
    los_org = g2["los_final"].median().rename("LOS_med")
    tab_origen = pd.concat([casos_org, obitos_org, mort_org, los_org], axis=1)\
                   .sort_values("Casos", ascending=False).head(10)

    # Por tipo
    g3 = df.groupby("tipo", dropna=False)
    casos_tipo = g3.size().rename("Casos")
    obitos_tipo = g3["cond_egreso"].apply(lambda s: (s == "Óbito").sum()).rename("Óbitos")
    egresos_tipo = g3["fec_egr"].apply(lambda s: s.notna().sum())
    mort_tipo = (obitos_tipo / egresos_tipo.replace(0, np.nan) * 100).rename("Mort.%")
    los_tipo = g3["los_final"].median().rename("LOS_med")
    tab_tipo = pd.concat([casos_tipo, obitos_tipo, mort_tipo, los_tipo], axis=1)\
                 .sort_values("Casos", ascending=False).head(10)

    # KPC/MBL
    tab_kpc = df["kpc_mbl"].fillna("No informado").value_counts().rename_axis("Estado").to_frame("Pacientes")

    # Redondeos y formateos de columnas numéricas
    for t in [tab_med, tab_origen, tab_tipo]:
        if "Mort.%" in t.columns:
            t["Mort.%"] = t["Mort.%"].astype(float).round(1)
        if "LOS_med" in t.columns:
            t["LOS_med"] = t["LOS_med"].astype(float).round(1)

    return {"por_medico": tab_med, "por_origen": tab_origen, "por_tipo": tab_tipo, "kpc": tab_kpc}

# =========================
# CSS y render Markdown
# =========================
CSS_CONTENT = """
:root{
  --bg:#0b1220; --panel:#0f172a; --ink:#e5e7eb; --muted:#9ca3af; --accent:#22c55e;
  --border:#1f2937; --accent2:#38bdf8; --warn:#f59e0b;
}
*{box-sizing:border-box}
html,body{margin:0;padding:0;background:var(--bg);color:var(--ink);font:16px/1.5 system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Cantarell,Noto Sans,Helvetica Neue,Arial}
.page{max-width:1120px;margin:0 auto;padding:32px 20px}
h1,h2,h3{color:#fff;letter-spacing:.1px}
h1{font-size:32px;margin:0 0 6px}
h2{font-size:22px;margin:28px 0 12px;border-bottom:1px solid var(--border);padding-bottom:6px}
h3{font-size:18px;margin:22px 0 10px}
.badgebar{display:flex;gap:12px;flex-wrap:wrap;margin:8px 0 18px}
.badge{background:var(--panel);border:1px solid var(--border);padding:6px 10px;border-radius:999px;color:var(--muted)}
.kpi-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:12px 0 8px}
.kpi{background:var(--panel);border:1px solid var(--border);border-radius:14px;padding:14px 16px}
.kpi .label{color:var(--muted);font-size:.85rem;margin-bottom:6px}
.kpi .value{font-size:1.6rem;font-weight:700;color:#fff}
.kpi .sub{color:var(--muted);font-size:.85rem;margin-top:6px}
.grid-2{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}
.card{background:var(--panel);border:1px solid var(--border);border-radius:14px;padding:12px}
figure{margin:0}
figure img{width:100%;display:block;border-radius:10px;border:1px solid var(--border)}
figcaption{color:var(--muted);font-size:.85rem;margin-top:6px}
.tablewrap{overflow:auto;border:1px solid var(--border);border-radius:12px}
table{width:100%;border-collapse:collapse;background:var(--panel)}
th,td{padding:10px 12px;border-bottom:1px solid var(--border);text-align:left}
thead th{position:sticky;top:0;background:#0e162a;color:#e2e8f0}
tbody tr:nth-child(odd){background:#0d1526}
.note{border-left:4px solid var(--accent2);padding:10px 12px;background:#0d1625;border-radius:8px;margin-top:8px;color:#cde1ff}
.toc{display:flex;gap:8px;flex-wrap:wrap;margin:12px 0 16px}
.toc a{color:var(--accent2);text-decoration:none;border:1px solid var(--border);border-radius:999px;padding:6px 10px}
@media (max-width:960px){
  .kpi-grid{grid-template-columns:repeat(2,minmax(0,1fr))}
  .grid-2{grid-template-columns:1fr}
}
"""

def fmt_dt(d) -> str:
    if d is None or pd.isna(d):
        return "-"
    return pd.to_datetime(d).strftime("%Y-%m-%d")

def write_css():
    (ASSETS_DIR / "report.css").write_text(CSS_CONTENT, encoding="utf-8")

def write_markdown(kpis: dict, tables: dict):
    # Asegura CSS
    write_css()

    # KPIs formateados
    k_adm = fmt_int(kpis['admisiones'])
    k_egr = fmt_int(kpis['egresos'])
    k_obt = fmt_int(kpis['obitos'])
    k_m_eg = fmt_pct(kpis['mort_sobre_egresos'])
    k_m_ad = fmt_pct(kpis['mort_sobre_admisiones'])
    k_los_med = fmt_float(kpis['los_mediana'])
    k_los_iqr = f"Q1 {fmt_float(kpis['los_q1'])} – Q3 {fmt_float(kpis['los_q3'])}"
    k_los_mean = fmt_float(kpis['los_media'])
    k_ap_med = fmt_float(kpis['apache_med'])
    k_ap_iqr = f"Q1 {fmt_float(kpis['apache_q1'])} – Q3 {fmt_float(kpis['apache_q3'])}"
    k_sf_med = fmt_float(kpis['sofa_med'])
    k_sf_iqr = f"Q1 {fmt_float(kpis['sofa_q1'])} – Q3 {fmt_float(kpis['sofa_q3'])}"
    k_vi = fmt_pct(kpis['vi_rate'])
    k_cvc = fmt_float(kpis['vvc_per100'])
    k_hd = fmt_float(kpis['hd_per100'])
    k_la = fmt_float(kpis['la_per100'])
    k_ecg = fmt_float(kpis['ecg_prom_pt'], nd=2)

    lines = []
    # Front matter para Jekyll
    lines += ["---", "title: Informe Operativo UCI", "layout: null", "---", ""]
    # Enlace a CSS (desde contenido, funciona en Pages)
    lines += ['<link rel="stylesheet" href="assets/report.css">', '']
    lines += ['<div class="page">']
    lines += [f'<h1>Informe Operativo UCI</h1>']
    lines += [f'<div class="badgebar"><span class="badge">Actualizado: {datetime.utcnow().strftime("%Y-%m-%d %H:%M")} UTC</span><span class="badge">Período: {fmt_dt(kpis["periodo_ini"])} → {fmt_dt(kpis["periodo_fin"])}</span></div>']

    # TOC
    lines += ['<div class="toc">']
    lines += ['<a href="#resumen-ejecutivo">Resumen</a>',
              '<a href="#dinamica-asistencial">Dinámica</a>',
              '<a href="#severidad-y-estancia">Severidad</a>',
              '<a href="#vigilancia-microbiologica">KPC/MBL</a>',
              '<a href="#casuistica-top">Casuística</a>']
    lines += ['</div>']

    # KPIs en tarjetas
    lines += ['<h2 id="resumen-ejecutivo">Resumen ejecutivo</h2>']
    lines += ['<div class="kpi-grid">']
    lines += [f'<div class="kpi"><div class="label">Admisiones</div><div class="value">{k_adm}</div></div>']
    lines += [f'<div class="kpi"><div class="label">Egresos</div><div class="value">{k_egr}</div></div>']
    lines += [f'<div class="kpi"><div class="label">Óbitos</div><div class="value">{k_obt}</div><div class="sub">Mort./egresos {k_m_eg} · Mort./adm {k_m_ad}</div></div>']
    lines += [f'<div class="kpi"><div class="label">Vent. invasiva</div><div class="value">{k_vi}</div></div>']
    lines += [f'<div class="kpi"><div class="label">LOS mediana</div><div class="value">{k_los_med} d</div><div class="sub">{k_los_iqr} · media {k_los_mean}</div></div>']
    lines += [f'<div class="kpi"><div class="label">APACHE II 24 h</div><div class="value">{k_ap_med}</div><div class="sub">{k_ap_iqr}</div></div>']
    lines += [f'<div class="kpi"><div class="label">SOFA 48 h</div><div class="value">{k_sf_med}</div><div class="sub">{k_sf_iqr}</div></div>']
    lines += [f'<div class="kpi"><div class="label">Dispositivos/100 adm</div><div class="value">{k_cvc} CVC</div><div class="sub">HD {k_hd} · Líneas {k_la} · ECG/pt {k_ecg}</div></div>']
    lines += ['</div>']  # kpi-grid

    # Gráficos: dinámica
    lines += ['<h2 id="dinamica-asistencial">Dinámica asistencial</h2>']
    lines += ['<div class="grid-2">']
    lines += ['<div class="card"><figure><img src="assets/timeseries_adm_disc.png" alt="Admisiones y egresos"><figcaption>Admisiones y egresos diarios</figcaption></figure></div>']
    lines += ['<div class="card"><figure><img src="assets/census_daily.png" alt="Censo diario"><figcaption>Censo diario UCI</figcaption></figure></div>']
    lines += ['</div>']

    # Severidad y estancia
    lines += ['<h2 id="severidad-y-estancia">Severidad y estancia</h2>']
    lines += ['<div class="grid-2">']
    if (ASSETS_DIR / "los_hist.png").exists():
        lines += ['<div class="card"><figure><img src="assets/los_hist.png" alt="Distribución LOS"><figcaption>Distribución de LOS (días)</figcaption></figure></div>']
    if (ASSETS_DIR / "apache_box.png").exists():
        lines += ['<div class="card"><figure><img src="assets/apache_box.png" alt="APACHE II"><figcaption>APACHE II a 24 h</figcaption></figure></div>']
    if (ASSETS_DIR / "sofa_box.png").exists():
        lines += ['<div class="card"><figure><img src="assets/sofa_box.png" alt="SOFA 48 h"><figcaption>SOFA a 48 h</figcaption></figure></div>']
    lines += ['</div>']

    # Vigilancia
    lines += ['<h2 id="vigilancia-microbiologica">Vigilancia microbiológica</h2>']
    if (ASSETS_DIR / "kpc_bars.png").exists():
        lines += ['<div class="card"><figure><img src="assets/kpc_bars.png" alt="KPC/MBL"><figcaption>Distribución por estado KPC/MBL</figcaption></figure></div>']

    # Casuística Top + tablas
    lines += ['<h2 id="casuistica-top">Casuística (Top)</h2>']
    if (ASSETS_DIR / "casemix_bars.png").exists():
        lines += ['<div class="card"><figure><img src="assets/casemix_bars.png" alt="Origen Top"><figcaption>Pacientes por origen (Top 8)</figcaption></figure></div>']

    # Tablas: por médico / origen / tipo / KPC
    # Formateo final (porcentajes a 1 dec, LOS a 1 dec)
    def _format_table(df: pd.DataFrame, perc_cols=("Mort.%",), dec_cols=("LOS_med", "APACHE_med", "SOFA48_med")) -> pd.DataFrame:
        out = df.copy()
        for c in out.columns:
            if c in perc_cols:
                out[c] = out[c].astype(float).round(1).map(lambda v: fmt_pct(v))
            elif c in dec_cols and c in out:
                out[c] = out[c].astype(float).round(1).map(lambda v: fmt_float(v))
            elif pd.api.types.is_integer_dtype(out[c]) or pd.api.types.is_float_dtype(out[c]):
                out[c] = out[c].map(fmt_int)
        return out

    tab_med = _format_table(tables["por_medico"].reset_index().rename(columns={"medico":"Médico"}))
    tab_org = _format_table(tables["por_origen"].reset_index().rename(columns={"origen":"Origen"}), dec_cols=("LOS_med",))
    tab_tip = _format_table(tables["por_tipo"].reset_index().rename(columns={"tipo":"Tipo"}), dec_cols=("LOS_med",))
    tab_kpc = tables["kpc"].reset_index().copy()
    if "Pacientes" in tab_kpc.columns:
        tab_kpc["Pacientes"] = tab_kpc["Pacientes"].map(fmt_int)

    lines += ['<h3>Por médico tratante</h3>','<div class="tablewrap">', md_table(tab_med, index=False), '</div>']
    lines += ['<h3>Por origen del paciente (Top 10)</h3>','<div class="tablewrap">', md_table(tab_org, index=False), '</div>']
    lines += ['<h3>Por tipo de paciente (Top 10)</h3>','<div class="tablewrap">', md_table(tab_tip, index=False), '</div>']
    lines += ['<h3>KPC/MBL</h3>','<div class="tablewrap">', md_table(tab_kpc, index=False), '</div>']

    # Nota metodológica
    lines += [f'<div class="note"><strong>Notas metodológicas:</strong> Mortalidad sin ajuste por gravedad ni case-mix. '
              f'El indicador “Mort./egresos” usa el número de egresos como denominador. LOS recalculado cuando faltan días en la hoja.</div>']

    lines += ['</div>']  # fin .page

    (REPORT_DIR / "index.md").write_text("\n".join(lines), encoding="utf-8")

# =========================
# Main
# =========================
def main():
    df_raw = load_data()
    df = prepare(df_raw)
    timeseries_and_census(df)
    distribution_plots(df)
    bar_plots(df)
    kpis = compute_kpis(df)
    tables = group_tables(df)
    write_markdown(kpis, tables)

if __name__ == "__main__":
    main()
