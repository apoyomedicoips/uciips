# scripts/uci_build_report.py
from __future__ import annotations
import os
from pathlib import Path
import base64, json, re, unicodedata
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
# CSV público de Google Sheets (pestaña 'base'):
SHEET_CSV_URL = os.getenv("SHEET_CSV_URL", "").strip()
# Si prefieres Service Account (opcional):
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

def md_table(df: pd.DataFrame, index=False) -> str:
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
    adm = df.groupby(df["fec_ing"].dt.date).size().rename("Admisiones").to_frame()
    dis = df[df["fec_egr"].notna()].groupby(df["fec_egr"].dt.date).size().rename("Egresos").to_frame()
    start = min(df["fec_ing"].min(), df["fec_egr"].min() if df["fec_egr"].notna().any() else df["fec_ing"].min())
    end = max(df["fec_egr"].max() if df["fec_egr"].notna().any() else df["fec_ing"].max(), df["fec_ing"].max())
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
    ts.plot(ax=plt.gca())
    plt.title("Admisiones y Egresos diarios"); plt.xlabel("Fecha"); plt.ylabel("Conteo")
    plt.tight_layout(); plt.savefig(ASSETS_DIR / "timeseries_adm_disc.png", dpi=150); plt.close(fig1)

    fig2 = plt.figure()
    census.plot(ax=plt.gca())
    plt.title("Censo diario UCI (pacientes presentes)"); plt.xlabel("Fecha"); plt.ylabel("Pacientes")
    plt.tight_layout(); plt.savefig(ASSETS_DIR / "census_daily.png", dpi=150); plt.close(fig2)

    return ts, census

def distribution_plots(df: pd.DataFrame):
    los = pd.to_numeric(df["los_final"], errors="coerce").dropna()
    if not los.empty:
        fig = plt.figure()
        plt.hist(los, bins=range(0, int(max(1, los.max())) + 2))
        plt.title("Distribución de LOS (días)"); plt.xlabel("Días"); plt.ylabel("Pacientes")
        plt.tight_layout(); plt.savefig(ASSETS_DIR / "los_hist.png", dpi=150); plt.close(fig)

    ap = pd.to_numeric(df["apache2"], errors="coerce").dropna()
    if not ap.empty:
        fig = plt.figure()
        plt.boxplot(ap, vert=True, labels=["APACHE II (24 h)"])
        plt.title("APACHE II (24 h)"); plt.ylabel("Puntaje")
        plt.tight_layout(); plt.savefig(ASSETS_DIR / "apache_box.png", dpi=150); plt.close(fig)

    so = pd.to_numeric(df["sofa48"], errors="coerce").dropna()
    if not so.empty:
        fig = plt.figure()
        plt.boxplot(so, vert=True, labels=["SOFA 48 h"])
        plt.title("SOFA a 48 h"); plt.ylabel("Puntaje")
        plt.tight_layout(); plt.savefig(ASSETS_DIR / "sofa_box.png", dpi=150); plt.close(fig)

def bar_plots(df: pd.DataFrame):
    k = df["kpc_mbl"].fillna("").replace("", "No informado").value_counts().sort_values(ascending=False)
    if not k.empty:
        fig = plt.figure()
        k.plot(kind="bar", ax=plt.gca())
        plt.title("Estado KPC/MBL"); plt.ylabel("Pacientes")
        plt.tight_layout(); plt.savefig(ASSETS_DIR / "kpc_bars.png", dpi=150); plt.close(fig)

    o = df["origen"].fillna("No informado").value_counts().head(8)
    if not o.empty:
        fig = plt.figure()
        o.plot(kind="bar", ax=plt.gca())
        plt.title("Casos por origen (Top 8)"); plt.ylabel("Pacientes")
        plt.tight_layout(); plt.savefig(ASSETS_DIR / "casemix_bars.png", dpi=150); plt.close(fig)

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
    g = df.groupby("medico", dropna=False)
    tab_med = pd.DataFrame({
        "Casos": g.size(),
        "Óbitos": g.apply(lambda x: (x["cond_egreso"]=="Óbito").sum()),
        "Mort.%": g.apply(lambda x: safe_pct((x["cond_egreso"]=="Óbito").sum(), x["fec_egr"].notna().sum())),
        "LOS_med": g["los_final"].median(),
        "APACHE_med": g["apache2"].median(),
        "SOFA48_med": g["sofa48"].median()
    }).sort_values(["Óbitos","Casos"], ascending=[False, False])

    g2 = df.groupby("origen", dropna=False)
    tab_origen = pd.DataFrame({
        "Casos": g2.size(),
        "Óbitos": g2.apply(lambda x: (x["cond_egreso"]=="Óbito").sum()),
        "Mort.%": g2.apply(lambda x: safe_pct((x["cond_egreso"]=="Óbito").sum(), x["fec_egr"].notna().sum())),
        "LOS_med": g2["los_final"].median()
    }).sort_values("Casos", ascending=False).head(10)

    g3 = df.groupby("tipo", dropna=False)
    tab_tipo = pd.DataFrame({
        "Casos": g3.size(),
        "Óbitos": g3.apply(lambda x: (x["cond_egreso"]=="Óbito").sum()),
        "Mort.%": g3.apply(lambda x: safe_pct((x["cond_egreso"]=="Óbito").sum(), x["fec_egr"].notna().sum())),
        "LOS_med": g3["los_final"].median()
    }).sort_values("Casos", ascending=False).head(10)

    tab_kpc = df["kpc_mbl"].fillna("No informado").value_counts().rename_axis("Estado").to_frame("Pacientes")

    return {"por_medico": tab_med, "por_origen": tab_origen, "por_tipo": tab_tipo, "kpc": tab_kpc}

# =========================
# Render Markdown (Jekyll)
# =========================
def fmt_dt(d) -> str:
    if d is None or pd.isna(d):
        return "-"
    return pd.to_datetime(d).strftime("%Y-%m-%d")

def write_markdown(kpis: dict, tables: dict):
    lines = []
    # Front matter para que Jekyll convierta a HTML sin layout
    lines += ["---", "title: Informe Operativo UCI", "layout: null", "---", ""]
    lines += [ "# Informe Operativo UCI", "" ]
    lines += [ f"_Actualizado: {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} (UTC)_" , "" ]
    lines += [ f"**Período:** {fmt_dt(kpis['periodo_ini'])} → {fmt_dt(kpis['periodo_fin'])}", "" ]

    lines += ["## Resumen ejecutivo", ""]
    lines += [f"- **Admisiones:** {kpis['admisiones']}"]
    lines += [f"- **Egresos:** {kpis['egresos']}"]
    lines += [f"- **Óbitos:** {kpis['obitos']}  ·  **Mortalidad/egresos:** {kpis['mort_sobre_egresos']:.1f}%  ·  **Mortalidad/admisiones:** {kpis['mort_sobre_admisiones']:.1f}%"]
    lines += [f"- **LOS (días):** mediana {kpis['los_mediana'] or 0:.1f}  (Q1 {kpis['los_q1'] or 0:.1f} – Q3 {kpis['los_q3'] or 0:.1f})  ·  media {kpis['los_media'] or 0:.1f}"]
    lines += [f"- **APACHE II (24 h):** mediana {kpis['apache_med'] or 0:.1f}  (Q1 {kpis['apache_q1'] or 0:.1f} – Q3 {kpis['apache_q3'] or 0:.1f})"]
    lines += [f"- **SOFA 48 h:** mediana {kpis['sofa_med'] or 0:.1f}  (Q1 {kpis['sofa_q1'] or 0:.1f} – Q3 {kpis['sofa_q3'] or 0:.1f})"]
    vi_rate = 0.0 if kpis["vi_rate"] is None else kpis["vi_rate"]
    lines += [f"- **Ventilación invasiva:** {vi_rate:.1f}% de los pacientes"]
    lines += [f"- **Dispositivos/100 adm.:** CVC {kpis['vvc_per100']:.1f} · HD {kpis['hd_per100']:.1f} · Líneas art. {kpis['la_per100']:.1f} · ECG/paciente {kpis['ecg_prom_pt']:.2f}", ""]

    lines += ["## Dinámica asistencial", ""]
    lines += ["![Admisiones y egresos](assets/timeseries_adm_disc.png)", ""]
    lines += ["![Censo diario](assets/census_daily.png)", ""]

    lines += ["## Severidad y estancia", ""]
    if (ASSETS_DIR / "los_hist.png").exists():
        lines += ["![Distribución LOS](assets/los_hist.png)", ""]
    if (ASSETS_DIR / "apache_box.png").exists():
        lines += ["![APACHE II (24 h)](assets/apache_box.png)", ""]
    if (ASSETS_DIR / "sofa_box.png").exists():
        lines += ["![SOFA a 48 h](assets/sofa_box.png)", ""]

    lines += ["## Vigilancia microbiológica", ""]
    if (ASSETS_DIR / "kpc_bars.png").exists():
        lines += ["![KPC/MBL](assets/kpc_bars.png)", ""]

    lines += ["## Casuística (Top)", ""]
    if (ASSETS_DIR / "casemix_bars.png").exists():
        lines += ["![Origen del paciente](assets/casemix_bars.png)", ""]

    lines += ["### Por médico tratante", md_table(tables["por_medico"].reset_index().rename(columns={"medico":"Médico"}), index=False), ""]
    lines += ["### Por origen del paciente (Top 10)", md_table(tables["por_origen"].reset_index().rename(columns={"origen":"Origen"}), index=False), ""]
    lines += ["### Por tipo de paciente (Top 10)", md_table(tables["por_tipo"].reset_index().rename(columns={"tipo":"Tipo"}), index=False), ""]
    lines += ["### KPC/MBL", md_table(tables["kpc"].reset_index(), index=False), ""]
    lines += ["> **Notas:** Tasas no ajustadas por gravedad. Interpretar mortalidad por médico con cautela (case-mix).", ""]

    (REPORT_DIR / "index.md").write_text("\n".join(lines), encoding="utf-8")

# =========================
# Main
# =========================
def main():
    df_raw = load_data()
    df = prepare(df_raw)
    ts, census = timeseries_and_census(df)
    distribution_plots(df)
    bar_plots(df)
    kpis = compute_kpis(df)
    tables = group_tables(df)
    write_markdown(kpis, tables)

if __name__ == "__main__":
    main()

