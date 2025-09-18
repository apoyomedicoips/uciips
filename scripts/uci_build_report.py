# scripts/uci_build_report.py
from __future__ import annotations
import os
from pathlib import Path
import io
import base64
import json
import re
import unicodedata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# =========================
# Config
# =========================
ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT / "reports"
ASSETS_DIR = REPORTS_DIR / "assets"
DOCS_DIR = ROOT / "docs"

ASSETS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)

SHEET_CSV_URL = os.getenv("SHEET_CSV_URL", "").strip()
WRITE_TO_DOCS_INDEX = os.getenv("WRITE_TO_DOCS_INDEX", "false").lower() in {"1", "true", "yes"}

# Opcional (si usas Service Account en vez de CSV público):
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
    """Parsea fechas dd/mm/yyyy (o con hora), corrige años tipo 1025→2025."""
    s = sr.astype(str).str.strip()
    s = s.replace({"": pd.NA, "NaT": pd.NA, "nan": pd.NA})
    # Primer parseo
    dt = pd.to_datetime(s, dayfirst=True, errors="coerce")
    # Corrección de años mal tipeados (1000-1100 ~> +1000)
    def _fix(y):
        if pd.isna(y):
            return pd.NaT
        if y.year >= 1000 and y.year <= 1100:
            try:
                return y.replace(year=y.year + 1000)
            except Exception:
                return y
        return y
    return dt.apply(_fix)

def to_bool(sr: pd.Series) -> pd.Series:
    """Normaliza Sí/No (cubre SI/NO, yes/1; devuelve booleano o NA)."""
    s = sr.astype(str).str.strip().str.lower()
    si = {"si", "sí", "yes", "y", "1", "verdadero", "true"}
    no = {"no", "n", "0", "falso", "false"}
    out = pd.Series(pd.NA, index=sr.index, dtype="boolean")
    out = out.mask(s.isin(si), True)
    out = out.mask(s.isin(no), False)
    return out

def to_int(sr: pd.Series) -> pd.Series:
    return pd.to_numeric(sr, errors="coerce").astype("Int64")

def median_iqr(x: pd.Series) -> tuple[float|None, float|None, float|None]:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if x.empty:
        return None, None, None
    return float(x.median()), float(x.quantile(0.25)), float(x.quantile(0.75))

def safe_pct(num: float, den: float) -> float:
    return float(num) / float(den) * 100.0 if den else 0.0

def md_table(df: pd.DataFrame, index=False) -> str:
    try:
        return df.to_markdown(index=index)
    except Exception:
        # Fallback muy simple
        cols = list(df.columns)
        rows = [ "| " + " | ".join(map(str, cols)) + " |",
                 "|" + "|".join("---" for _ in cols) + "|" ]
        for _, r in df.iterrows():
            rows.append("| " + " | ".join(map(lambda v: "" if pd.isna(v) else str(v), r.tolist())) + " |")
        return "\n".join(rows)

# =========================
# Carga de datos
# =========================
def load_from_csv_url(url: str) -> pd.DataFrame:
    if not url:
        raise RuntimeError("SHEET_CSV_URL vacío. Defínelo en el workflow o usa Service Account.")
    df = pd.read_csv(url, dtype=str)
    return df

def load_from_gsheets_service_account(b64_json: str, sheet_id: str, tab: str="base") -> pd.DataFrame:
    import gspread
    from google.oauth2.service_account import Credentials
    if not b64_json or not sheet_id:
        raise RuntimeError("Faltan credenciales o GSHEET_ID para Service Account.")
    info = json.loads(base64.b64decode(b64_json).decode("utf-8"))
    scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
    creds = Credentials.from_service_account_info(info, scopes=scopes)
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(sheet_id)
    ws = sh.worksheet(tab)
    records = ws.get_all_records()
    return pd.DataFrame.from_records(records)

def load_data() -> pd.DataFrame:
    if SHEET_CSV_URL:
        return load_from_csv_url(SHEET_CSV_URL)
    if GSHEETS_CREDENTIALS_B64 and GSHEET_ID:
        return load_from_gsheets_service_account(GSHEETS_CREDENTIALS_B64, GSHEET_ID, GSHEET_TAB)
    raise RuntimeError("No se configuró una fuente de datos. Usa SHEET_CSV_URL o Service Account.")

# =========================
# Limpieza y normalización
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
        "Traumatologia":"Traumatología",
        "Urologia":"Urología",
        "Mastologia":"Mastología",
        "IPS INTERIOR":"IPS Interior",
        "Cx Gral Piso":"Cx Gral Piso",
        "Cx Gral Urgencias":"Cx Gral Urgencias",
        "Reanimacion":"Reanimación",
        "Urgencias":"Urgencias",
        "Clinica Medica":"Clínica Médica",
        "Neurocx":"Neurocx",
        "Urologia Urgencias":"Urología Urgencias",
        "Mastologia Programada":"Mastología Programada",
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

    # Outcome y KPC
    if "cond_egreso" in df.columns:
        df["cond_egreso"] = df["cond_egreso"].astype(str).map(canon_outcome)
    if "kpc_mbl" in df.columns:
        df["kpc_mbl"] = df["kpc_mbl"].astype(str).map(canon_kpc)

    # Origen / Tipo / Médico
    if "origen" in df.columns:
        df["origen"] = df["origen"].astype(str).map(canon_servicio)
    if "tipo" in df.columns:
        df["tipo"] = df["tipo"].astype(str).str.strip()
    if "medico" in df.columns:
        df["medico"] = df["medico"].astype(str).str.strip()

    # Recalcular LOS si falta o es negativo
    if "fec_ing" in df.columns and "fec_egr" in df.columns:
        los_calc = (df["fec_egr"] - df["fec_ing"]).dt.days
        df["los_calc"] = los_calc.where(los_calc >= 0)
    if "los" in df.columns:
        df["los_final"] = df["los"].fillna(df.get("los_calc"))
    else:
        df["los_final"] = df.get("los_calc")

    # Filtrar filas sin fecha de ingreso válida
    df = df[df["fec_ing"].notna()].copy()

    return df

# =========================
# Métricas y figuras
# =========================
def make_timeseries(df: pd.DataFrame):
    adm = df.groupby(df["fec_ing"].dt.date).size().rename("Admisiones").to_frame()
    dis = df[df["fec_egr"].notna()].groupby(df["fec_egr"].dt.date).size().rename("Egresos").to_frame()
    idx = pd.date_range(
        min(df["fec_ing"].min(), df["fec_egr"].min() if df["fec_egr"].notna().any() else df["fec_ing"].min()),
        max(df["fec_egr"].max() if df["fec_egr"].notna().any() else df["fec_ing"].max(), df["fec_ing"].max()),
        freq="D"
    ).date
    ts = pd.DataFrame(index=idx)
    ts["Admisiones"] = adm.reindex(idx).fillna(0).astype(int)
    ts["Egresos"] = dis.reindex(idx).fillna(0).astype(int)

    # Censo diario (pacientes en UCI)
    census = pd.Series(0, index=pd.Index(idx, name="Fecha"), dtype=int)
    for _, r in df.iterrows():
        start = r["fec_ing"].date()
        end = (r["fec_egr"].date() if pd.notna(r["fec_egr"]) else r["fec_ing"].date())
        # incluir ambos extremos
        for d in pd.date_range(start, end, freq="D").date:
            if d in census.index:
                census.loc[d] = int(census.loc[d]) + 1

    # Fig 1: Admisiones/Egresos
    fig1 = plt.figure()
    ts.plot(ax=plt.gca())
    plt.title("Admisiones y Egresos diarios")
    plt.xlabel("Fecha"); plt.ylabel("Conteo")
    (ASSETS_DIR / "timeseries_adm_disc.png").unlink(missing_ok=True)
    plt.tight_layout(); plt.savefig(ASSETS_DIR / "timeseries_adm_disc.png", dpi=150); plt.close(fig1)

    # Fig 2: Censo diario
    fig2 = plt.figure()
    census.plot(ax=plt.gca())
    plt.title("Censo diario UCI (pacientes presentes)")
    plt.xlabel("Fecha"); plt.ylabel("Pacientes")
    (ASSETS_DIR / "census_daily.png").unlink(missing_ok=True)
    plt.tight_layout(); plt.savefig(ASSETS_DIR / "census_daily.png", dpi=150); plt.close(fig2)

    return ts, census

def make_distribution_plots(df: pd.DataFrame):
    # LOS
    los = pd.to_numeric(df["los_final"], errors="coerce").dropna()
    fig = plt.figure()
    plt.hist(los, bins=range(0, int(max(los.max(), 1)) + 2))
    plt.title("Distribución de LOS (días)"); plt.xlabel("Días"); plt.ylabel("Pacientes")
    (ASSETS_DIR / "los_hist.png").unlink(missing_ok=True)
    plt.tight_layout(); plt.savefig(ASSETS_DIR / "los_hist.png", dpi=150); plt.close(fig)

    # APACHE
    ap = pd.to_numeric(df["apache2"], errors="coerce").dropna()
    if not ap.empty:
        fig = plt.figure()
        plt.boxplot(ap, vert=True, labels=["APACHE II"])
        plt.title("APACHE II (24 h)"); plt.ylabel("Puntaje")
        (ASSETS_DIR / "apache_box.png").unlink(missing_ok=True)
        plt.tight_layout(); plt.savefig(ASSETS_DIR / "apache_box.png", dpi=150); plt.close(fig)

    # SOFA 48
    so = pd.to_numeric(df["sofa48"], errors="coerce").dropna()
    if not so.empty:
        fig = plt.figure()
        plt.boxplot(so, vert=True, labels=["SOFA 48 h"])
        plt.title("SOFA a 48 h"); plt.ylabel("Puntaje")
        (ASSETS_DIR / "sofa_box.png").unlink(missing_ok=True)
        plt.tight_layout(); plt.savefig(ASSETS_DIR / "sofa_box.png", dpi=150); plt.close(fig)

def make_bar_plots(df: pd.DataFrame):
    # KPC/MBL
    k = df["kpc_mbl"].fillna("").replace("", "No informado")
    k_ct = k.value_counts().sort_values(ascending=False)
    if not k_ct.empty:
        fig = plt.figure()
        k_ct.plot(kind="bar", ax=plt.gca())
        plt.title("Estado KPC/MBL")
        plt.ylabel("Pacientes")
        (ASSETS_DIR / "kpc_bars.png").unlink(missing_ok=True)
        plt.tight_layout(); plt.savefig(ASSETS_DIR / "kpc_bars.png", dpi=150); plt.close(fig)

    # Casuística por origen (top 8)
    o = df["origen"].fillna("No informado")
    o_ct = o.value_counts().head(8)
    if not o_ct.empty:
        fig = plt.figure()
        o_ct.plot(kind="bar", ax=plt.gca())
        plt.title("Casos por origen (Top 8)")
        plt.ylabel("Pacientes")
        (ASSETS_DIR / "casemix_bars.png").unlink(missing_ok=True)
        plt.tight_layout(); plt.savefig(ASSETS_DIR / "casemix_bars.png", dpi=150); plt.close(fig)

# =========================
# Cálculos clave
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

    # Dispositivos por 100 admisiones
    vvc_per100 = safe_pct(df["vvc"].fillna(0).sum(), n)
    hd_per100 = safe_pct(df["cateter_hd"].fillna(0).sum(), n)
    la_per100 = safe_pct(df["lineas_art"].fillna(0).sum(), n)
    ecg_per_pt = float(df["ecg"].fillna(0).sum()) / n if n else 0.0

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
        "ecg_prom_pt": ecg_per_pt,
        "periodo_ini": periodo_ini, "periodo_fin": periodo_fin
    }

def make_group_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    # Por médico
    g = df.groupby("medico", dropna=False)
    tab_med = pd.DataFrame({
        "Casos": g.size(),
        "Óbitos": g.apply(lambda x: (x["cond_egreso"]=="Óbito").sum()),
        "Mort.%": g.apply(lambda x: safe_pct((x["cond_egreso"]=="Óbito").sum(), x["fec_egr"].notna().sum())),
        "LOS_med": g["los_final"].median(),
        "APACHE_med": g["apache2"].median(),
        "SOFA48_med": g["sofa48"].median()
    }).sort_values(["Óbitos","Casos"], ascending=[False, False])

    # Por origen (Top 10)
    g2 = df.groupby("origen", dropna=False)
    tab_origen = pd.DataFrame({
        "Casos": g2.size(),
        "Óbitos": g2.apply(lambda x: (x["cond_egreso"]=="Óbito").sum()),
        "Mort.%": g2.apply(lambda x: safe_pct((x["cond_egreso"]=="Óbito").sum(), x["fec_egr"].notna().sum())),
        "LOS_med": g2["los_final"].median()
    }).sort_values("Casos", ascending=False).head(10)

    # Por tipo (Top 10)
    g3 = df.groupby("tipo", dropna=False)
    tab_tipo = pd.DataFrame({
        "Casos": g3.size(),
        "Óbitos": g3.apply(lambda x: (x["cond_egreso"]=="Óbito").sum()),
        "Mort.%": g3.apply(lambda x: safe_pct((x["cond_egreso"]=="Óbito").sum(), x["fec_egr"].notna().sum())),
        "LOS_med": g3["los_final"].median()
    }).sort_values("Casos", ascending=False).head(10)

    # KPC/MBL
    tab_kpc = df["kpc_mbl"].fillna("No informado").value_counts().rename_axis("Estado").to_frame("Pacientes")

    return {
        "por_medico": tab_med,
        "por_origen": tab_origen,
        "por_tipo": tab_tipo,
        "kpc": tab_kpc
    }

# =========================
# Render MD
# =========================
def fmt_dt(d: pd.Timestamp|datetime|None) -> str:
    if d is None or pd.isna(d):
        return "-"
    if isinstance(d, pd.Timestamp):
        d = d.to_pydatetime()
    return d.strftime("%Y-%m-%d")

def write_markdown(kpis: dict, tables: dict, ts: pd.DataFrame):
    md = []
    md.append("# Informe Operativo UCI")
    md.append("")
    md.append(f"_Actualizado: {datetime.now().strftime('%Y-%m-%d %H:%M')} (UTC)_")
    md.append("")
    md.append(f"**Período:** {fmt_dt(kpis['periodo_ini'])} → {fmt_dt(kpis['periodo_fin'])}")
    md.append("")
    md.append("## Resumen ejecutivo")
    md.append("")
    md.append("- **Admisiones:** {admisiones}".format(**kpis))
    md.append("- **Egresos:** {egresos}".format(**kpis))
    md.append("- **Óbitos:** {obitos}  ·  **Mortalidad/egresos:** {0:.1f}%  ·  **Mortalidad/admisiones:** {1:.1f}%"
              .format(kpis["mort_sobre_egresos"], kpis["mort_sobre_admisiones"]))
    md.append("- **LOS (días):** mediana {los_mediana:.1f}  (Q1 {los_q1:.1f} – Q3 {los_q3:.1f})  ·  media {los_media:.1f}"
              .format(**{k: (0.0 if kpis[k] is None else kpis[k]) for k in ["los_mediana","los_q1","los_q3","los_media"]}))
    md.append("- **APACHE II (24 h):** mediana {apache_med:.1f}  (Q1 {apache_q1:.1f} – Q3 {apache_q3:.1f})".format(
        **{k: (0.0 if kpis[k] is None else kpis[k]) for k in ["apache_med","apache_q1","apache_q3"]}))
    md.append("- **SOFA 48 h:** mediana {sofa_med:.1f}  (Q1 {sofa_q1:.1f} – Q3 {sofa_q3:.1f})".format(
        **{k: (0.0 if kpis[k] is None else kpis[k]) for k in ["sofa_med","sofa_q1","sofa_q3"]}))
    md.append("- **Ventilación invasiva:** {0:.1f}% de los pacientes".format(0.0 if kpis["vi_rate"] is None else kpis["vi_rate"]))
    md.append("- **Dispositivos/100 adm.:** CVC {vvc_per100:.1f} · HD {hd_per100:.1f} · Líneas art. {la_per100:.1f} · ECG/paciente {ecg_prom_pt:.2f}"
              .format(**kpis))
    md.append("")
    md.append("## Dinámica asistencial")
    md.append("")
    md.append("![Admisiones y egresos](assets/timeseries_adm_disc.png)")
    md.append("")
    md.append("![Censo diario](assets/census_daily.png)")
    md.append("")
    md.append("## Severidad y estancia")
    md.append("")
    md.append("![Distribución LOS](assets/los_hist.png)")
    if (ASSETS_DIR / "apache_box.png").exists():
        md.append("")
        md.append("![APACHE II (24 h)](assets/apache_box.png)")
    if (ASSETS_DIR / "sofa_box.png").exists():
        md.append("")
        md.append("![SOFA a 48 h](assets/sofa_box.png)")
    md.append("")
    md.append("## Vigilancia microbiológica")
    md.append("")
    if (ASSETS_DIR / "kpc_bars.png").exists():
        md.append("![KPC/MBL](assets/kpc_bars.png)")
    md.append("")
    md.append("## Casuística (Top)")
    md.append("")
    if (ASSETS_DIR / "casemix_bars.png").exists():
        md.append("![Origen del paciente](assets/casemix_bars.png)")
    md.append("")
    md.append("### Por médico tratante")
    md.append(md_table(tables["por_medico"].reset_index().rename(columns={"medico":"Médico"}), index=False))
    md.append("")
    md.append("### Por origen del paciente (Top 10)")
    md.append(md_table(tables["por_origen"].reset_index().rename(columns={"origen":"Origen"}), index=False))
    md.append("")
    md.append("### Por tipo de paciente (Top 10)")
    md.append(md_table(tables["por_tipo"].reset_index().rename(columns={"tipo":"Tipo"}), index=False))
    md.append("")
    md.append("### KPC/MBL")
    md.append(md_table(tables["kpc"].reset_index(), index=False))
    md.append("")
    md.append("> **Notas:** Tasas no ajustadas por riesgo ni por gravedad. Interpretar mortalidad por médico con cautela (case-mix).")
    md.append("")

    out_md = "\n".join(md)
    (REPORTS_DIR / "uci_report.md").write_text(out_md, encoding="utf-8")

    if WRITE_TO_DOCS_INDEX:
        (DOCS_DIR / "index.md").write_text(out_md, encoding="utf-8")

# =========================
# Main
# =========================
def main():
    df_raw = load_data()
    df = prepare(df_raw)
    ts, census = make_timeseries(df)
    make_distribution_plots(df)
    make_bar_plots(df)
    kpis = compute_kpis(df)
    tables = make_group_tables(df)
    write_markdown(kpis, tables, ts)

if __name__ == "__main__":
    main()

