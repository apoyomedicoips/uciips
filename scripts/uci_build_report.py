# scripts/uci_build_report.py
from __future__ import annotations
import os, json, base64, unicodedata
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "report"
ASSETS_DIR = REPORT_DIR / "assets"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

SHEET_CSV_URL = os.getenv("SHEET_CSV_URL", "").strip()
GSHEETS_CREDENTIALS_B64 = os.getenv("GSHEETS_CREDENTIALS_B64", "").strip()
GSHEET_ID = os.getenv("GSHEET_ID", "").strip()
GSHEET_TAB = os.getenv("GSHEET_TAB", "base")
TIMEZONE = os.getenv("TZ", "UTC")

# ---------- util ----------
def _accent_fold(s: str) -> str:
    if s is None:
        return ""
    return "".join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c))

def parse_date_series(sr: pd.Series) -> pd.Series:
    s = sr.astype(str).str.replace("\u00A0", " ").str.strip().replace({"": pd.NA, "NaT": pd.NA, "nan": pd.NA})
    dt = pd.to_datetime(s, dayfirst=True, errors="coerce")
    def _fix(y):
        if pd.isna(y):
            return pd.NaT
        if 1000 <= y.year <= 1100:
            try: return y.replace(year=y.year + 1000)
            except Exception: return y
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

# ---------- load ----------
def load_from_csv_url(url: str) -> pd.DataFrame:
    if not url:
        raise RuntimeError("SHEET_CSV_URL vacío.")
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
    raise RuntimeError("Configura SHEET_CSV_URL o las variables de Service Account.")

# ---------- normalize ----------
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
    if not isinstance(x, str): return ""
    t = _accent_fold(x).lower().strip().rstrip(":")
    if "obito" in t or "óbito" in t: return "Óbito"
    if "alta" in t: return "Alta a piso"
    return x.strip().rstrip(":")

def canon_kpc(x: str) -> str:
    if not isinstance(x, str): return ""
    t = _accent_fold(x).lower()
    if "negativo" in t: return "Negativo"
    if "pendiente retorno" in t or "pendiente hr" in t: return "Pendiente HR ingreso"
    if "prevalencia" in t: return "HR de Prevalencia"
    if "ingreso" in t: return "HR de Ingreso"
    if "portador" in t or "plasmido" in t or "mdr" in t: return "Conocido portador MDR"
    return x

def canon_servicio(x: str) -> str:
    if not isinstance(x, str): return ""
    t = _accent_fold(x).strip()
    repl = {"Traumatologia":"Traumatología","Urologia":"Urología","Mastologia":"Mastología",
            "IPS INTERIOR":"IPS Interior","Reanimacion":"Reanimación","Clinica Medica":"Clínica Médica"}
    return repl.get(t, x)

def prepare(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.rename(columns=COLMAP).copy()
    for col in ["marca_temporal","fec_nac","fec_ing","fec_egr"]:
        if col in df: df[col] = parse_date_series(df[col])
    for col in ["edad","apache2","sofa48","vvc","cateter_hd","lineas_art","ecg","los","reg_intern","prontuario"]:
        if col in df: df[col] = to_int(df[col])
    for col in ["vi","tubo_dren","traqueo","caf","pocus","doppler_tc","fibro"]:
        if col in df: df[col] = to_bool(df[col])
    if "cond_egreso" in df: df["cond_egreso"] = df["cond_egreso"].astype(str).map(canon_outcome)
    if "kpc_mbl" in df: df["kpc_mbl"] = df["kpc_mbl"].astype(str).map(canon_kpc)
    if "origen" in df: df["origen"] = df["origen"].astype(str).map(canon_servicio)
    if "tipo" in df: df["tipo"] = df["tipo"].astype(str).str.strip()
    if "medico" in df: df["medico"] = df["medico"].astype(str).str.strip()
    if "fec_ing" in df and "fec_egr" in df:
        los_calc = (df["fec_egr"] - df["fec_ing"]).dt.days
        df["los_calc"] = los_calc.where(los_calc >= 0)
    df["los_final"] = df.get("los").fillna(df.get("los_calc"))
    df = df[df["fec_ing"].notna()].copy()
    return df

# ---------- payload (JSON) ----------
def export_payload(df: pd.DataFrame) -> dict:
    def to_iso(d):
        if pd.isna(d): return None
        return pd.to_datetime(d).date().isoformat()
    def yesno(x):
        if pd.isna(x): return ""
        return "Sí" if bool(x) else "No"
    records = []
    cols = ["fec_ing","fec_egr","medico","origen","tipo","cond_egreso","kpc_mbl","vi","los_final","apache2","sofa48"]
    for _, r in df[cols].iterrows():
        records.append({
            "fec_ing": to_iso(r["fec_ing"]),
            "fec_egr": to_iso(r["fec_egr"]),
            "medico": (r["medico"] or "").strip(),
            "origen": (r["origen"] or "").strip(),
            "tipo": (r["tipo"] or "").strip(),
            "cond_egreso": (r["cond_egreso"] or "").strip(),
            "kpc": (r["kpc_mbl"] or "").strip(),
            "vi": yesno(r["vi"]),
            "los": None if pd.isna(r["los_final"]) else int(r["los_final"]),
            "apache2": None if pd.isna(r["apache2"]) else int(r["apache2"]),
            "sofa48": None if pd.isna(r["sofa48"]) else int(r["sofa48"]),
        })
    payload = {
        "updated": datetime.utcnow().strftime("%Y-%m-%d %H:%M") + " UTC",
        "timezone": TIMEZONE,
        "n": len(records),
        "records": records
    }
    # Guardamos para diagnóstico (puede fallar si .gitignore bloquea, pero no afecta el tablero)
    (ASSETS_DIR / "data.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return payload

# ---------- HTML ----------
HTML_TMPL = r"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>UCI · Tablero interactivo</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<script src="https://cdn.plot.ly/plotly-2.33.0.min.js"></script>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
<style>
:root{ --primary:#0ea5e9; --secondary:#8b5cf6; --txt:#e5e7eb; --muted:#93a4b8 }
html,body{height:100%}
body{
  font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, 'Noto Sans', Helvetica, Arial, sans-serif;
  background: radial-gradient(1200px 600px at 20% -10%, #1f3b68 0%, #0b1020 55%, #0b0f18 100%);
  color:var(--txt);
}
.hdr{background:linear-gradient(135deg,var(--primary),var(--secondary));color:#fff;border-radius:16px;padding:16px 18px;margin:12px 0;
display:flex;align-items:center;justify-content:space-between;gap:16px}
.glass{background:linear-gradient(180deg,rgba(255,255,255,.06),rgba(255,255,255,.02));backdrop-filter:blur(10px);border:1px solid rgba(255,255,255,.08);border-radius:14px;box-shadow:0 10px 30px rgba(0,0,0,.35)}
.card-kpi{ padding:12px 14px;border-radius:14px;display:flex;gap:12px;align-items:center }
.k-val{ font-size:1.35rem; font-weight:800 }
.k-lbl{ color:#c6d0e0; font-size:.9rem }
.btn-clear{ color:#fff; border-color: rgba(255,255,255,.45)}
.btn-clear:hover{ background: rgba(255,255,255,.1)}
.ctrl .form-select,.ctrl .form-control{ background: rgba(255,255,255,.92) }
.badge-filter{ background:rgba(255,255,255,.12); border:1px solid rgba(255,255,255,.25)}
.empty{ border:1px dashed rgba(255,255,255,.35); border-radius:12px; padding:16px; color:#cfe3ff; background:rgba(255,255,255,.04) }
a,a:hover{color:#9ecbff}
</style>
</head>
<body>
<div class="container py-3">

  <div class="hdr">
    <div>
      <h1 class="h5 m-0">UCI · Tablero interactivo</h1>
      <div class="small opacity-75">Filtra con controles o clic en barras/sectores • doble clic para quitar zoom local.</div>
    </div>
    <div class="text-end">
      <div id="updated" class="small">Actualizado: —</div>
      <button id="btnResetAll" class="btn btn-outline-light btn-sm btn-clear mt-2">Limpiar filtros</button>
    </div>
  </div>

  <div class="glass p-3 mb-3 ctrl">
    <div class="row g-2 align-items-end">
      <div class="col-12 col-md-3">
        <label class="form-label mb-1">Rango de fechas (ingreso)</label>
        <div class="d-flex gap-2">
          <input id="fIni" type="date" class="form-control form-control-sm">
          <input id="fFin" type="date" class="form-control form-control-sm">
        </div>
        <div class="d-flex gap-1 mt-2 flex-wrap">
          <button class="btn btn-sm btn-outline-secondary" data-quick="30">⏱️ 30 días</button>
          <button class="btn btn-sm btn-outline-secondary" data-quick="90">90 días</button>
          <button class="btn btn-sm btn-outline-secondary" data-quick="365">12 meses</button>
          <button class="btn btn-sm btn-outline-secondary" data-quick="all">Todo</button>
        </div>
      </div>
      <div class="col-12 col-md-3">
        <label class="form-label mb-1">Médico tratante</label>
        <select id="fMed" class="form-select form-select-sm"><option value="">Todos</option></select>
      </div>
      <div class="col-12 col-md-3">
        <label class="form-label mb-1">Origen del paciente</label>
        <select id="fOrg" class="form-select form-select-sm"><option value="">Todos</option></select>
      </div>
      <div class="col-12 col-md-3">
        <label class="form-label mb-1">Tipo de paciente</label>
        <select id="fTipo" class="form-select form-select-sm"><option value="">Todos</option></select>
      </div>
      <div class="col-12 mt-2 d-flex gap-3 align-items-center">
        <div class="form-check"><input id="fObitos" class="form-check-input" type="checkbox"><label class="form-check-label">Solo Óbitos</label></div>
        <div class="form-check"><input id="fVI" class="form-check-input" type="checkbox"><label class="form-check-label">Solo VI = Sí</label></div>
        <span id="activeFilters" class="badge rounded-pill badge-filter ms-auto d-none">—</span>
      </div>
    </div>
  </div>

  <div class="row g-2 mb-1">
    <div class="col-6 col-md-3"><div class="glass card-kpi"><div>👣</div><div><div class="k-val" id="k_adm">0</div><div class="k-lbl">Admisiones</div></div></div></div>
    <div class="col-6 col-md-3"><div class="glass card-kpi"><div>🚪</div><div><div class="k-val" id="k_egr">0</div><div class="k-lbl">Egresos</div></div></div></div>
    <div class="col-6 col-md-3"><div class="glass card-kpi"><div>🖤</div><div><div class="k-val" id="k_ob">0</div><div class="k-lbl">Óbitos</div></div></div></div>
    <div class="col-6 col-md-3"><div class="glass card-kpi"><div>📉</div><div><div class="k-val" id="k_mort">0%</div><div class="k-lbl">Mortalidad / egresos</div></div></div></div>
  </div>
  <div class="row g-2 mb-3">
    <div class="col-6 col-md-3"><div class="glass card-kpi"><div>🕒</div><div><div class="k-val" id="k_los_med">—</div><div class="k-lbl">LOS (mediana)</div></div></div></div>
    <div class="col-6 col-md-3"><div class="glass card-kpi"><div>📋</div><div><div class="k-val" id="k_ap_med">—</div><div class="k-lbl">APACHE II (mediana)</div></div></div></div>
    <div class="col-6 col-md-3"><div class="glass card-kpi"><div>❤️‍🔥</div><div><div class="k-val" id="k_sofa_med">—</div><div class="k-lbl">SOFA 48h (mediana)</div></div></div></div>
    <div class="col-6 col-md-3"><div class="glass card-kpi"><div>🫁</div><div><div class="k-val" id="k_vi">—</div><div class="k-lbl">Vent. invasiva</div></div></div></div>
  </div>

  <div id="noData" class="empty d-none mb-3"><b>No hay datos con “Fecha de Ingreso” válida.</b><div class="small">Revisa que <code>assets/data.json</code> exista y tenga registros.</div></div>

  <div class="row g-3">
    <div class="col-12"><div class="glass p-2"><div id="g_ts"   style="height:280px"></div></div></div>
    <div class="col-md-4"><div class="glass p-2"><div id="g_med"  style="height:320px"></div></div></div>
    <div class="col-md-4"><div class="glass p-2"><div id="g_org"  style="height:320px"></div></div></div>
    <div class="col-md-4"><div class="glass p-2"><div id="g_tipo" style="height:320px"></div></div></div>
    <div class="col-md-4"><div class="glass p-2"><div id="g_cond" style="height:300px"></div></div></div>
    <div class="col-md-4"><div class="glass p-2"><div id="g_los"  style="height:300px"></div></div></div>
    <div class="col-md-4"><div class="glass p-2"><div id="g_kpc"  style="height:300px"></div></div></div>
  </div>

  <div class="text-end mt-3"><a href="assets/data.json" target="_blank" rel="noreferrer">Ver JSON</a></div>
</div>

<!-- DATA EMBEBIDA -->
<script type="application/json" id="PAYLOAD">__PAYLOAD_JSON__</script>

<script>
const cfg = {displayModeBar:false, responsive:true};
let RAW = []; let FILTERED = [];
const S = {dateFrom:null,dateTo:null,medico:"",origen:"",tipo:"",cond:"",obitosOnly:false,viOnly:false};

function median(arr){ const v=arr.filter(x=>Number.isFinite(x)).slice().sort((a,b)=>a-b); if(!v.length) return null; const m=Math.floor(v.length/2); return v.length%2?v[m]:(v[m-1]+v[m])/2;}
function fmtPct(x){ return x==null ? "—" : (x.toFixed(1)+"%"); }
function setText(id,t){ const el=document.getElementById(id); if(el) el.textContent=(t==null||t==="")?"—":t; }
function uniqSorted(arr){ return [...new Set(arr.filter(x=>x&&x.trim()))].sort((a,b)=>a.localeCompare(b,'es',{sensitivity:'base'}));}

function applyFilters(){
  FILTERED = RAW.filter(r=>{
    if(!r.fec_ing) return false;
    const d = new Date(r.fec_ing+"T00:00:00Z");
    if(S.dateFrom && d < new Date(S.dateFrom+"T00:00:00Z")) return false;
    if(S.dateTo   && d > new Date(S.dateTo+"T00:00:00Z")) return false;
    if(S.medico && r.medico !== S.medico) return false;
    if(S.origen && r.origen !== S.origen) return false;
    if(S.tipo   && r.tipo   !== S.tipo)   return false;
    if(S.cond   && r.cond_egreso !== S.cond) return false;
    if(S.obitosOnly && r.cond_egreso !== "Óbito") return false;
    if(S.viOnly && r.vi !== "Sí") return false;
    return true;
  });
  const badge = document.getElementById('activeFilters'); const parts=[];
  if(S.dateFrom||S.dateTo) parts.push(`Fecha: ${S.dateFrom||'…'} → ${S.dateTo||'…'}`);
  if(S.medico) parts.push(`Médico: ${S.medico}`);
  if(S.origen) parts.push(`Origen: ${S.origen}`);
  if(S.tipo) parts.push(`Tipo: ${S.tipo}`);
  if(S.cond) parts.push(`Cond: ${S.cond}`);
  if(S.obitosOnly) parts.push("Solo óbitos");
  if(S.viOnly) parts.push("Solo VI=Sí");
  if(parts.length){ badge.textContent=parts.join(" · "); badge.classList.remove('d-none'); } else { badge.classList.add('d-none'); }
}

function kpis(){
  const n=FILTERED.length;
  const eg=FILTERED.filter(d=>!!d.fec_egr).length;
  const ob=FILTERED.filter(d=>d.cond_egreso==="Óbito").length;
  const mort=eg?(ob*100/eg):0;
  const losArr=FILTERED.map(d=>Number.isFinite(d.los)?d.los:NaN);
  const apArr =FILTERED.map(d=>Number.isFinite(d.apache2)?d.apache2:NaN);
  const soArr =FILTERED.map(d=>Number.isFinite(d.sofa48)?d.sofa48:NaN);
  const viPct=n?(FILTERED.filter(d=>d.vi==="Sí").length*100/n):0;
  setText('k_adm',n); setText('k_egr',eg); setText('k_ob',ob); setText('k_mort',fmtPct(mort));
  const mlos=median(losArr); setText('k_los_med', mlos==null?'—':mlos.toFixed(1));
  const map =median(apArr);  setText('k_ap_med',  map==null?'—':map.toFixed(1));
  const mso =median(soArr);  setText('k_sofa_med',mso==null?'—':mso.toFixed(1));
  setText('k_vi', n?fmtPct(viPct):'—');
}

function groupCount(arr, key, topN=null){
  const m=new Map(); for(const r of arr){ const v=(r[key]||'—').trim()||'—'; m.set(v,(m.get(v)||0)+1); }
  let a=[...m.entries()].sort((x,y)=>y[1]-x[1]); if(topN) a=a.slice(0,topN);
  return { labels:a.map(x=>x[0]), values:a.map(x=>x[1]) };
}
function timeSeries(arr){ const m=new Map(); for(const r of arr){ if(!r.fec_ing) continue; m.set(r.fec_ing,(m.get(r.fec_ing)||0)+1); } const dates=[...m.keys()].sort(); return {x:dates,y:dates.map(d=>m.get(d))}; }

function buildCharts(){
  const ts=timeSeries(FILTERED);
  Plotly.react('g_ts', [{x:ts.x,y:ts.y,type:'scatter',mode:'lines+markers',fill:'tozeroy',name:'Admisiones'}],
    {margin:{l:40,r:10,t:10,b:30},xaxis:{rangeslider:{visible:true}},yaxis:{title:"Admisiones/día"},paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)'},{displayModeBar:false,responsive:true});
  const med=groupCount(FILTERED,'medico',10);
  Plotly.react('g_med',[{x:med.values.reverse(),y:med.labels.slice().reverse(),type:'bar',orientation:'h'}],
    {margin:{l:120,r:20,t:10,b:30},xaxis:{title:'Casos'},paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)'},{displayModeBar:false,responsive:true});
  const org=groupCount(FILTERED,'origen',10);
  Plotly.react('g_org',[{x:org.values.reverse(),y:org.labels.slice().reverse(),type:'bar',orientation:'h'}],
    {margin:{l:120,r:20,t:10,b:30},xaxis:{title:'Casos'},paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)'},{displayModeBar:false,responsive:true});
  const tip=groupCount(FILTERED,'tipo',10);
  Plotly.react('g_tipo',[{x:tip.values.reverse(),y:tip.labels.slice().reverse(),type:'bar',orientation:'h'}],
    {margin:{l:120,r:20,t:10,b:30},xaxis:{title:'Casos'},paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)'},{displayModeBar:false,responsive:true});
  const cond=groupCount(FILTERED,'cond_egreso');
  Plotly.react('g_cond',[{labels:cond.labels,values:cond.values,type:'pie',hole:.45}],
    {margin:{l:10,r:10,t:10,b:10},legend:{orientation:'h'},paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)'},{displayModeBar:false,responsive:true});
  const los=FILTERED.map(d=>Number.isFinite(d.los)?d.los:null).filter(x=>Number.isFinite(x));
  const maxLos=Math.max(1,...los,1);
  Plotly.react('g_los',[{x:los,type:'histogram',xbins:{start:0,end:maxLos,size:1}}],
    {margin:{l:40,r:10,t:10,b:30},xaxis:{title:'Días'},yaxis:{title:'Pacientes'},paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)'},{displayModeBar:false,responsive:true});
  const kpc=groupCount(FILTERED,'kpc');
  Plotly.react('g_kpc',[{x:kpc.values.reverse(),y:kpc.labels.slice().reverse(),type:'bar',orientation:'h'}],
    {margin:{l:120,r:20,t:10,b:30},xaxis:{title:'Pacientes'},paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)'},{displayModeBar:false,responsive:true});
}

function attachInteractions(){
  document.getElementById('g_med').on('plotly_click',ev=>{ const label=ev.points?.[0]?.y; if(!label) return; document.getElementById('fMed').value=label; S.medico=label; refreshAll(); });
  document.getElementById('g_org').on('plotly_click',ev=>{ const label=ev.points?.[0]?.y; if(!label) return; document.getElementById('fOrg').value=label; S.origen=label; refreshAll(); });
  document.getElementById('g_tipo').on('plotly_click',ev=>{ const label=ev.points?.[0]?.y; if(!label) return; document.getElementById('fTipo').value=label; S.tipo=label; refreshAll(); });
  document.getElementById('g_cond').on('plotly_click',ev=>{ const label=ev.points?.[0]?.label; if(!label) return; S.cond=(S.cond===label)?"":label; refreshAll(); });
  document.getElementById('g_ts').on('plotly_relayout',ev=>{
    const a=ev['xaxis.range[0]']||ev['xaxis.range']?.[0];
    const b=ev['xaxis.range[1]']||ev['xaxis.range']?.[1];
    if(a&&b){ S.dateFrom=a.slice(0,10); S.dateTo=b.slice(0,10); document.getElementById('fIni').value=S.dateFrom; document.getElementById('fFin').value=S.dateTo; refreshAll(); }
  });
  document.getElementById('fMed').addEventListener('change',e=>{ S.medico=e.target.value; refreshAll(); });
  document.getElementById('fOrg').addEventListener('change',e=>{ S.origen=e.target.value; refreshAll(); });
  document.getElementById('fTipo').addEventListener('change',e=>{ S.tipo=e.target.value; refreshAll(); });
  document.getElementById('fObitos').addEventListener('change',e=>{ S.obitosOnly=e.target.checked; refreshAll(); });
  document.getElementById('fVI').addEventListener('change',e=>{ S.viOnly=e.target.checked; refreshAll(); });
  document.getElementById('fIni').addEventListener('change',e=>{ S.dateFrom=e.target.value||null; refreshAll(); });
  document.getElementById('fFin').addEventListener('change',e=>{ S.dateTo=e.target.value||null; refreshAll(); });
  document.querySelectorAll('[data-quick]').forEach(btn=>{
    btn.addEventListener('click',()=>{
      const q=btn.getAttribute('data-quick');
      if(q==='all'){ S.dateFrom=null; S.dateTo=null; document.getElementById('fIni').value=''; document.getElementById('fFin').value=''; }
      else{
        const dates=[...new Set(RAW.map(r=>r.fec_ing))].sort();
        if(!dates.length){ refreshAll(); return; }
        const maxd=dates[dates.length-1];
        const from=new Date(maxd); from.setDate(from.getDate()-parseInt(q,10));
        S.dateFrom=from.toISOString().slice(0,10); S.dateTo=maxd;
        document.getElementById('fIni').value=S.dateFrom; document.getElementById('fFin').value=S.dateTo;
      }
      refreshAll();
    });
  });
  document.getElementById('btnResetAll').addEventListener('click',()=>{
    S.dateFrom=S.dateTo=null; S.medico=S.origen=S.tipo=S.cond=""; S.obitosOnly=S.viOnly=false;
    ['fIni','fFin','fMed','fOrg','fTipo'].forEach(id=>{ const el=document.getElementById(id); if(el.tagName==='SELECT') el.value=''; else el.value=''; });
    document.getElementById('fObitos').checked=false; document.getElementById('fVI').checked=false;
    refreshAll();
  });
}

function setSelectOptions(id, values){
  const sel=document.getElementById(id); const keep=sel.value;
  sel.innerHTML='<option value="">Todos</option>'+values.map(v=>`<option>${v}</option>`).join('');
  if(values.includes(keep)) sel.value=keep;
}

function refreshAll(){ applyFilters(); document.getElementById('noData').classList.toggle('d-none', FILTERED.length>0); kpis(); buildCharts(); }

function tryInitFromEmbedded(){
  try{
    const raw = document.getElementById('PAYLOAD').textContent;
    if(!raw) return null;
    return JSON.parse(raw);
  }catch(e){ console.warn('No payload embebido', e); return null; }
}

async function bootstrap(){
  let payload = tryInitFromEmbedded();
  if(!payload){
    // Fallback: intentar assets/data.json por si existe
    try{
      const r=await fetch('assets/data.json'); if(r.ok){ payload=await r.json(); }
    }catch(e){ console.error(e); }
  }
  if(!payload || !payload.records){ document.getElementById('noData').classList.remove('d-none'); return; }
  RAW = payload.records;
  document.getElementById('updated').textContent = "Actualizado: " + (payload.updated || "—");
  setSelectOptions('fMed',  [...new Set(RAW.map(r=>r.medico))].filter(Boolean).sort());
  setSelectOptions('fOrg',  [...new Set(RAW.map(r=>r.origen))].filter(Boolean).sort());
  setSelectOptions('fTipo', [...new Set(RAW.map(r=>r.tipo))].filter(Boolean).sort());
  // rango por defecto: últimos 180 días
  const dates=[...new Set(RAW.map(r=>r.fec_ing))].filter(Boolean).sort();
  if(dates.length){
    const maxd=dates[dates.length-1]; const from=new Date(maxd); from.setDate(from.getDate()-180);
    S.dateFrom=from.toISOString().slice(0,10); S.dateTo=maxd;
    document.getElementById('fIni').value=S.dateFrom; document.getElementById('fFin').value=S.dateTo;
  }
  attachInteractions(); refreshAll();
}
bootstrap();
</script>
</body>
</html>
"""

def write_plotly_html(payload: dict):
    html = HTML_TMPL.replace("__PAYLOAD_JSON__", json.dumps(payload, ensure_ascii=False))
    (REPORT_DIR / "index.html").write_text(html, encoding="utf-8")

# ---------- main ----------
def main():
    df_raw = load_data()
    df = prepare(df_raw)
    payload = export_payload(df)   # guarda report/assets/data.json y devuelve dict
    write_plotly_html(payload)     # genera report/index.html con DATA embebida

if __name__ == "__main__":
    main()
