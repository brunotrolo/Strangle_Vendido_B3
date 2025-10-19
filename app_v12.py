# app_v10.py
# ------------------------------------------------------------
# Strangle Vendido Coberto — v9
# Sprint 1: Indicador cobertura, Presets (largura dinâmica), "Por que #1?"
# Sprint 2: Comparador de cenários (2 colunas) + Checklist de saída guiada
# Alinhamento horizontal no comparador (tabela combinada + cards pareados por rank)
# Multi-vencimentos + Top N + defaults cobertura (1000)
# parse_pasted_chain ignora título "Opções ..." e segunda linha vazia
# ------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import requests
import yfinance as yf
import io
import re
from bs4 import BeautifulSoup
from datetime import datetime, date
import math

# -------------------------
# Configuração básica
# -------------------------
st.set_page_config(page_title="Strangle Vendido Coberto — v9", page_icon="💼", layout="wide")

# CSS para legibilidade
st.markdown("""
<style>
.big-title {font-size:1.15rem; font-weight:700; margin: 0 0 .25rem 0;}
.small-help {color:var(--text-color-secondary, #6b7280); font-size:.95rem; margin: 0 0 .5rem 0;}
.strike-card{padding:.75rem 1rem; border:1px solid; border-radius:12px;}
.strike-label{font-size:.95rem; margin-bottom:.15rem; opacity:.85;}
.strike-value{font-size:1.6rem; font-weight:800;}
.badge {display:inline-block; padding:4px 8px; border-radius:999px; font-weight:700; font-size:.85rem;}
.badge-green {background:#10B98122; color:#065F46; border:1px solid #10B98166;}
.badge-amber {background:#F59E0B22; color:#7C2D12; border:1px solid #F59E0B66;}
.badge-red {background:#EF444422; color:#7F1D1D; border:1px solid #EF444466;}
.note {color:#6b7280; font-size:.9rem;}
.kv {background:#f3f4f6; padding:2px 6px; border-radius:6px; font-family:ui-monospace, SFMono-Regular, Menlo, monospace;}
.scenario-title {font-size:1.05rem; font-weight:800; margin:.25rem 0 .5rem;}
.card-placeholder {padding:1rem; border:1px dashed #cbd5e1; border-radius:12px; color:#64748b; text-align:center;}
@media (prefers-color-scheme: light) {
  .strike-card{ background:#fafafa; border-color:#e5e7eb; }
  .strike-label{ color:#4b5563; }
  .strike-value{ color:#111827; }
}
@media (prefers-color-scheme: dark) {
  .strike-card{ background:#111827; border-color:#374151; }
  .strike-label{ color:#d1d5db; }
  .strike-value{ color:#f9fafb; }
  .kv {background:#111827; border:1px solid #374151; color:#e5e5e5;}
  .card-placeholder {border-color:#334155; color:#94a3b8;}
}
</style>
""", unsafe_allow_html=True)

CONTRACT_SIZE = 100   # padrão B3
CACHE_TTL = 300       # 5 min
COMB_LIMIT_DEFAULT = 30  # limite interno (não aparece na UI)

# -------------------------
# Utilitários
# -------------------------
def br_to_float(s: str):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return np.nan
    if isinstance(s, (int, float)):
        return float(s)
    s = str(s).strip()
    if s == "":
        return np.nan
    s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except Exception:
        return np.nan

def pct_to_float(s: str):
    val = br_to_float(s)
    return val / 100.0 if pd.notna(val) else np.nan

def parse_date_br(d: str):
    if pd.isna(d):
        return None
    d = str(d).strip()
    try:
        return datetime.strptime(d, "%d/%m/%Y").date()
    except Exception:
        try:
            return datetime.strptime(d, "%Y-%m-%d").date()
        except Exception:
            return None

def format_brl(x: float):
    if pd.isna(x):
        return "—"
    return f"R$ {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def yahoo_symbol_from_b3(ticker_b3: str):
    t = (ticker_b3 or "").strip().upper()
    if not t.endswith(".SA"):
        t = t + ".SA"
    return t

# -------------------------
# Black–Scholes helpers
# -------------------------
SQRT_2 = math.sqrt(2.0)
def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / SQRT_2))

def d1_d2(S, K, r, sigma, T):
    if S <= 0 or K <= 0 or sigma <= 0 or T <= 0:
        return (np.nan, np.nan)
    try:
        d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        return d1, d2
    except Exception:
        return (np.nan, np.nan)

def prob_ITM_call(S, K, r, sigma, T):
    _, d2 = d1_d2(S, K, r, sigma, T)
    return norm_cdf(d2) if not np.isnan(d2) else np.nan

def prob_ITM_put(S, K, r, sigma, T):
    _, d2 = d1_d2(S, K, r, sigma, T)
    return norm_cdf(-d2) if not np.isnan(d2) else np.nan

# -------------------------
# Cache: lista de tickers (dadosdemercado)
# -------------------------
@st.cache_data(ttl=CACHE_TTL)
def fetch_b3_tickers():
    url = "https://www.dadosdemercado.com.br/acoes"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    try:
        dfs = pd.read_html(r.text)
        best = None
        for df in dfs:
            cols = [c.lower() for c in df.columns.astype(str)]
            if any("código" in c or "codigo" in c or "ticker" in c for c in cols) and any("empresa" in c or "razão" in c or "razao" in c or "nome" in c or "companhia" in c for c in cols):
                best = df
                break
        if best is None:
            best = dfs[0]
        best.columns = [str(c).strip() for c in best.columns]
        code_col = None
        name_col = None
        for c in best.columns:
            cl = c.lower()
            if code_col is None and ("código" in cl or "codigo" in cl or "ticker" in cl or "símbolo" in cl or "simbolo" in cl or cl=="cód."):
                code_col = c
            if name_col is None and ("empresa" in cl or "razão" in cl or "razao" in cl or "nome" in cl or "companhia" in cl):
                name_col = c
        if code_col is None:
            code_col = best.columns[0]
        if name_col is None:
            name_col = best.columns[1] if len(best.columns) > 1 else best.columns[0]
        out = best[[code_col, name_col]].copy()
        out.columns = ["ticker", "empresa"]
        out["ticker"] = out["ticker"].astype(str).str.strip().str.upper()
        out["empresa"] = out["empresa"].astype(str).str.strip()
        out = out[out["ticker"].str.match(r"^[A-Z]{3,5}\d{0,2}$")]
        out = out.drop_duplicates(subset=["ticker"]).reset_index(drop=True)
        return out
    except Exception:
        return pd.DataFrame(columns=["ticker", "empresa"])

# -------------------------
# Cache: preço yfinance + HV20 proxy
# -------------------------
@st.cache_data(ttl=CACHE_TTL)
def fetch_yf_price_and_hv20(y_ticker: str):
    try:
        info = yf.Ticker(y_ticker)
        hist = info.history(period="60d", interval="1d")
        price = np.nan
        if not hist.empty:
            if "Close" in hist.columns:
                price = float(hist["Close"].iloc[-1])
            elif "Adj Close" in hist.columns:
                price = float(hist["Adj Close"].iloc[-1])
        hv20 = np.nan
        if len(hist) >= 21:
            ret = hist["Close"].pct_change().dropna()
            if len(ret) >= 20:
                vol20 = ret.tail(20).std()
                hv20 = vol20 * math.sqrt(252.0) * 100.0  # em %
        return price, hv20
    except Exception:
        return np.nan, np.nan

# -------------------------
# Parsing da option chain colada
# -------------------------
def parse_pasted_chain(text: str):
    if not text or text.strip() == "":
        return pd.DataFrame()

    # Normaliza NBSP e recorta o título/linha em branco do opcoes.net
    raw = text.replace("\u00A0", " ").strip()

    # Remove 1ª (título "Opções ...") e, se existir, 2ª linha vazia
    lines = raw.splitlines()
    if len(lines) >= 1:
        first = lines[0].strip().lower()
        if ("opções" in first) or ("opcoes" in first):
            # Se a 2ª linha for vazia, remove também
            if len(lines) >= 2 and lines[1].strip() == "":
                lines = lines[2:]
            else:
                lines = lines[1:]
    raw = "\n".join(lines).strip()

    # Se não vier com TAB, tenta converter múltiplos espaços em TAB
    if "\t" not in raw:
        raw = re.sub(r"[ ]{2,}", "\t", raw)

    # Tenta ler como TSV (ou ; como fallback)
    try:
        df = pd.read_csv(io.StringIO(raw), sep="\t", engine="python")
    except Exception:
        try:
            df = pd.read_csv(io.StringIO(raw), sep=";", engine="python")
        except Exception:
            return pd.DataFrame()

    # Padroniza nomes de colunas
    cols = {c: c.strip() for c in df.columns}
    df.rename(columns=cols, inplace=True)

    def find_col(cands):
        for c in df.columns:
            lc = c.lower()
            for p in cands:
                if p in lc:
                    return c
        return None

    col_ticker = find_col(["ticker", "código", "codigo", "símbolo", "simbolo"])
    col_venc   = find_col(["venc", "vencimento"])
    col_tipo   = find_col(["tipo"])
    col_strike = find_col(["strike"])
    col_ultimo = find_col(["último", "ultimo", "last"])
    col_iv     = find_col(["vol. impl", "vol impl", "impl", "iv"])
    col_delta  = find_col(["delta"])

    if not all([col_ticker, col_venc, col_tipo, col_strike, col_ultimo]):
        return pd.DataFrame()

    out = pd.DataFrame()
    out["symbol"]     = df[col_ticker].astype(str).str.strip()
    out["type"]       = df[col_tipo].astype(str).str.upper().str.contains("CALL").map({True: "C", False: "P"})
    out["strike"]     = df[col_strike].apply(br_to_float)
    out["last"]       = df[col_ultimo].apply(br_to_float)
    out["expiration"] = df[col_venc].apply(parse_date_br)
    out["impliedVol"] = df[col_iv].apply(pct_to_float) if col_iv else np.nan
    out["delta"]      = df[col_delta].apply(br_to_float) if col_delta else np.nan

    out = out[pd.notna(out["strike"]) & pd.notna(out["expiration"])].copy()
    return out.reset_index(drop=True)

def business_days_between(d1: date, d2: date):
    if d1 is None or d2 is None:
        return np.nan
    if d2 < d1:
        return 0
    try:
        return np.busday_count(d1, d2)
    except Exception:
        return (d2 - d1).days

# -------------------------
# Layout principal
# -------------------------
st.title("💼 Strangle Vendido Coberto — v9")
st.caption("Cole a option chain do opcoes.net, escolha o(s) vencimento(s) e veja as sugestões didáticas de strangle coberto.")

# 1) Seleção de ticker
st.markdown('<div class="big-title">🔎 Selecione pelo nome da empresa ou ticker</div><div class="small-help">Digite para pesquisar por nome ou código.</div>', unsafe_allow_html=True)
tickers_df = fetch_b3_tickers()
if tickers_df.empty:
    st.warning("Não consegui carregar a lista de tickers agora. Digite o código manualmente no campo abaixo.")
    user_ticker = st.text_input("Ticker da B3", value="PETR4")
else:
    tickers_df["label"] = tickers_df["ticker"] + " — " + tickers_df["empresa"]
    default_idx = int((tickers_df["ticker"] == "PETR4").idxmax()) if "PETR4" in set(tickers_df["ticker"]) else 0
    sel_label = st.selectbox(" ", options=tickers_df["label"].tolist(),
                             index=default_idx if default_idx is not None else 0,
                             label_visibility="collapsed")
    sel_row = tickers_df.loc[tickers_df["label"] == sel_label].iloc[0]
    user_ticker = sel_row["ticker"]

# 2) Preço via yfinance (spot)
y_ticker = yahoo_symbol_from_b3(user_ticker)
spot, hv20_auto = fetch_yf_price_and_hv20(y_ticker)

strike_html = f"""
<div class="strike-card">
  <div class="strike-label">Preço à vista (yfinance)</div>
  <div class="strike-value">{format_brl(spot)}</div>
</div>
"""
st.markdown(strike_html, unsafe_allow_html=True)

# =========================
# 3) Sidebar: parâmetros essenciais
# =========================
st.sidebar.header("⚙️ Parâmetros & Cobertura")

# --- COBERTURA (defaults 1000 + number_input no caixa) ---
qty_shares = st.sidebar.number_input(
    f"Ações em carteira ({user_ticker})",
    min_value=0,
    max_value=1_000_000,
    value=1000,              # padrão 1000
    step=100,                # setinhas + / - pulam de 100 em 100
    help="Usado apenas para validar CALL coberta (✅/❌). Mais ações permitem mais lotes cobertos."
)

cash_avail = st.sidebar.number_input(
    f"Caixa disponível (R$) ({user_ticker})",
    min_value=0.0,
    value=1000.00,           # padrão 1000,00
    step=100.00,             # setinhas + / - de 100 em 100
    format="%.2f",
    help="Usado apenas para validar PUT coberta (✅/❌) no strike da PUT. Mais caixa permite mais lotes."
)
cash_avail_val = float(cash_avail)

contract_size = st.sidebar.number_input(
    f"Tamanho do contrato ({user_ticker})",
    1, 1000, CONTRACT_SIZE, step=1,
    help="Quantidade de ações por contrato. Aumenta proporcionalmente o prêmio total e a exigência de cobertura."
)

st.sidebar.markdown("---")

# PERFIL DE RISCO (define filtros e penalização)
st.sidebar.markdown("### 🧭 Perfil de risco")
risk_presets = {
    "Conservador": {"min_width": 0.08, "max_leg": 0.20, "max_comb": 0.15, "alpha": 3},
    "Neutro":      {"min_width": 0.06, "max_leg": 0.25, "max_comb": 0.20, "alpha": 2},
    "Agressivo":   {"min_width": 0.04, "max_leg": 0.35, "max_comb": 0.25, "alpha": 1},
}
risk_choice = st.sidebar.selectbox("Escolha um preset", options=list(risk_presets.keys()), index=1)

# ÚNICO CONTROLE EXTRA QUE FICA:
use_delta_filter = st.sidebar.checkbox(
    "Filtrar por |Δ| ~ 0,10–0,25 (se disponível)", value=True,
    help="Quando marcado, restringe as pernas a deltas típicos de OTM saudável. Reduz chance de exercício mantendo prêmio razoável."
)

# Exibição dos valores efetivos (informativo)
_p = risk_presets[risk_choice]
st.sidebar.caption(
    f"Valores efetivos → Prob. máx/Perna: {int(_p['max_leg']*100)}% · Prob. máx/Média: {int(_p['max_comb']*100)}% · α: {_p['alpha']} · Largura mín.: {int(_p['min_width']*100)}%"
)

# PARÂMETROS INTERNOS (não exibidos)
hv20_default = float(hv20_auto) if pd.notna(hv20_auto) else 20.0  # proxy automática
hv20_input = hv20_default
r_input = 11.0  # % anual (fixo interno)
comb_limit = COMB_LIMIT_DEFAULT

# ALERTAS (mantidos)
st.sidebar.markdown("---")
dias_alerta = st.sidebar.number_input(
    "Alerta de saída (dias para o vencimento) ≤",
    1, 30, 7,
    help="Mostra aviso de tempo quando faltar menos ou igual a este número de dias. Valores menores disparam alerta mais cedo."
)
meta_captura = st.sidebar.number_input(
    "Meta de captura do crédito (%)",
    50, 100, 75,
    help="Alvo didático para encerrar a operação com lucro. Valores maiores significam esperar capturar uma fração maior do crédito."
)
janela_pct = st.sidebar.number_input(
    "Janela de alerta no strike (±%)",
    1, 20, 5,
    help="Sensibilidade para avisos de 'encostar' no strike. Maior: mais avisos; menor: somente quando muito perto."
)

# 4) Colar a option chain (com formulário para mobile)
st.subheader(f"3) Colar a option chain de {user_ticker} (opcoes.net)")

if "pasted_chain" not in st.session_state:
    st.session_state["pasted_chain"] = ""

with st.form("chain_form", clear_on_submit=False):
    pasted_input = st.text_area(
        "Cole aqui a tabela (em celular: cole e toque em 'Confirmar')",
        value=st.session_state["pasted_chain"],
        height=220,
        help=("A tabela precisa conter: Ticker, Vencimento, Tipo (CALL/PUT), "
              "Strike, Último, (opcional) Vol. Impl. (%), Delta.")
    )
    c1, c2 = st.columns(2)
    confirm = c1.form_submit_button("✅ Confirmar")
    clear   = c2.form_submit_button("🧹 Limpar")

if confirm:
    st.session_state["pasted_chain"] = pasted_input
elif clear:
    st.session_state["pasted_chain"] = ""
    pasted_input = ""

pasted = st.session_state["pasted_chain"]

if not pasted.strip():
    st.info("Cole a tabela do opcoes.net e toque em **Confirmar** para continuar.")
    st.stop()

df_chain = parse_pasted_chain(pasted)
if df_chain.empty:
    st.error("Não consegui interpretar a tabela colada. Verifique se os títulos/colunas vieram corretamente e toque em **Confirmar** novamente.")
    st.stop()

# 5) Selecionar vencimentos (multi)
unique_exps = sorted([d for d in df_chain["expiration"].dropna().unique()])
if not unique_exps:
    st.error("Não identifiquei a coluna de Vencimento na tabela colada.")
    st.stop()

st.markdown("### 📅 Vencimento — escolha uma ou mais datas")
col_v1, col_v2 = st.columns([1, 1])
with col_v1:
    selected_exps = st.multiselect(
        "Datas disponíveis:",
        options=unique_exps,
        default=[unique_exps[0]] if len(unique_exps) > 0 else [],
        format_func=lambda d: d.strftime("%Y-%m-%d"),
        help="Selecione uma ou várias datas. Você também pode marcar 'Selecionar todos'."
    )
with col_v2:
    select_all = st.checkbox("Selecionar todos os vencimentos", value=False)
if select_all:
    selected_exps = unique_exps.copy()

if not selected_exps:
    st.warning("Selecione pelo menos um vencimento para continuar.")
    st.stop()

# ---------- Quantidade de recomendações (Top N) ----------
st.markdown("### 🔢 Quantidade de recomendações (Top N)")
top_n = st.slider(
    "Escolha quantas recomendações exibir (sempre a partir das melhores):",
    min_value=3, max_value=10, value=3, step=1
)

# ---------- Comparador de cenários ----------
st.markdown("### 🧪 Comparador de cenários")
risk_presets_all = list(risk_presets.keys())
compare_two = st.checkbox("Comparar 2 cenários (lado a lado)", value=False)
if compare_two:
    col_cmp1, col_cmp2 = st.columns(2)
    with col_cmp1:
        preset_left = st.selectbox("Preset (Lado A)", options=risk_presets_all, index=1, key="preset_left")
    with col_cmp2:
        preset_right = st.selectbox("Preset (Lado B)", options=risk_presets_all, index=2 if len(risk_presets_all)>=3 else 1, key="preset_right")
else:
    preset_left = risk_choice
    preset_right = None  # não usado

today = datetime.utcnow().date()
S = float(spot) if pd.notna(spot) and spot > 0 else df_chain["strike"].median()
r = 11.0 / 100.0  # fixo interno
sigma_proxy = (float(hv20_auto) if pd.notna(hv20_auto) else 20.0) / 100.0

# ------------- Funções auxiliares -------------
def near_strike(price, strike, pct):
    try:
        return abs(price - strike) <= (strike * (pct/100.0))
    except Exception:
        return False

def coverage_badge(covered_call: bool, covered_put: bool):
    if covered_call and covered_put:
        cls = "badge badge-green"; txt = "Cobertura: ✅ CALL · ✅ PUT"
    elif covered_call or covered_put:
        cls = "badge badge-amber"; txt = "Cobertura parcial"
    else:
        cls = "badge badge-red"; txt = "Sem cobertura"
    return f'<span class="{cls}">{txt}</span>'

def compute_recos_for_preset(preset_name: str):
    preset = risk_presets[preset_name]
    max_poe_leg  = preset["max_leg"]
    max_poe_comb = preset["max_comb"]
    alpha        = preset["alpha"]
    min_width_pct = preset["min_width"]

    all_pairs = []

    for this_exp in selected_exps:
        bus_days = business_days_between(today, this_exp)
        T_years  = float(bus_days) / 252.0 if pd.notna(bus_days) and bus_days > 0 else 1/252.0

        df = df_chain[df_chain["expiration"] == this_exp].copy().reset_index(drop=True)
        if df.empty:
            continue

        df["price"] = df["last"].astype(float)
        df["sigma"] = df["impliedVol"].fillna(sigma_proxy)

        calls = df[df["type"] == "C"].copy()
        puts  = df[df["type"] == "P"].copy()
        calls["OTM"] = calls["strike"].astype(float) > S
        puts["OTM"]  = puts["strike"].astype(float)  < S

        if use_delta_filter and "delta" in df.columns:
            calls = calls[calls["OTM"] & pd.notna(calls["price"]) & calls["delta"].abs().between(0.10, 0.25, inclusive="both")]
            puts  = puts[puts["OTM"]  & pd.notna(puts["price"])  & puts["delta"].abs().between(0.10, 0.25, inclusive="both")]
        else:
            calls = calls[calls["OTM"] & pd.notna(calls["price"])]
            puts  = puts[puts["OTM"]  & pd.notna(puts["price"])]

        if calls.empty or puts.empty:
            continue

        # PoE por perna
        puts["poe"]  = puts.apply(lambda rw: prob_ITM_put(S, float(rw["strike"]), r, float(rw["sigma"]) if pd.notna(rw["sigma"]) and rw["sigma"]>0 else sigma_proxy, T_years), axis=1)
        calls["poe"] = calls.apply(lambda rw: prob_ITM_call(S, float(rw["strike"]), r, float(rw["sigma"]) if pd.notna(rw["sigma"]) and rw["sigma"]>0 else sigma_proxy, T_years), axis=1)

        # Combinações controladas (internas)
        plim = int(COMB_LIMIT_DEFAULT)
        puts_small  = puts.sort_values(["price"], ascending=False).head(plim).copy()
        calls_small = calls.sort_values(["price"], ascending=False).head(plim).copy()

        pairs = []
        for _, prow in puts_small.iterrows():
            for _, crow in calls_small.iterrows():
                kp = float(prow["strike"]); kc = float(crow["strike"])
                if not (kp < S < kc):
                    continue
                prem_put  = float(prow["price"])
                prem_call = float(crow["price"])
                cred = prem_put + prem_call
                be_low  = kp - cred
                be_high = kc + cred
                poe_p = float(prow["poe"]) if pd.notna(prow["poe"]) else np.nan
                poe_c = float(crow["poe"]) if pd.notna(crow["poe"]) else np.nan
                pairs.append({
                    "expiration": this_exp,
                    "days_to_exp": bus_days,
                    "PUT": prow["symbol"],
                    "CALL": crow["symbol"],
                    "Kp": kp,
                    "Kc": kc,
                    "premio_put": prem_put,
                    "premio_call": prem_call,
                    "credito": cred,
                    "be_low": be_low,
                    "be_high": be_high,
                    "poe_put": poe_p,
                    "poe_call": poe_c,
                })
        if not pairs:
            continue

        pairs_df = pd.DataFrame(pairs)

        # Largura mínima entre strikes
        width_ok = (pairs_df["Kc"] - pairs_df["Kp"]) >= (S * min_width_pct)
        pairs_df = pairs_df[width_ok]
        if pairs_df.empty:
            continue

        # Filtros duros
        pairs_df["poe_leg_max"] = pairs_df[["poe_put","poe_call"]].max(axis=1)
        pairs_df["poe_comb"]    = pairs_df[["poe_put","poe_call"]].mean(axis=1)
        pairs_df = pairs_df[
            (pairs_df["poe_put"]  <= max_poe_leg) &
            (pairs_df["poe_call"] <= max_poe_leg) &
            (pairs_df["poe_comb"] <= max_poe_comb)
        ]
        if pairs_df.empty:
            continue

        # Score
        pairs_df["p_inside"] = (1 - pairs_df["poe_put"].fillna(0) - pairs_df["poe_call"].fillna(0)).clip(lower=0)
        pairs_df["score"] = pairs_df["credito"] * (pairs_df["p_inside"] ** preset["alpha"])

        all_pairs.append(pairs_df)

    if not all_pairs:
        return None, None, preset

    all_df = pd.concat(all_pairs, ignore_index=True)
    all_df = all_df.sort_values(["score","p_inside","credito"], ascending=[False, False, False]).reset_index(drop=True)
    top_df = all_df.head(top_n).copy()
    return top_df, all_df, preset

# ---------- Helpers p/ exibição alinhada ----------
def prep_display_table(df, preset):
    if df is None or df.empty:
        return pd.DataFrame()
    tmp = df.copy()
    tmp["Vencimento"] = tmp["expiration"].map(lambda d: d.strftime("%Y-%m-%d"))
    tmp["Strike PUT"] = tmp["Kp"].map(lambda x: f"{x:.2f}")
    tmp["Strike CALL"] = tmp["Kc"].map(lambda x: f"{x:.2f}")
    tmp["Prêmio PUT (R$)"]   = tmp["premio_put"].map(lambda x: f"{x:.2f}")
    tmp["Prêmio CALL (R$)"]  = tmp["premio_call"].map(lambda x: f"{x:.2f}")
    tmp["Crédito/ação (R$)"] = tmp["credito"].map(lambda x: f"{x:.2f}")
    tmp["Break-evens (mín–máx)"] = tmp.apply(lambda r: f"{r['be_low']:.2f} — {r['be_high']:.2f}", axis=1)
    tmp["Prob. exercício PUT (%)"]  = (100*tmp["poe_put"]).map(lambda x: f"{x:.1f}")
    tmp["Prob. exercício CALL (%)"] = (100*tmp["poe_call"]).map(lambda x: f"{x:.1f}")
    tmp["p_dentro (%)"] = (100*tmp["p_inside"]).map(lambda x: f"{x:.1f}")

    def tag_risco_row(r):
        tags = []
        if "poe_leg_max" in r and "poe_comb" in r:
            if r["poe_leg_max"] > (preset["max_leg"] * 0.9):
                tags.append("⚠️ perna alta")
            if r["poe_comb"] > (preset["max_comb"] * 0.9):
                tags.append("⚠️ média alta")
        if r["p_inside"] < 0.70:
            tags.append("🎯 dentro < 70%")
        return " · ".join(tags)

    tmp["Notas"] = df.apply(tag_risco_row, axis=1)
    sel = tmp[[
        "Vencimento","PUT","Strike PUT","CALL","Strike CALL",
        "Prêmio PUT (R$)","Prêmio CALL (R$)","Crédito/ação (R$)",
        "Break-evens (mín–máx)",
        "Prob. exercício PUT (%)","Prob. exercício CALL (%)",
        "p_dentro (%)",
        "Notas"
    ]].copy()
    sel.insert(0, "Rank", range(1, len(sel)+1))
    return sel

def build_compare_table(dfA, presetA, dfB, presetB):
    dfa = prep_display_table(dfA, presetA)
    dfb = prep_display_table(dfB, presetB)

    if dfa.empty and dfb.empty:
        return pd.DataFrame()

    # Adiciona sufixo A/B nos nomes (exceto Rank)
    if not dfa.empty:
        dfa = dfa.rename(columns={c: f"{c} [A]" for c in dfa.columns if c != "Rank"})
    if not dfb.empty:
        dfb = dfb.rename(columns={c: f"{c} [B]" for c in dfb.columns if c != "Rank"})

    merged = pd.merge(dfa, dfb, on="Rank", how="outer")
    merged = merged.sort_values("Rank").fillna("—")
    return merged

def render_card_for_row(rw, preset, side_key: str, title: str, idx: int, qty_shares=0, cash_avail_val=0.0, contract_size=100, dias_alerta=7, meta_captura=75, janela_pct=5, spot=0.0):
    if rw is None:
        st.markdown(f"<div class='card-placeholder'>— Sem candidato neste rank —</div>", unsafe_allow_html=True)
        return

    lots_key = f"lots_{side_key}_{idx}"
    if "lot_map" not in st.session_state:
        st.session_state["lot_map"] = {}
    if lots_key not in st.session_state["lot_map"]:
        st.session_state["lot_map"][lots_key] = 0

    lots = st.number_input(
        f"#{idx+1} — Lotes (1 lote = 1 PUT + 1 CALL) [{title}]",
        min_value=0, max_value=10000, value=st.session_state["lot_map"][lots_key], key=lots_key,
        help="Quantidade de lotes para esta sugestão. Aumenta proporcionalmente o prêmio total e as exigências de cobertura."
    )
    st.session_state["lot_map"][lots_key] = lots

    effective_contract_size = int(contract_size) if contract_size else CONTRACT_SIZE
    premio_total = rw["credito"] * effective_contract_size * lots

    with st.container(border=True):
        venc_txt = rw["expiration"].strftime("%Y-%m-%d")
        st.markdown(
            f"**#{idx+1} → Vender PUT `{rw['PUT']}` (Kp={rw['Kp']:.2f}) + CALL `{rw['CALL']}` (Kc={rw['Kc']:.2f}) · Vencimento: `{venc_txt}`**"
        )
        c1, c2, c3 = st.columns([1.0, 1.2, 1.2])
        c1.metric("Crédito/ação", format_brl(rw["credito"]))
        c2.metric("Break-evens (mín–máx)", f"{rw['be_low']:.2f} — {rw['be_high']:.2f}")
        c3.metric("Prob. exercício (PUT / CALL)", f"{100*rw['poe_put']:.1f}% / {100*rw['poe_call']:.1f}%")

        # Cobertura
        required_shares = effective_contract_size * lots
        required_cash   = rw["Kp"] * effective_contract_size * lots
        covered_call = qty_shares >= required_shares if lots > 0 else True
        covered_put  = (cash_avail_val if pd.notna(cash_avail_val) else 0.0) >= required_cash if lots > 0 else True

        st.markdown(coverage_badge(covered_call, covered_put), unsafe_allow_html=True)
        st.caption(
            f"CALL coberta exige **{required_shares} ações**; PUT coberta exige **{format_brl(required_cash)}** em caixa (no strike da PUT)."
            + (" Nenhuma exigência enquanto lotes = 0." if lots == 0 else "")
        )

        if not covered_call or not covered_put:
            st.warning(
                f"Cobertura insuficiente para {lots} lote(s): "
                f"precisa de **{required_shares} ações** e **{format_brl(required_cash)} em caixa** "
                f"(CALL usa ações; PUT usa caixa no strike da PUT)."
            )

        d1, d2 = st.columns([1.1, 2.0])
        d1.metric("🎯 Prêmio estimado (total)", format_brl(premio_total))
        d2.markdown(
            f"**Cálculo:** `crédito/ação × contrato × lotes` = "
            f"`{rw['credito']:.2f} × {effective_contract_size} × {lots}` → **{format_brl(premio_total)}**"
        )

        # Alertas informativos
        if rw["days_to_exp"] <= dias_alerta:
            st.info(f"⏳ Faltam {int(rw['days_to_exp'])} dia(s) para o vencimento. Considere realizar lucro se capturou ~{meta_captura}% do crédito.")
        if abs(spot - rw["Kc"]) <= rw["Kc"] * (janela_pct/100.0):
            st.warning("🔺 CALL ameaçada (preço perto do strike da CALL). Sugestão: recomprar a CALL para travar o ganho.")
        if abs(spot - rw["Kp"]) <= rw["Kp"] * (janela_pct/100.0):
            st.warning("🔻 PUT ameaçada (preço perto do strike da PUT). Sugestão: avaliar recompra da PUT ou rolagem.")

        # “Por que #1?” — só quando idx==0
        if idx == 0:
            with st.expander("🔎 Por que esta ficou em #1?"):
                largura = rw["Kc"] - rw["Kp"]
                st.markdown(
                    f"- **Score** = `crédito × (p_inside^α)` = `{rw['credito']:.4f} × ({rw['p_inside']:.4f}^{risk_presets[preset]['alpha']})` → **{rw['score']:.4f}**"
                )
                st.markdown(
                    f"- **Probabilidades**: PoE PUT `{100*rw['poe_put']:.1f}%` · PoE CALL `{100*rw['poe_call']:.1f}%` · média `{100*((rw['poe_put']+rw['poe_call'])/2):.1f}%` (limite `{int(100*risk_presets[preset]['max_comb'])}%`)."
                )
                st.markdown(
                    f"- **Largura entre strikes**: `Kc - Kp = {largura:.2f}` (mínimo exigido `{(S*risk_presets[preset]['min_width']):.2f}` com spot `{S:.2f}`)."
                )
                st.markdown(
                    f"<span class='note'>Resumo: este par equilibra bom crédito com alta chance de ficar entre strikes (p_inside) e cumpre os filtros ativos do seu preset.</span>",
                    unsafe_allow_html=True
                )

        # Explicações por sugestão
        with st.expander("📘 O que significa cada item?"):
            premio_put_txt   = format_brl(rw["premio_put"])
            premio_call_txt  = format_brl(rw["premio_call"])
            credito_acao_txt = format_brl(rw["credito"])
            be_low_txt       = f"{rw['be_low']:.2f}".replace(".", ",")
            be_high_txt      = f"{rw['be_high']:.2f}".replace(".", ",")
            poe_put_txt      = (f"{100*rw['poe_put']:.1f}%".replace(".", ",")) if pd.notna(rw["poe_put"]) else "—"
            poe_call_txt     = (f"{100*rw['poe_call']:.1f}%".replace(".", ",")) if pd.notna(rw["poe_call"]) else "—"

            st.markdown(f"""
<p><b>Crédito/ação</b><br>
É o total que você recebe ao vender <b>1 PUT</b> + <b>1 CALL</b> (por ação).<br>
<b>Exemplo desta sugestão:</b> PUT paga <b>{premio_put_txt}</b> e CALL paga <b>{premio_call_txt}</b> → crédito/ação = <b>{credito_acao_txt}</b>.
</p>

<p><b>Break-evens (mín–máx)</b><br>
Faixa de preço no vencimento em que o resultado ainda é maior ou igual a zero.<br>
<b>Exemplo desta sugestão:</b> <b>{be_low_txt} — {be_high_txt}</b>.
</p>

<p><b>Probabilidade de exercício (PUT / CALL)</b><br>
Estimativa (modelo Black–Scholes) de cada opção terminar dentro do dinheiro no vencimento.<br>
<b>Exemplo desta sugestão:</b> PUT <b>{poe_put_txt}</b> (chance do preço ficar <i>abaixo</i> do strike da PUT) / CALL <b>{poe_call_txt}</b> (chance do preço ficar <i>acima</i> do strike da CALL).
</p>

<p><b>Lotes e prêmio total</b><br>
Cada lote = vender <b>1 PUT + 1 CALL</b>. Cada contrato = <b>{effective_contract_size} ações</b>.<br>
<b>Prêmio total</b> = <b>crédito/ação × contrato × lotes</b>.<br>
<b>Exemplo com os valores acima:</b> {credito_acao_txt} × {effective_contract_size} × {lots} → <b>{format_brl(rw["credito"] * effective_contract_size * lots)}</b>.
</p>

<p><b>Regras práticas de saída</b><br>
⏳ Faltando <b>{dias_alerta}</b> dias ou menos, acompanhe com mais atenção.<br>
📈 Se o preço à vista encostar no strike da CALL, <b>recompre a CALL</b>.<br>
🎯 Capturou ~<b>{meta_captura}%</b> do crédito? <b>Encerre a operação</b> para garantir o ganho.
</p>
""", unsafe_allow_html=True)

        # === Checklist de Saída (didático) ===
        if "checklist_state" not in st.session_state:
            st.session_state["checklist_state"] = {}

        chk_key = f"{side_key}_{idx}_{rw['PUT']}_{rw['CALL']}"
        state = st.session_state["checklist_state"].get(chk_key, {"meta": False, "call": False, "tempo": False, "nota": ""})

        # Pré-marcação (não automatiza decisão)
        meta_auto = False
        call_auto = abs(spot - rw["Kc"]) <= rw["Kc"] * (janela_pct / 100.0)
        tempo_auto = rw["days_to_exp"] <= dias_alerta
        if chk_key not in st.session_state["checklist_state"]:
            state["meta"] = meta_auto
            state["call"] = call_auto
            state["tempo"] = tempo_auto

        badges = []
        if state.get("meta"):
            badges.append('<span class="badge badge-green">🎯 Meta atingida</span>')
        if state.get("call"):
            badges.append('<span class="badge badge-red">🔺 CALL pressionada</span>')
        if state.get("tempo"):
            badges.append('<span class="badge badge-amber">⏳ Pouco tempo</span>')
        if badges:
            st.markdown(" ".join(badges), unsafe_allow_html=True)

        with st.expander("✅ Checklist de saída (aprendizado prático)"):
            c1, c2, c3 = st.columns(3)
            state["meta"] = c1.checkbox(
                f"🎯 Capturou ~{meta_captura}% do crédito", value=state.get("meta", False),
                key=f"chk_meta_{chk_key}",
                help="Objetivo didático de realização definida por você."
            )
            state["call"] = c2.checkbox(
                f"🔺 Preço encostou no strike da CALL (Kc={rw['Kc']:.2f})", value=state.get("call", False),
                key=f"chk_call_{chk_key}",
                help="Spot dentro da janela definida ao redor do strike da CALL."
            )
            state["tempo"] = c3.checkbox(
                f"⏳ Faltam ≤ {dias_alerta} dias", value=state.get("tempo", False),
                key=f"chk_tempo_{chk_key}",
                help="Janela de tempo que você configurou nos alertas."
            )

            if any([state["meta"], state["call"], state["tempo"]]):
                st.markdown("---")
                st.markdown("### 🧭 O que fazer no Home Broker (guia genérico para alunos)")
                st.markdown("""
| Situação | Ação sugerida | Como fazer no Home Broker |
|:---|:---|:---|
| 🎯 Capturou meta (~{meta}%) | **Encerrar operação** para garantir lucro. | Na aba **Opções**, localize suas posições **vendidas** (CALL e PUT). Clique em **comprar** os mesmos códigos para **zerar** (ex.: recomprar 1 CALL e 1 PUT). |
| 🔺 CALL encostou | **Recomprar a CALL** para travar o ganho e evitar exercício. | Procure a **CALL** pelo código (ex.: `{call_sym}`); selecione **comprar 1 contrato** no mesmo vencimento. Opcional: **rolar** para strike acima ou próximo vencimento. |
| ⏳ Pouco tempo (≤{dias} dias) | **Rolar** a posição para o próximo vencimento (mesmo strike ou ajustado). | **Compre** as opções atuais para zerar e **venda** novo par (PUT+CALL) no vencimento seguinte. |
""".format(meta=int(meta_captura), call_sym=rw['CALL'], dias=int(dias_alerta)))

        st.session_state["checklist_state"][chk_key] = state

# =============== Execução principal (1 ou 2 cenários) ===============
def compute_and_render_single(preset_name):
    top_df, all_df, preset = compute_recos_for_preset(preset_name)
    if top_df is None or top_df.empty:
        st.info("Nenhuma combinação válida após aplicar filtros e largura mínima neste cenário.")
        return
    st.markdown(f"<div class='scenario-title'>📊 Cenário — {preset_name}</div>", unsafe_allow_html=True)
    # tabela simples
    st.dataframe(prep_display_table(top_df, preset), use_container_width=True, hide_index=True)
    st.markdown("—")
    # cards sequenciais
    for i, rw in top_df.iterrows():
        render_card_for_row(
            rw, preset_name, side_key="S", title=f"Cenário — {preset_name}", idx=i,
            qty_shares=qty_shares, cash_avail_val=cash_avail_val,
            contract_size=contract_size, dias_alerta=dias_alerta,
            meta_captura=meta_captura, janela_pct=janela_pct, spot=float(spot)
        )

def compute_and_render_compare(preset_left, preset_right):
    top_left, all_left, p_left = compute_recos_for_preset(preset_left)
    top_right, all_right, p_right = compute_recos_for_preset(preset_right)

    # 1) Tabela combinada, alinhada por Rank
    st.markdown("<div class='scenario-title'>📊 Tabela comparativa — Lado A vs Lado B</div>", unsafe_allow_html=True)
    comp_table = build_compare_table(top_left, p_left, top_right, p_right)
    if comp_table.empty:
        st.info("Nenhum candidato válido em ambos os cenários com os filtros atuais.")
        return
    st.dataframe(comp_table, use_container_width=True, hide_index=True)

    st.markdown("—")

    # 2) Cards pareados por Rank (duas colunas por linha)
    max_len = max(len(top_left) if top_left is not None else 0,
                  len(top_right) if top_right is not None else 0)
    for i in range(max_len):
        colA, colB = st.columns(2)
        with colA:
            st.markdown(f"<div class='scenario-title'>📊 Lado A — {preset_left}</div>", unsafe_allow_html=True)
            rwA = None if (top_left is None or i >= len(top_left)) else top_left.iloc[i]
            if rwA is not None:
                rwA = rwA.copy()
            render_card_for_row(
                rwA, preset_left, side_key=f"L", title=f"Lado A — {preset_left}", idx=i,
                qty_shares=qty_shares, cash_avail_val=cash_avail_val,
                contract_size=contract_size, dias_alerta=dias_alerta,
                meta_captura=meta_captura, janela_pct=janela_pct, spot=float(spot)
            )
        with colB:
            st.markdown(f"<div class='scenario-title'>📊 Lado B — {preset_right}</div>", unsafe_allow_html=True)
            rwB = None if (top_right is None or i >= len(top_right)) else top_right.iloc[i]
            if rwB is not None:
                rwB = rwB.copy()
            render_card_for_row(
                rwB, preset_right, side_key=f"R", title=f"Lado B — {preset_right}", idx=i,
                qty_shares=qty_shares, cash_avail_val=cash_avail_val,
                contract_size=contract_size, dias_alerta=dias_alerta,
                meta_captura=meta_captura, janela_pct=janela_pct, spot=float(spot)
            )

# ====== Renderização: 1 cenário ou comparador alinhado ======
if compare_two:
    compute_and_render_compare(preset_left, preset_right)
else:
    compute_and_render_single(preset_left)

# =========================
# 🧩 Resumo das Ações Marcadas
# =========================
#st.markdown("---")
#st.subheader("🧩 Resumo das ações marcadas")
#_any = False
#if "checklist_state" in st.session_state:
#    for _key, _stt in st.session_state["checklist_state"].items():
#        if any([_stt.get("meta"), _stt.get("call"), _stt.get("tempo")]):
#            _any = True
#            badges = []
#            if _stt.get("meta"): badges.append("🎯 Meta")
#            if _stt.get("call"): badges.append("🔺 CALL")
#            if _stt.get("tempo"): badges.append("⏳ Tempo")
#            st.markdown(f"- **{_key}** → " + " · ".join(badges), unsafe_allow_html=True)
#if not _any:
#    st.info("Nenhuma ação marcada ainda. Marque os checkboxes nos cards acima para ver o resumo consolidado aqui.")

# =========================
# ℹ️ Como cada parâmetro afeta o Top 3 (guia final)
# =========================
st.markdown("---")
with st.expander("ℹ️ Como cada parâmetro afeta o Top 3"):
    st.markdown("""
**Exemplo de referência:**  
`spot: R$ 6,00; strikes: Kp = 5,50 / Kc = 6,50; crédito/ação: R$ 0,18; contrato: 100 ações; lotes: 2.`

---

### Perfil de risco (preset)
Define automaticamente a **largura mínima entre strikes**, as **probabilidades máximas** (por perna e média) e a **penalização α** do ranking.
- **Conservador:** largura `8%`, por perna `20%`, média `15%`, α `3`.
- **Neutro:** largura `6%`, por perna `25%`, média `20%`, α `2`.
- **Agressivo:** largura `4%`, por perna `35%`, média `25%`, α `1`.

---

### Filtrar por |Δ| (0,10–0,25)
Restringe a deltas típicos de OTM saudável (se disponível).
- **Ativar:** tende a reduzir PoE mantendo prêmios razoáveis.  
**Ex.:** CALL com `|Δ| = 0,35` seria filtrada; com `|Δ| = 0,18` passaria.

---

### Alerta de saída (dias)
Quando exibir aviso pelo tempo restante.
- **Diminuir:** o alerta aparece mais cedo.  
**Ex.:** com alerta em `7 dias`, o ⏳ aparece quando faltam `<= 7` dias.

---

### Meta de captura do crédito (%)
Alvo didático para encerrar com lucro.
- **Aumentar:** você tende a esperar mais.  
**Ex.:** meta `75%` → `R$ 0,18 x 0,75 = R$ 0,135` por ação.

---

### Janela no strike (±%)
Sensibilidade para avisos de “encostar” no strike.
- **Aumentar:** mais avisos.
- **Diminuir:** aviso só quando muito perto.  
**Ex.:** `Kc = 6,50` e janela `±5%` → alerta se spot entre `6,18` e `6,83`.
""")

# Rodapé
st.markdown("---")
st.caption("Dica: se a cotação do yfinance parecer defasada, clique no ícone de recarregar (cache ~5 min).")
