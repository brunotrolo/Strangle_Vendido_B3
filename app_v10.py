# app_v9.py
# ------------------------------------------------------------
# Strangle Vendido Coberto — v9
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
@media (prefers-color-scheme: light) {
  .strike-card{ background:#fafafa; border-color:#e5e7eb; }
  .strike-label{ color:#4b5563; }
  .strike-value{ color:#111827; }
}
@media (prefers-color-scheme: dark) {
  .strike-card{ background:#111827; border-color:#374151; }
  .strike-label{ color:#d1d5db; }
  .strike-value{ color:#f9fafb; }
}
</style>
""", unsafe_allow_html=True)

CONTRACT_SIZE = 100  # padrão B3
CACHE_TTL = 300      # 5 min

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
    soup = BeautifulSoup(r.text, "html.parser")
    try:
        dfs = pd.read_html(r.text)
        best = None
        for df in dfs:
            cols = [c.lower() for c in df.columns.astype(str)]
            if any("código" in c or "codigo" in c or "ticker" in c for c in cols) and any("empresa" in c or "razão" in c or "razao" in c or "nome" in c for c in cols):
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
            if name_col is None and ("empresa" in cl or "razão" in cl or "razao" in cl or "nome" in cl ou "companhia" in cl):
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

    raw = text.strip()
    if "\t" not in raw:
        raw = re.sub(r"[ ]{2,}", "\t", raw)

    try:
        df = pd.read_csv(io.StringIO(raw), sep="\t", engine="python")
    except Exception:
        try:
            df = pd.read_csv(io.StringIO(raw), sep=";", engine="python")
        except Exception:
            return pd.DataFrame()

    cols = {c: c.strip() for c in df.columns}
    df.rename(columns=cols, inplace=True)

    def find_col(cands):
        for c in df.columns:
            lc = c.lower()
            for p in cands:
                if p in lc:
                    return c
        return None

    col_ticker = find_col(["ticker"])
    col_venc = find_col(["venc", "vencimento"])
    col_tipo = find_col(["tipo"])
    col_strike = find_col(["strike"])
    col_ultimo = find_col(["último", "ultimo", "last"])
    col_iv = find_col(["vol. impl", "vol impl", "impl", "iv"])
    col_delta = find_col(["delta"])

    if not all([col_ticker, col_venc, col_tipo, col_strike, col_ultimo]):
        return pd.DataFrame()

    out = pd.DataFrame()
    out["symbol"] = df[col_ticker].astype(str).str.strip()
    out["type"] = df[col_tipo].astype(str).str.upper().str.contains("CALL").map({True:"C", False:"P"})
    out["strike"] = df[col_strike].apply(br_to_float)
    out["last"]   = df[col_ultimo].apply(br_to_float)
    out["expiration"] = df[col_venc].apply(parse_date_br)
    out["impliedVol"] = df[col_iv].apply(pct_to_float) if col_iv else np.nan
    out["delta"] = df[col_delta"].apply(br_to_float) if col_delta else np.nan

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
st.caption("Cole a option chain do opcoes.net, escolha o vencimento e veja as sugestões didáticas de strangle coberto.")

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

# 3) Sidebar: parâmetros & regras
st.sidebar.header("⚙️ Parâmetros & Cobertura")

# Explicação geral (sem mudar lógica)
with st.sidebar.expander("ℹ️ Como cada parâmetro afeta o Top 3"):
    st.markdown("""
**Volatilidade (HV20 %)**: proxy de σ. **Maior →** prêmios tendem a subir **e** probabilidade de exercício ↑. **Menor →** prêmios ↓ e probabilidade ↓.  
**Taxa r (anual %)**: efeito pequeno no PoE/preço teórico; use algo próximo da SELIC.  
**Ações em carteira**: usado só para **validar a CALL coberta** (✅/❌). Mais ações → mais lotes cobertos.  
**Caixa disponível (R$)**: usado só para **validar a PUT coberta** (✅/❌) no strike da PUT.  
**Tamanho do contrato**: multiplica o **prêmio total** e os **requisitos de cobertura** (ações/caixa).  
**Alerta de saída (dias)**: define quando mostrar aviso de tempo. **Menor →** alerta aparece mais cedo.  
**Meta de captura (%)**: alvo para encerrar com lucro. **Maior →** você tende a esperar mais.  
**Janela no strike (±%)**: sensibilidade para avisos de “encostar” no strike. **Maior →** mais avisos; **menor →** só quando muito perto.  
**Limite por perna**: quantos strikes por lado entram na combinação. **Maior →** mais candidatos (mais lento).  
**Prob. máx por perna / média**: filtros “duros”. **Menor →** setups mais conservadores (pode zerar a lista).  
**Penalização (α)**: quão forte o ranking pune probabilidade alta. **Maior →** prioriza PoE baixa, mesmo com prêmio menor.  
**Filtro por |Δ|**: quando ligado, tende a reduzir exercício mantendo prêmios razoáveis.  
**Largura mínima (%)**: força pares com strikes mais afastados. **Maior →** menor risco, menos candidatos.
""")

hv20_default = float(hv20_auto) if pd.notna(hv20_auto) else 20.0
hv20_input = st.sidebar.number_input(
    "HV20 (σ anual – proxy) [%]",
    0.0, 200.0, hv20_default, step=0.10, format="%.2f",
    help="Volatilidade histórica anualizada de 20 dias (proxy de σ). ↑ aumenta prêmios e também a probabilidade de exercício."
)
r_input = st.sidebar.number_input(
    "r (anual) [%]",
    0.0, 50.0, 11.0, step=0.10, format="%.2f",
    help="Taxa livre de risco usada no Black–Scholes. Efeito pequeno; use algo próximo da SELIC."
)

st.sidebar.markdown("---")
qty_shares = st.sidebar.number_input(
    f"Ações em carteira ({user_ticker})",
    0, 1_000_000, 0, step=100,
    help="Usado só para validar CALL coberta (✅/❌). Mais ações permitem mais lotes cobertos."
)
cash_avail = st.sidebar.text_input(
    f"Caixa disponível (R$) ({user_ticker})",
    value="0,00",
    help="Usado só para validar PUT coberta (✅/❌) no strike da PUT. Mais caixa permite mais lotes."
)
try:
    cash_avail_val = br_to_float(cash_avail)
except Exception:
    cash_avail_val = 0.0
contract_size = st.sidebar.number_input(
    f"Tamanho do contrato ({user_ticker})",
    1, 1000, CONTRACT_SIZE, step=1,
    help="Quantidade de ações por contrato. Aumenta proporcionalmente o prêmio total e a exigência de cobertura."
)

st.sidebar.markdown("---")
dias_alerta = st.sidebar.number_input(
    "Alerta de saída (dias para o vencimento) ≤",
    1, 30, 7,
    help="Mostra aviso de tempo quando faltar ≤ este número de dias. Valores menores disparam alerta mais cedo."
)
meta_captura = st.sidebar.number_input(
    "Meta de captura do crédito (%)",
    50, 100, 75,
    help="Alvo didático para encerrar a operação com lucro. ↑ significa esperar capturar uma fração maior do crédito."
)
janela_pct = st.sidebar.number_input(
    "Janela de alerta no strike (±%)",
    1, 20, 5,
    help="Sensibilidade para avisos de 'encostar' no strike. ↑ mais avisos (janela larga); ↓ só quando muito perto."
)

st.sidebar.markdown("---")
comb_limit = st.sidebar.slider(
    "Limite por perna para cruzar pares (velocidade)",
    10, 200, 30, step=10,
    help="Quantos strikes por lado entram na combinação (impacta cobertura da busca e desempenho). ↑ mais combinações (mais lento)."
)

# ---------- Novos controles (priorizar baixa probabilidade) ----------
st.sidebar.markdown("### 🎯 Preferência por Baixa Probabilidade")
max_poe_leg  = st.sidebar.slider(
    "Prob. máx por perna (%)", 5, 50, 25, step=1,
    help="Filtro 'duro' por perna (PUT e CALL). ↓ deixa o app mais conservador, pode reduzir fortemente os candidatos."
) / 100.0
max_poe_comb = st.sidebar.slider(
    "Prob. média máx (PUT/CALL) (%)", 5, 50, 20, step=1,
    help="Filtro 'duro' para a média da probabilidade das duas pernas. ↓ prioriza setups com menor chance de exercício combinada."
) / 100.0
alpha        = st.sidebar.slider(
    "Penalização por prob. (α)", 1, 5, 2, step=1,
    help="Peso da punição do ranking sobre probabilidades altas. ↑ prioriza ainda mais PoE baixa mesmo se o prêmio for menor."
)
use_delta_filter = st.sidebar.checkbox(
    "Filtrar por |Δ| ~ 0,10–0,25 (se disponível)", value=True,
    help="Quando marcado, restringe as pernas a deltas típicos de OTM saudável. Reduz chance de exercício mantendo prêmio razoável."
)
min_width_pct = st.sidebar.slider(
    "Largura mínima entre strikes (% do spot)", 1, 20, 6, step=1,
    help="Exige distância mínima entre Kp e Kc. ↑ força pares mais 'largos' (menor risco), mas reduz candidatos."
) / 100.0

# 4) Colar a option chain
st.subheader(f"3) Colar a option chain de {user_ticker} (opcoes.net)")
pasted = st.text_area(
    "Cole aqui a tabela (Ctrl/Cmd+C no site → Ctrl/Cmd+V aqui)",
    height=220,
    help="A tabela precisa conter: Ticker, Vencimento, Tipo (CALL/PUT), Strike, Último, (opcional) Vol. Impl. (%), Delta."
)

df_chain = parse_pasted_chain(pasted)
if df_chain.empty:
    st.info("Cole a tabela para continuar.")
    st.stop()

# 5) Selecionar vencimento
unique_exps = sorted([d for d in df_chain["expiration"].dropna().unique()])
if not unique_exps:
    st.error("Não identifiquei a coluna de Vencimento na tabela colada.")
    st.stop()

sel_exp = st.selectbox(
    "📅 Vencimento — escolha uma data:",
    options=unique_exps,
    format_func=lambda d: d.strftime("%Y-%m-%d"),
    help="Selecione o vencimento para o qual você colou a option chain."
)
today = datetime.utcnow().date()
bus_days = business_days_between(today, sel_exp)
T_years = float(bus_days) / 252.0 if pd.notna(bus_days) and bus_days > 0 else 1/252.0

# 6) Filtrar pela data e calcular métricas
df = df_chain[df_chain["expiration"] == sel_exp].copy().reset_index(drop=True)

# Fallbacks de preço: usar 'last' como mid/credito unitário por opção
df["price"] = df["last"].astype(float)

# sigma por opção: IV se houver, senão HV20 proxy
sigma_proxy = hv20_input / 100.0
df["sigma"] = df["impliedVol"].fillna(sigma_proxy)

# separa calls/puts e aplica condição OTM
S = float(spot) if pd.notna(spot) and spot > 0 else df["strike"].median()
calls = df[df["type"] == "C"].copy()
puts  = df[df["type"] == "P"].copy()

calls["OTM"] = calls["strike"].astype(float) > S
puts["OTM"]  = puts["strike"].astype(float)  < S

# --- filtro opcional por Delta quando disponível ---
if use_delta_filter and "delta" in df.columns:
    calls = calls[calls["OTM"] & pd.notna(calls["price"]) & calls["delta"].abs().between(0.10, 0.25, inclusive="both")]
    puts  = puts[puts["OTM"]  & pd.notna(puts["price"])  & puts["delta"].abs().between(0.10, 0.25, inclusive="both")]
else:
    calls = calls[calls["OTM"] & pd.notna(calls["price"])]
    puts  = puts[puts["OTM"]  & pd.notna(puts["price"])]

if calls.empty or puts.empty:
    st.warning("Não encontrei CALL e PUT OTM simultaneamente nesse vencimento. Experimente ajustar filtros (Delta/limites).")
    st.stop()

# PoE (probabilidade de exercício)
r = r_input / 100.0
def poe_side(row, side):
    K = float(row["strike"])
    sig = float(row["sigma"]) if pd.notna(row["sigma"]) and row["sigma"] > 0 else sigma_proxy
    return prob_ITM_call(S, K, r, sig, T_years) if side == "C" else prob_ITM_put(S, K, r, sig, T_years)

puts["poe"]  = puts.apply(lambda rw: poe_side(rw, "P"), axis=1)
calls["poe"] = calls.apply(lambda rw: poe_side(rw, "C"), axis=1)

# 7) Combinações PUT x CALL (limitadas para velocidade) — configurável
comb_limit = int(comb_limit)
puts_small  = puts.sort_values(["price"], ascending=False).head(comb_limit).copy()
calls_small = calls.sort_values(["price"], ascending=False).head(comb_limit).copy()

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

pairs_df = pd.DataFrame(pairs)
if pairs_df.empty:
    st.warning("Não há pares de PUT e CALL OTM válidos para esse vencimento e preço à vista.")
    st.stop()

# ---------- Largura mínima entre strikes ----------
width_ok = (pairs_df["Kc"] - pairs_df["Kp"]) >= (S * min_width_pct)
pairs_df = pairs_df[width_ok]

if pairs_df.empty:
    st.warning("Todas as combinações ficaram abaixo da largura mínima entre strikes. Reduza o filtro de largura.")
    st.stop()

# ---------- Filtros duros de probabilidade ----------
pairs_df["poe_leg_max"] = pairs_df[["poe_put","poe_call"]].max(axis=1)
pairs_df["poe_comb"]    = pairs_df[["poe_put","poe_call"]].mean(axis=1)

pairs_df = pairs_df[
    (pairs_df["poe_put"]  <= max_poe_leg) &
    (pairs_df["poe_call"] <= max_poe_leg) &
    (pairs_df["poe_comb"] <= max_poe_comb)
]

if pairs_df.empty:
    st.warning("Nenhuma combinação passou pelos limites de probabilidade. Tente relaxar os sliders de probabilidade.")
    st.stop()

# ---------- Novo score com p_inside e penalização α ----------
pairs_df["p_inside"] = (1 - pairs_df["poe_put"].fillna(0) - pairs_df["poe_call"].fillna(0)).clip(lower=0)
pairs_df["score"] = pairs_df["credito"] * (pairs_df["p_inside"] ** alpha)
pairs_df = pairs_df.sort_values(["score","p_inside","credito"], ascending=[False, False, False]).reset_index(drop=True)

# 8) Top 3 + flags de alerta
top3 = pairs_df.head(3).copy()
top3["days_to_exp"] = bus_days

def near_strike(price, strike, pct):
    try:
        return abs(price - strike) <= (strike * (pct/100.0))
    except Exception:
        return False

top3["alert_call"] = top3.apply(lambda r: near_strike(S, r["Kc"], janela_pct), axis=1)
top3["alert_put"]  = top3.apply(lambda r: near_strike(S, r["Kp"], janela_pct), axis=1)
top3["alert_days"] = top3["days_to_exp"] <= dias_alerta

# --- Tabela Top 3
top3_display = top3.copy()
top3_display["Prêmio PUT (R$)"]   = top3_display["premio_put"].map(lambda x: f"{x:.2f}")
top3_display["Prêmio CALL (R$)"]  = top3_display["premio_call"].map(lambda x: f"{x:.2f}")
top3_display["Crédito/ação (R$)"] = top3_display["credito"].map(lambda x: f"{x:.2f}")
top3_display["Break-evens (mín–máx)"] = top3_display.apply(lambda r: f"{r['be_low']:.2f} — {r['be_high']:.2f}", axis=1)
top3_display["Prob. exercício PUT (%)"]  = (100*top3_display["poe_put"]).map(lambda x: f"{x:.1f}")
top3_display["Prob. exercício CALL (%)"] = (100*top3_display["poe_call"]).map(lambda x: f"{x:.1f}")
top3_display["p_dentro (%)]"] = (100*top3_display["p_inside"]).map(lambda x: f"{x:.1f}")

def tag_risco(row):
    tags = []
    if row["poe_leg_max"] > (max_poe_leg * 0.9):
        tags.append("⚠️ prob. por perna alta")
    if row["poe_comb"] > (max_poe_comb * 0.9):
        tags.append("⚠️ prob. média alta")
    if row["p_inside"] < 0.70:
        tags.append("🎯 dentro < 70%")
    return " · ".join(tags)

top3_display["Notas"] = top3.apply(tag_risco, axis=1)

top3_display = top3_display[[
    "PUT","Kp",
    "CALL","Kc",
    "Prêmio PUT (R$)","Prêmio CALL (R$)","Crédito/ação (R$)",
    "Break-evens (mín–máx)",
    "Prob. exercício PUT (%)","Prob. exercício CALL (%)",
    "p_dentro (%)]",
    "Notas"
]]
top3_display.rename(columns={"Kp":"Strike PUT","Kc":"Strike CALL"}, inplace=True)

st.subheader("🏆 Top 3 (prioriza prêmio com baixa probabilidade)")
st.dataframe(top3_display, use_container_width=True, hide_index=True)

# 9) Cartões detalhados
st.markdown("—")
st.subheader("📋 Recomendações detalhadas")

if "lot_map" not in st.session_state:
    st.session_state["lot_map"] = {}
for idx in top3.index:
    if idx not in st.session_state["lot_map"]:
        st.session_state["lot_map"][idx] = 0

for i, rw in top3.iterrows():
    rank = i + 1
    key_lotes = f"lots_{i}"
    lots = st.number_input(
        f"#{rank} — Lotes (1 lote = 1 PUT + 1 CALL)",
        min_value=0, max_value=10000, value=st.session_state["lot_map"][i], key=key_lotes,
        help="Quantidade de lotes para esta sugestão. Aumenta proporcionalmente o prêmio total e as exigências de cobertura."
    )
    st.session_state["lot_map"][i] = lots

    effective_contract_size = int(contract_size) if contract_size else CONTRACT_SIZE
    premio_total = rw["credito"] * effective_contract_size * lots

    with st.container(border=True):
        st.markdown(
            f"**#{rank} → Vender PUT `{rw['PUT']}` (Strike={rw['Kp']:.2f}) + CALL `{rw['CALL']}` (Strike={rw['Kc']:.2f})**"
        )
        c1, c2, c3 = st.columns([1.0, 1.2, 1.2])
        c1.metric("Crédito/ação", format_brl(rw["credito"]))
        c2.metric("Break-evens (mín–máx)", f"{rw['be_low']:.2f} — {rw['be_high']:.2f}")
        c3.metric("Prob. exercício (PUT / CALL)", f"{100*rw['poe_put']:.1f}% / {100*rw['poe_call']:.1f}%")

        # Cobertura
        required_shares = effective_contract_size * lots
        required_cash   = rw["Kp"] * effective_contract_size * lots
        covered_call = qty_shares >= required_shares
        covered_put  = (cash_avail_val if pd.notna(cash_avail_val) else 0.0) >= required_cash

        cov_msg = f"🛡️ Cobertura — CALL: {'✅' if covered_call else '❌'}  ·  PUT: {'✅' if covered_put else '❌'}"
        st.caption(cov_msg)
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

        # Alertas
        if bus_days <= dias_alerta:
            st.info(f"⏳ Faltam {bus_days} dia(s) para o vencimento. Considere realizar lucro se capturou ~{meta_captura}% do crédito.")
        if abs(spot - rw["Kc"]) <= rw["Kc"] * (janela_pct/100.0):
            st.warning("🔺 CALL ameaçada (preço perto do strike da CALL). Sugestão: recomprar a CALL para travar o ganho.")
        if abs(spot - rw["Kp"]) <= rw["Kp"] * (janela_pct/100.0):
            st.warning("🔻 PUT ameaçada (preço perto do strike da PUT). Sugestão: avaliar recompra da PUT ou rolagem.")

        # Explicações (HTML leve p/ evitar quebras)
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

# Rodapé
st.markdown("---")
st.caption("Dica: se a cotação do yfinance parecer defasada, clique no ícone de recarregar (cache ~5 min).")
