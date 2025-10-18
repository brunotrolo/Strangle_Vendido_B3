# app_v9.py
# -----------------------------------------------
# Strangle Vendido Coberto — v9 (B3)
# Python + Streamlit
# -----------------------------------------------
import io
import math
import re
import textwrap
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Tuple, Optional

import pandas as pd
import numpy as np
import streamlit as st

# ---- Config Streamlit
st.set_page_config(
    page_title="Strangle Vendido Coberto — B3 (v9)",
    page_icon="📊",
    layout="wide",
)

# -----------------------------------------------
# Constantes
# -----------------------------------------------
CONTRACT_SIZE = 100  # padrão B3 (100 ações por contrato)

# -----------------------------------------------
# Utilidades simples
# -----------------------------------------------
def format_brl(x: float) -> str:
    try:
        return f"R$ {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return str(x)

def safe_float(x, default=np.nan):
    try:
        if x is None:
            return default
        if isinstance(x, (int, float, np.number)):
            return float(x)
        s = str(x).strip().replace("R$", "").replace(" ", "").replace(".", "").replace(",", ".")
        return float(s)
    except Exception:
        return default

def parse_b3_date(s: str) -> Optional[datetime]:
    # tenta dd/mm/aaaa; dd-mm-aaaa; aaaa-mm-dd
    s = (s or "").strip()
    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            pass
    return None

# -----------------------------------------------
# Black–Scholes aproximado para PoE
# -----------------------------------------------
def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def d1(S, K, r, sigma, T):
    if S <= 0 or K <= 0 or sigma <= 0 or T <= 0:
        return np.nan
    return (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))

def d2(S, K, r, sigma, T):
    v = d1(S, K, r, sigma, T)
    if np.isnan(v):
        return np.nan
    return v - sigma * math.sqrt(T)

def poe_put(S, K, r, sigma, T):
    """
    Probabilidade de exercício da PUT ~ P(S_T < K).
    No modelo BS (medida risco-neutra): P(S_T < K) = N(-d2)
    """
    v = d2(S, K, r, sigma, T)
    if np.isnan(v):
        return np.nan
    return norm_cdf(-v)

def poe_call(S, K, r, sigma, T):
    """
    Probabilidade de exercício da CALL ~ P(S_T > K) = N(d2)
    """
    v = d2(S, K, r, sigma, T)
    if np.isnan(v):
        return np.nan
    return norm_cdf(v)

# -----------------------------------------------
# yfinance (spot e HV20) — tolerante a falhas
# -----------------------------------------------
@st.cache_data(show_spinner=False, ttl=300)
def fetch_yf_price_and_hv20(yahoo_symbol: str) -> Tuple[Optional[float], Optional[float]]:
    try:
        import yfinance as yf
        h = yf.Ticker(yahoo_symbol).history(period="6mo")
        if h is None or len(h) == 0:
            return (np.nan, np.nan)
        spot = float(h["Close"].iloc[-1])
        # HV20 simples
        ret = h["Close"].pct_change().dropna()
        hv20 = float(ret.tail(20).std() * math.sqrt(252))
        return (spot, hv20)
    except Exception:
        return (np.nan, np.nan)

def yahoo_symbol_from_b3(ticker: str) -> str:
    """
    Mapeia PETR4 -> PETR4.SA; VALE3 -> VALE3.SA; ETFs (IVVB11 -> IVVB11.SA)
    """
    t = (ticker or "").strip().upper()
    if not t.endswith(".SA"):
        t = t + ".SA"
    return t

# -----------------------------------------------
# Leitura/limpeza de option chain colada do opcoes.net.br
# -----------------------------------------------
def try_read_tables_from_text(raw: str) -> List[pd.DataFrame]:
    """
    O usuário cola uma ou mais tabelas (HTML renderizado/copypaste).
    Tentamos converter em DataFrames a partir de '|' ou abas/CSV.
    """
    if not raw or not raw.strip():
        return []

    # 1) tenta CSV/TSV
    dfs = []
    try:
        df = pd.read_csv(io.StringIO(raw), sep=None, engine="python")
        if df is not None and len(df.columns) > 1:
            dfs.append(df)
    except Exception:
        pass

    # 2) tenta por blocos de linhas com separador '|'
    blocks = re.split(r"\n\s*\n", raw.strip())
    for b in blocks:
        if "|" in b and "\n" in b:
            try:
                rows = [r for r in b.splitlines() if r.strip()]
                cols = [c.strip() for c in rows[0].split("|")]
                data = [ [c.strip() for c in r.split("|")] for r in rows[1:] ]
                df2 = pd.DataFrame(data, columns=cols)
                if len(df2.columns) > 1:
                    dfs.append(df2)
            except Exception:
                pass

    return dfs

def pick_best_table(dfs: List[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """
    Escolhe a 'melhor' tabela que contenha colunas típicas (código, strike, bid/ask, vencimento).
    """
    if not dfs:
        return None

    best = None
    for df in dfs:
        cs = [str(c).lower() for c in df.columns]
        score = 0
        if any("cód" in c or "cod" in c or "código" in c or "codigo" in c for c in cs):
            score += 1
        if any("strike" in c or "preço de exercício" in c or "preco de exercicio" in c for c in cs):
            score += 1
        if any("venc" in c or "vencimento" in c for c in cs):
            score += 1
        if any("bid" in c or "of.compra" in c or "oferta de compra" in c for c in cs):
            score += 1
        if any("ask" in c or "of.venda" in c or "oferta de venda" in c for c in cs):
            score += 1
        if score >= 3:
            best = df
            break
    if best is None:
        best = dfs[0]
    best.columns = [str(c).strip() for c in best.columns]

    # tenta identificar colunas principais por nome aproximado
    code_col = None
    name_col = None
    for c in best.columns:
        cl = c.lower()
        if code_col is None and ("código" in cl or "codigo" in cl or "cód" in cl or "cod." in cl or "ticker" in cl or "símbolo" in cl or "simbolo" in cl or cl == "cód."):
            code_col = c
        if name_col is None and ("empresa" in cl or "razão" in cl or "razao" in cl or "ativo" in cl or "subjacente" in cl):
            name_col = c

    return best

def normalize_chain(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza nomes de colunas, converte números e datas.
    """
    d = df.copy()
    # renomeia heurístico
    ren = {}
    for c in d.columns:
        cl = c.lower()
        if "código" in cl or "codigo" in cl or cl in ("cód.", "cód", "cod", "ticker", "símbolo", "simbolo"):
            ren[c] = "codigo"
        elif "venc" in cl:
            ren[c] = "vencimento"
        elif "strike" in cl or "exerc" in cl:
            ren[c] = "strike"
        elif "bid" in cl or "of.compra" in cl or "compra" in cl:
            ren[c] = "bid"
        elif "ask" in cl or "of.venda" in cl or "venda" in cl:
            ren[c] = "ask"
        elif "tipo" in cl:
            ren[c] = "tipo"
        elif "ativo" in cl or "subjacente" in cl or "empresa" in cl or "razão" in cl or "razao" in cl:
            ren[c] = "ativo"
    d = d.rename(columns=ren)

    # garante colunas
    for c in ["codigo", "vencimento", "strike", "bid", "ask", "ativo", "tipo"]:
        if c not in d.columns:
            d[c] = np.nan

    # números
    d["strike"] = d["strike"].apply(safe_float)
    d["bid"] = d["bid"].apply(safe_float)
    d["ask"] = d["ask"].apply(safe_float)
    # datas
    d["venc_dt"] = d["vencimento"].apply(parse_b3_date)

    # tipo pela letra final (CALL = final A até L; PUT = M até X)
    def infer_tipo(code: str) -> str:
        code = (str(code) or "").strip().upper()
        if not code:
            return ""
        last = code[-1]
        if last >= "M":
            return "PUT"
        return "CALL"

    d["tipo"] = d["tipo"].fillna(d["codigo"].apply(infer_tipo))
    return d

# -----------------------------------------------
# Filtrar CALLs OTM & PUTs OTM por um strike alvo S (spot)
# -----------------------------------------------
def filter_otm(d: pd.DataFrame, spot: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    calls = d[(d["tipo"].str.upper() == "CALL") & (d["strike"] > spot)].copy()
    puts = d[(d["tipo"].str.upper() == "PUT") & (d["strike"] < spot)].copy()
    # keep apenas preços válidos
    calls = calls[(calls["bid"].notna()) | (calls["ask"].notna())]
    puts = puts[(puts["bid"].notna()) | (puts["ask"].notna())]
    return calls, puts

# -----------------------------------------------
# Seleção combinatória básica: Top 3 por maior crédito/ação
# -----------------------------------------------
def pick_top3_strangles(calls: pd.DataFrame, puts: pd.DataFrame, S: float, r: float, sigma: float, T_days: int) -> pd.DataFrame:
    """
    Combina um strike de CALL e um de PUT; classifica por maior crédito (bid/ask mid)
    e calcula be_low/be_high + PoE.
    """
    if len(calls) == 0 or len(puts) == 0:
        return pd.DataFrame()

    def mid(row):
        b = row.get("bid", np.nan)
        a = row.get("ask", np.nan)
        if np.isnan(b) and np.isnan(a):
            return np.nan
        if np.isnan(b):
            return a
        if np.isnan(a):
            return b
        return (b + a) / 2.0

    c = calls.copy()
    p = puts.copy()
    c["mid"] = c.apply(mid, axis=1)
    p["mid"] = p.apply(mid, axis=1)

    # drop NaN mid
    c = c.dropna(subset=["mid"])
    p = p.dropna(subset=["mid"])
    if len(c) == 0 or len(p) == 0:
        return pd.DataFrame()

    T = max(T_days / 252.0, 1e-6)
    rows = []
    for _, rc in c.iterrows():
        for _, rp in p.iterrows():
            credito = float(rc["mid"] + rp["mid"])
            be_low = float(rp["strike"] - credito)
            be_high = float(rc["strike"] + credito)

            poe_p = poe_put(S, rp["strike"], r, sigma, T)
            poe_c = poe_call(S, rc["strike"], r, sigma, T)

            rows.append({
                "codigo_call": rc["codigo"],
                "Kc": float(rc["strike"]),
                "premio_call": float(rc["mid"]),
                "codigo_put": rp["codigo"],
                "Kp": float(rp["strike"]),
                "premio_put": float(rp["mid"]),
                "credito": float(credito),
                "be_low": float(be_low),
                "be_high": float(be_high),
                "poe_put": float(poe_p) if poe_p == poe_p else np.nan,
                "poe_call": float(poe_c) if poe_c == poe_c else np.nan,
            })

    out = pd.DataFrame(rows)
    out = out.sort_values("credito", ascending=False).head(3).reset_index(drop=True)
    return out

# -----------------------------------------------
# UI
# -----------------------------------------------
st.title("Strangle Vendido Coberto — B3 (v9)")
st.caption("Cole a option chain do opcoes.net.br, selecione o vencimento, e veja o Top 3 por crédito/ação.")

# 1) Entrada do ticker
user_ticker = st.text_input("Ticker do ativo subjacente (ex.: PETR4, VALE3, ITUB4)", value="PETR4").strip().upper()

# 2) Preço à vista (yfinance) e HV20
col_a, col_b = st.columns([1, 2], gap="small")

with col_a:
    st.subheader("Preço à vista (Spot) & HV20 (automático)")
    y_ticker = yahoo_symbol_from_b3(user_ticker)
    spot, hv20_auto = fetch_yf_price_and_hv20(y_ticker)

    strike_html = f"""
<div class="strike-card">
  <div class="strike-label">Strike (preço à vista via yfinance)</div>
  <div class="strike-value">{format_brl(spot)}</div>
</div>
"""
    st.markdown(strike_html, unsafe_allow_html=True)

with col_b:
    st.subheader("Cole aqui a option chain (opcoes.net.br)")
    raw_chain = st.text_area(
        "Dica: copie as tabelas da página (PUT e CALL) ou exporte e cole o CSV/TSV.",
        height=220,
    )

# 3) Sidebar: parâmetros (HV20, r) e cobertura
st.sidebar.header("⚙️ Parâmetros & Cobertura")

hv20_default = float(hv20_auto if hv20_auto == hv20_auto else 0.35)
sigma = st.sidebar.number_input("Volatilidade anualizada (HV20, em decimal)", min_value=0.01, max_value=3.0, value=hv20_default, step=0.01, format="%.2f")
risk_free = st.sidebar.number_input("Taxa livre de risco anual (r, decimal)", min_value=0.0, max_value=1.0, value=0.12, step=0.005, format="%.3f")
dias_ate = st.sidebar.number_input("Dias até o vencimento escolhido", min_value=1, max_value=365, value=30, step=1)
lots = st.sidebar.number_input("Lotes (cada lote = 1 PUT + 1 CALL)", min_value=1, max_value=1000, value=1, step=1)
dias_alerta = st.sidebar.slider("Alerta de saída (dias para o vencimento)", min_value=1, max_value=30, value=7, step=1)

# 4) Processa a chain
dfs = try_read_tables_from_text(raw_chain)
best = pick_best_table(dfs) if dfs else None

if best is None:
    st.info("Cole a option chain do opcoes.net.br para continuar.")
else:
    norm = normalize_chain(best)
    if norm["venc_dt"].notna().any():
        vencs = sorted({d.date() for d in norm["venc_dt"] if pd.notna(d)})
        sel_venc = st.selectbox("Selecione o vencimento", options=[str(v) for v in vencs])
        target_date = datetime.strptime(sel_venc, "%Y-%m-%d").date()
        dsel = norm[norm["venc_dt"].dt.date == target_date].copy()
    else:
        st.warning("Não encontrei coluna de vencimento legível; usando todos os dados colados.")
        dsel = norm.copy()

    if spot != spot or spot <= 0:
        st.error("Preço à vista não disponível pelo yfinance. Informe manualmente o 'Strike de referência' na sidebar.")
    else:
        calls, puts = filter_otm(dsel, spot)
        if len(calls) == 0 or len(puts) == 0:
            st.warning("Não encontrei CALLs OTM e/ou PUTs OTM para o spot atual. Verifique a chain colada e o ticker.")
        else:
            top = pick_top3_strangles(calls, puts, S=spot, r=risk_free, sigma=sigma, T_days=int(dias_ate))
            if len(top) == 0:
                st.warning("Não foi possível montar combinações OTM com prêmios válidos.")
            else:
                st.subheader("Top 3 (maior crédito por ação)")
                st.dataframe(
                    top.rename(columns={
                        "codigo_call": "CALL",
                        "Kc": "Strike CALL",
                        "premio_call": "Prêmio CALL (mid)",
                        "codigo_put": "PUT",
                        "Kp": "Strike PUT",
                        "premio_put": "Prêmio PUT (mid)",
                        "credito": "Crédito/ação",
                        "be_low": "BE baixo",
                        "be_high": "BE alto",
                        "poe_put": "PoE PUT",
                        "poe_call": "PoE CALL",
                    })
                )

                # Cartões detalhados
                for i, rw in top.iterrows():
                    with st.container(border=True):
                        st.markdown(f"### 🥇 Cenário #{i+1}")
                        col1, col2, col3, col4 = st.columns([1.2, 1, 1, 1.2])

                        with col1:
                            st.metric("CALL", f"{rw['codigo_call']} @ {format_brl(rw['Kc'])}")
                            st.caption(f"Prêmio mid ≈ {format_brl(rw['premio_call'])}")
                        with col2:
                            st.metric("PUT", f"{rw['codigo_put']} @ {format_brl(rw['Kp'])}")
                            st.caption(f"Prêmio mid ≈ {format_brl(rw['premio_put'])}")
                        with col3:
                            st.metric("Crédito/ação", format_brl(rw["credito"]))
                            st.caption(f"PoE PUT ≈ {100*rw['poe_put']:.1f}% | PoE CALL ≈ {100*rw['poe_call']:.1f}%")
                        with col4:
                            st.metric("Break-evens", f"{format_brl(rw['be_low'])} — {format_brl(rw['be_high'])}")
                            premio_total = rw["credito"] * CONTRACT_SIZE * lots
                            st.caption(
                                "Prêmio total (estimado): "
                                f"**{format_brl(premio_total)}**"
                            )

                        # Explicador (AQUI estava o f-string sem fechamento — corrigido)
                        with st.expander("📘 O que significa cada item?"):
                            st.markdown(
                                f"""
**Crédito/ação**  
Soma dos prêmios recebidos ao vender **1 PUT** e **1 CALL** (por **ação**).  
*Exemplo deste cenário:* a **PUT** paga **R$ {rw.get('premio_put',0):.2f}** e a **CALL** paga **R$ {rw.get('premio_call',0):.2f}**, somando **R$ {rw.get('credito',0):.2f}** por ação.  

**Break-evens (mín–máx)**  
Faixa de preço no vencimento onde o resultado ainda é ≥ 0.  
*Exemplo:* se o ativo oscilar entre **R$ {rw.get('be_low',0):.2f}** e **R$ {rw.get('be_high',0):.2f}**, você encerra a operação no zero a zero ou com lucro.  

**Probabilidade de exercício (PUT / CALL)**  
Estimativa (modelo Black–Scholes) da chance de cada opção ser exercida no vencimento.  
*Exemplo:* **PUT {100*rw.get('poe_put',0):.1f}%** → chance de o preço cair abaixo do strike **R$ {rw.get('Kp',0):.2f}**.  
**CALL {100*rw.get('poe_call',0):.1f}%** → chance de subir acima de **R$ {rw.get('Kc',0):.2f}**.  

**Lotes e prêmio total**  
Cada **lote** = vender **1 PUT + 1 CALL** (contrato = {CONTRACT_SIZE} ações).  
Prêmio total = **crédito/ação × contrato × lotes**.  
*Exemplo:* R$ {rw.get('credito',0):.2f} × {CONTRACT_SIZE} × {lots} = **{format_brl(premio_total)}**.  

**Regras práticas de saída**  
⏳ faltam ≤ **{dias_alerta}** dias → acompanhe com mais atenção.  
📈 se **S** encostar no **Strike da CALL ({rw.get('Kc',0):.2f})**, recompre a CALL.  
📉 se **S** encostar no **Strike da PUT ({rw.get('Kp',0):.2f})**, recompre a PUT.  
                                """
                            )

# Rodapé leve
st.markdown("---")
st.caption("Dica: se a cotação do yfinance parecer defasada, clique no ícone de recarregar (cache ~5 min).")
