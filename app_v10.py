# app_v10.py
# ------------------------------------------------------------
# Strangle Vendido Coberto — v9 (com priorização por baixa probabilidade)
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
# (restante do código inalterado até o final)
# -------------------------
# [ ... tudo igual à sua versão atual, sem mudanças ... ]
# ------------------------------------------------------------
# Adicionar o novo bloco de explicação no rodapé:
# ------------------------------------------------------------

st.markdown("---")
st.markdown("""
## ℹ️ Como cada parâmetro afeta o Top 3

**🔹 Exemplo de referência:**  
spot = **R$ 6,00**, strikes **Kp = 5,50 / Kc = 6,50**, crédito/ação = **R$ 0,18**,  
1 contrato = **100 ações**, 2 lotes = **200 ações**.

---

### **Volatilidade (HV20 %)**
Proxy da volatilidade anual (σ).  
- **Aumentar:** prêmios ↑ e probabilidade de exercício (PoE) ↑.  
- **Diminuir:** prêmios ↓ e PoE ↓.  
> 💬 Exemplo: se a HV20 subir de **20 % → 30 %**, o crédito pode aumentar de **R$ 0,18 → R$ 0,22**,  
> mas a PoE de PUT e CALL tende a subir **cerca de 3 a 5 p.p.**

---

### **Taxa r (anual %)**
Taxa livre de risco usada no modelo Black–Scholes.  
- Impacto pequeno; use algo próximo da **SELIC**.  
> 💬 Exemplo: se a taxa subir de **10 % → 12 %**, o efeito no crédito é de **centavos**  
> e a PoE praticamente **não se altera.**

---

### **Ações em carteira**
Usado apenas para validar **CALL coberta (✅/❌)**.  
- **Aumentar:** permite vender mais lotes cobertos.  
> 💬 Exemplo: 1 contrato = **100 ações**.  
> Se você tiver **200 ações**, pode operar **2 lotes** de CALL coberta.

---

### **Caixa disponível (R$)**
Usado apenas para validar **PUT coberta (✅/❌)** no strike da PUT.  
- **Aumentar:** viabiliza mais lotes de PUT coberta.  
> 💬 Exemplo: com **Kp = 5,50** e **2 lotes**, é preciso ter  
> **R$ 1.100 (2 × 100 × 5,50)** disponíveis em caixa.

---

### **Tamanho do contrato**
Número de ações por contrato (geralmente 100).  
- **Aumentar:** eleva o **prêmio total** e também as **exigências de cobertura**.  
> 💬 Exemplo: crédito/ação **R$ 0,18 × 100 = R$ 18** por lote;  
> com **2 lotes = R$ 36** no total.

---

### **Alerta de saída (dias)**
Define quando mostrar aviso de tempo até o vencimento.  
- **Diminuir:** o alerta aparece mais cedo.  
> 💬 Exemplo: com alerta em **7 dias**, o símbolo ⏳ aparece  
> quando faltarem **7 dias ou menos** para o vencimento.

---

### **Meta de captura do crédito (%)**
Alvo didático para encerrar a operação com lucro.  
- **Aumentar:** você tende a esperar mais para encerrar.  
> 💬 Exemplo: crédito **R$ 0,18 × 75 % = R$ 0,135 por ação**  
> como meta de realização.

---

### **Janela no strike (±%)**
Sensibilidade para avisos de “encostar” no strike.  
- **Aumentar:** mais avisos.  
- **Diminuir:** só quando o preço estiver muito próximo.  
> 💬 Exemplo: com **Kc = 6,50** e janela de **±5 %**,  
> o alerta aparece se o spot estiver entre **6,18 e 6,83.**

---

### **Limite por perna (combinações)**
Número de strikes de PUT e CALL cruzados em pares.  
- **Aumentar:** mais candidatos (app mais lento).  
> 💬 Exemplo: **30 → 100** amplia a busca e pode revelar pares melhores,  
> mas o cálculo leva mais tempo.

---

### **Probabilidade máx. por perna / média**
Filtros “duros” de probabilidade de exercício.  
- **Diminuir:** setups mais conservadores (pode zerar a lista).  
> 💬 Exemplo: se a média máxima for **20 %**, o app descarta  
> pares com PoE média **acima de 20 %.**

---

### **Penalização (α) no ranking**
Peso da punição sobre probabilidades altas no cálculo do **score**.  
- **Aumentar:** prioriza pares com **p_inside alto** (menor risco),  
mesmo que o prêmio seja um pouco menor.  
> 💬 Exemplo: com α **2 → 4**, pares com **p_inside** maior  
> sobem no ranking.

---

### **Filtro por |Δ| (0,10–0,25)**
Restringe a opções com deltas típicos de OTM saudável (se disponível).  
- **Ativar:** reduz a chance de exercício mantendo prêmios razoáveis.  
> 💬 Exemplo: CALL com |Δ| = **0,35** seria filtrada;  
> |Δ| = **0,18** passaria.

---

### **Largura mínima entre strikes (% do spot)**
Define a distância mínima entre Kp e Kc.  
- **Aumentar:** menor risco (pares mais “largos”), menos candidatos.  
> 💬 Exemplo: com spot **R$ 6,00** e largura mínima **6 %**,  
> a diferença mínima entre strikes é **Kc − Kp ≥ 0,36.**

---

✅ **Resumo final:**  
- Parâmetros **↑ (HV20, α, largura)** → setups mais seguros, prêmios moderados.  
- Parâmetros **↓ (PoE, janela, alerta)** → setups mais agressivos, prêmios maiores.  
- Busque sempre o equilíbrio entre **prêmio atraente** e **probabilidade controlada.**
""")

# ------------------------------------------------------------
# Rodapé final
# ------------------------------------------------------------
st.markdown("---")
st.caption("Dica: se a cotação do yfinance parecer defasada, clique no ícone de recarregar (cache ~5 min).")
