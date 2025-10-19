# Strangle Vendido Coberto — App (B3)
**Um aplicativo Streamlit para montar, comparar e entender estratégias de *strangle vendido coberto* com foco em maior prêmio e baixa probabilidade de exercício.**  

> **Resumo:** Cole a *option chain* do opcoes.net, escolha o(s) vencimento(s), selecione um **preset de risco** (Conservador/Neutro/Agressivo), e o app calcula as melhores combinações de **PUT OTM + CALL OTM** (strangle coberto), mostrando **prêmio**, **break-evens**, **probabilidades** (modelo Black–Scholes), **cobertura por sugestão**, **ranking explicável** e um **Checklist de Saída** com orientação prática (“como fazer no home broker”).

---

## 📦 Sumário
- [1. O que é um Strangle Vendido Coberto](#1-o-que-é-um-strangle-vendido-coberto)
- [2. O que o app faz (features)](#2-o-que-o-app-faz-features)
- [3. Instalação e execução](#3-instalação-e-execução)
- [4. Como usar — Passo a passo](#4-como-usar--passo-a-passo)
- [5. Entrada de dados: como colar a option chain](#5-entrada-de-dados-como-colar-a-option-chain)
- [6. Presets de risco (Conservador/Neutro/Agressivo)](#6-presets-de-risco-conservadorneutroagressivo)
- [7. Métricas, filtros e ranking](#7-métricas-filtros-e-ranking)
- [8. Comparador de cenários (lado a lado)](#8-comparador-de-cenários-lado-a-lado)
- [9. Top N dinâmico](#9-top-n-dinâmico)
- [10. Checklist de Saída (didático)](#10-checklist-de-saída-didático)
- [11. Guias práticos: como agir no Home Broker](#11-guias-práticos-como-agir-no-home-broker)
- [12. Exemplos práticos (três situações clássicas)](#12-exemplos-práticos-três-situações-clássicas)
- [13. Cenário-base detalhado + 3 procedimentos](#13-cenário-base-detalhado--3-procedimentos)
- [14. Dúvidas frequentes (FAQ)](#14-dúvidas-frequentes-faq)
- [15. Limitações & avisos importantes](#15-limitações--avisos-importantes)
- [16. Glossário rápido](#16-glossário-rápido)

---

## 1) O que é um Strangle Vendido Coberto
**Strangle vendido**: vender **1 PUT OTM** (strike abaixo do spot) + **1 CALL OTM** (strike acima do spot).  
**Coberto**:  
- a **PUT** é coberta por **caixa** (dinheiro suficiente para comprar as ações no strike, se precisar);  
- a **CALL** é coberta por **ações em carteira** (100 ações por contrato).

**Objetivo:** receber **prêmio** (crédito) com **controle de risco**, priorizando **baixa probabilidade de exercício** das pernas.

---

## 2) O que o app faz (features)
- 🔎 **Busca de ticker** (lista dinâmica via dadosdemercado.com.br).  
- 💵 **Preço à vista** via yfinance (Close mais recente).  
- 📥 **Cola a option chain** do opcoes.net (o app identifica colunas; ignora título/linha vazia).  
- 🗓️ **Escolha múltipla de vencimentos** (ou “todos”).  
- 🏆 **Top N dinâmico** (inicia no Top 3, pode ir até Top 10).  
- 🧭 **Presets de risco** (Conservador/Neutro/Agressivo) que ajustam filtros “duros” e priorização.  
- 🧮 **Cálculos didáticos**: prêmio por ação, break-evens, probabilidades por perna (PoE PUT/CALL), p_inside (prob. de ficar entre os strikes).  
- 📊 **Ranking explicável** (“Por que #1?”).  
- 🛡️ **Indicador de cobertura por sugestão** (CALL/PUT ✅/❌) com exigências numéricas.  
- 🧪 **Comparador de cenários** lado a lado (alinhado por rank).  
- ✅ **Checklist de Saída** com **badges**, **pré-marcação inteligente**, **campo de anotação**, **resumo final** e **guia “Como fazer no Home Broker”**.

---

## 3) Instalação e execução
**Requisitos:** Python 3.10+ (recomendado)

```bash
# 1) criar e ativar venv (opcional)
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 2) instalar dependências
pip install -r requirements.txt

# 3) executar
streamlit run app_v10.py
```
> Se sua versão local for `app_v12`/`app_v11`, apenas atualize o nome no comando `streamlit run`.

---

## 4) Como usar — Passo a passo
1. **Escolha o ticker** (ex.: PETR4).  
2. Confira o **Preço à vista** (yfinance).  
3. **Cole a option chain** do opcoes.net no campo próprio (há um botão **Confirmar** para celular).  
4. Selecione **um ou mais vencimentos** (ou “Selecionar todos”).  
5. Escolha o **Preset de risco** (Conservador/Neutro/Agressivo).  
6. Opcional: marque **Filtrar por |Δ| 0,10–0,25**.  
7. Ajuste **Top N** (quantidade de recomendações).  
8. Visualize o **Top N** e os **cards** com detalhes.  
9. Use o **Checklist de Saída** para treinar a disciplina: marque as situações e leia o **guia prático**.  
10. Veja ao final o **Resumo das ações marcadas** com suas anotações.

---

## 5) Entrada de dados: como colar a option chain
1. No **opcoes.net**, abra a listagem de **opções** do papel (ex.: PETR4).  
2. **Selecione e copie** (Ctrl/Cmd+C) a **tabela** (não é necessário copiar gráficos).  
3. No app, **cole** (Ctrl/Cmd+V) no campo.  
4. **Importante:** o site costuma trazer a primeira linha como título “Opções …” e uma linha vazia — o app **ignora** essas duas linhas automaticamente.  
5. Toque em **Confirmar** (especialmente no smartphone).

Campos mínimos esperados: **Ticker, Vencimento, Tipo (CALL/PUT), Strike, Último**.  
Se vierem **Vol. Impl. (%)** ou **Delta**, o app usa; senão, aplica **proxies** (HV20).

---

## 6) Presets de risco (Conservador/Neutro/Agressivo)
Os **presets** ajustam automaticamente:
- **Largura mínima** entre strikes (em % do spot).  
- **Prob. máx por perna (PUT/CALL)**.  
- **Prob. média máx (PUT/CALL)**.  
- **Penalização α** do ranking (peso para priorizar *p_inside* alto).

| Preset       | Largura mín. | PoE máx/Perna | PoE máx/Média | α |
|--------------|---------------|---------------|---------------|---|
| Conservador  | 8%            | 20%           | 15%           | 3 |
| Neutro       | 6%            | 25%           | 20%           | 2 |
| Agressivo    | 4%            | 35%           | 25%           | 1 |

> **Dica:** Combine o preset com o **Filtro por |Δ| 0,10–0,25** para setups OTM mais “saudáveis”.

---

## 7) Métricas, filtros e ranking
- **Crédito/ação (R$):** soma dos prêmios recebidos pelas duas pernas (PUT + CALL), por ação.  
- **Break-evens (mín–máx):** faixa no vencimento com resultado ≳ 0, calculada por `Kp − crédito` e `Kc + crédito`.  
- **Prob. de exercício por perna (PoE PUT/CALL):** estimada pelo **Black–Scholes** (usando IV se houver; senão, HV20 como proxy de σ).  
- **p_inside:** `1 − PoE(PUT) − PoE(CALL)`, truncado em [0,1] — **chance de ficar entre os strikes**.  
- **Score do ranking:** `score = crédito × (p_inside^α)` — prioriza crédito **com** alta chance de ficar entre strikes.  
- **Filtros “duros”:**  
  - **largura mínima** (distância entre Kc e Kp em % do spot),  
  - **PoE máx por perna**, **PoE média máx** → combinam para descartar pares mais arriscados.

---

## 8) Comparador de cenários (lado a lado)
- Marque **“Comparar 2 cenários (lado a lado)”** e escolha, por exemplo, **Conservador vs Agressivo**.  
- O app mostra uma **tabela única alinhada por Rank** e depois **cards par-a-par** (mesma linha = mesmo rank).  
- Isso facilita perceber o trade-off: **crédito maior** vs **probabilidades** e **largura**.

---

## 9) Top N dinâmico
- Controle **quantas recomendações** quer ver (de **Top 3** até **Top 10**, se houver dados).  
- Útil para aprofundar além do Top 3 e **“pescar” alternativas** que quase entraram.

---

## 10) Checklist de Saída (didático)
Dentro de **cada card**, há um bloco expansível:

- **🎯 Capturou ~X% do crédito** (meta)  
- **🔺 Preço encostou no strike da CALL** (janela ajustável)  
- **⏳ Faltam ≤ Y dias para o vencimento** (alerta de tempo)

**Como funciona:**
- **Pré-marcação inteligente** (⏳ e 🔺 entram marcados automaticamente quando as condições objetivas forem verdade).  
- **Badges visuais** (verde/âmbar/vermelho) ajudam a identificar a situação.  
- **Textos práticos** explicam o que fazer.  
- **📝 Anotação (opcional)** por card, salva na sessão.  
- **🧩 Resumo final** agrupa tudo que foi marcado com suas notas.

---

## 11) Guias práticos: como agir no Home Broker
Quando qualquer item do checklist é marcado, surge um **guia prático**:

| Situação | Ação sugerida | Como fazer no Home Broker |
|:---|:---|:---|
| 🎯 **Capturou meta** (~X%) | **Encerrar operação** para garantir lucro. | Na aba **Opções**, localize suas posições **vendidas** (CALL e PUT) e **compre** a mesma quantidade dos **mesmos códigos** para **zerar**. |
| 🔺 **CALL encostou** | **Recomprar a CALL** para travar o ganho e evitar exercício. | Procure a CALL pelo código; **comprar 1 contrato** (por lote). Opcional: **vender** outra CALL mais acima (rolar “up”) no mesmo/novo vencimento. |
| ⏳ **Pouco tempo** (≤ Y dias) | **Rolar** para outro vencimento (mesmo/novo strike). | **Compre** as opções atuais para zerar; **venda** nova PUT + CALL no vencimento seguinte (ajuste a faixa conforme o preset). |

> **Dica didática:** faça isso primeiro em **simulador** (se sua corretora tiver) ou com **quantidades mínimas**, para treinar o fluxo.

---

## 12) Exemplos práticos (três situações clássicas)
### A) Rolar para outro vencimento (⏳)
- Você está com **2 lotes** (200 ações cobrem a CALL) em **Kp=5,50** e **Kc=6,50**.  
- Faltam **5 dias** e você quer **estender** a estratégia.
1. **Comprar** 2 contratos da **PUT 5,50** e da **CALL 6,50** do mês atual (zera a posição).  
2. **Vender** 2 contratos de **PUT** e **CALL** no **próximo vencimento**, mantendo 5,50/6,50 **ou** ajustando (ex.: 5,60/6,70) conforme seu preset.  
3. Compare o **novo crédito** e **probabilidades** na tela.

### B) Recomprar a CALL (🔺)
- O spot encostou em **6,50**; a CALL está pressionada.
1. **Comprar** 2 contratos da **CALL 6,50** (mesmo vencimento) → **trava** o risco do lado da CALL.  
2. Decidir:  
   - **Rolar a CALL** para **6,70** (subir strike) no mesmo vencimento, **ou**  
   - Rolar para o **próximo vencimento**, **ou**  
   - **Encerrar tudo** se já capturou a meta.

### C) Encerrar para garantir lucro (🎯)
- Seu objetivo era capturar **~75% do crédito** e já está **atingido**.
1. **Comprar** 2 contratos da **PUT 5,50** e 2 da **CALL 6,50**.  
2. Conferir o **resultado** no extrato: lucro ≈ crédito recebido − custo de recomprar.

---

## 13) Cenário-base detalhado + 3 procedimentos
### Cenário-base (o que você tem na carteira)
- **Ativo:** PETR4  
- **Spot (preço à vista):** R$ 6,00  
- **Strangle vendido coberto** (*1 lote = 100 ações*):  
  - **Vendido:** PUT **Kp=5,50** por **R$ 0,12**  
  - **Vendido:** CALL **Kc=6,50** por **R$ 0,06**  
- **Crédito/ação:** R$ 0,12 + R$ 0,06 = **R$ 0,18**  
- **Lotes:** 2 (→ **200 ações** cobrem a CALL; **caixa** cobre a PUT)  
- **Crédito total** (teórico na abertura): `0,18 × 100 × 2 = R$ 36,00`

A partir daqui, **3 situações típicas**:

#### 1) “Rolar” a operação para outro vencimento (mesmos strikes ou ajustando)
**Quando usar:** faltam poucos dias para o vencimento (⏳) e você quer **estender** a estratégia, mantendo a lógica (ex.: vender de novo o mesmo **5,50/6,50** no mês seguinte) ou **ajustando** a faixa.

**Objetivo**
- Zerar as opções atuais (**comprando-as de volta**)  
- **Vender** a nova dupla (PUT+CALL) em **vencimento mais distante**

**No home broker (passo a passo)**
1. Vá à aba **Opções** → **Minhas posições** (ou “Posições em aberto”).  
2. Localize as opções **vendidas** (ex.: *PETR… PUT 5,50* e *PETR… CALL 6,50* do vencimento atual).  
3. **Compre** a mesma quantidade que você tem **vendida** (ex.: **2 contratos** de cada) para **zerar**.  
   - Ordem: **“Comprar 2 contratos”** de cada código do vencimento atual.  
4. Agora, procure o **novo vencimento** (ex.: **mês seguinte**).  
5. **Venda** a nova dupla (PUT e CALL), tipicamente **OTM** e com strikes que respeitem sua **largura/risco** (ex.: manter **5,50 / 6,50** ou ajustar p/ **5,40 / 6,60** se quiser mais “folga”).  
   - Ordem: **“Vender 2 contratos”** na PUT escolhida + **“Vender 2 contratos”** na CALL escolhida.

**Dicas rápidas**
- Se o spot está mais perto de **6,40**, você pode **subir a CALL** (ex.: **6,70**) e **subir um pouco a PUT** (ex.: **5,60**) para manter a **largura** desejada.  
- Compare o **novo crédito** com a **probabilidade (PoE)** no app antes de confirmar.

---

#### 2) Recomprar a CALL para “travar” o ganho (CALL encostou/ameaçada)
**Quando usar:** o preço **encostou** no strike da **CALL** (🔺). Você quer **parar de perder na CALL** se o mercado continuar subindo — travando o resultado dessa perna.

**Objetivo**
- **Zerar apenas a CALL** vendida (**recompra**)  
- Avaliar se mantém a **PUT** vendida, **rola a CALL** para cima, ou **encerra** tudo

**No home broker (passo a passo)**
1. Aba **Opções** → **Minhas posições**.  
2. Localize a **CALL vendida Kc=6,50** (ex.: *PETRK… 6,50*).  
3. **Compre** a mesma quantidade em que você está **vendido** (ex.: **2 contratos**) → isso **zera a CALL**.  
   - Ordem: **“Comprar 2 contratos da CALL 6,50 (mesmo vencimento)”**.  
4. Decida o próximo passo:  
   - **Rolar a CALL**: vender outra CALL **mais acima** (ex.: **6,70**) no **mesmo vencimento** ou no **vencimento seguinte**.  
   - **Manter a PUT**: se o spot está **longe** do strike da PUT, você pode continuar recebendo o **theta** da PUT vendida.  
   - **Fechar tudo**: se já **capturou a meta**, encerre também a PUT (ver item 3).

**Dicas rápidas**
- Recomprar a CALL geralmente **custa** (você “devolve” parte do crédito).  
- Compensa quando: a CALL ficou muito **pressionada** e você quer **cortar risco de exercício**, ou quer **trocar** por um strike mais alto (rolagem “up”).

---

#### 3) Encerrar toda a operação para garantir lucro (capturou a meta 🎯)
**Quando usar:** você já **capturou ~X% do crédito** (ex.: **75%**) e quer **tirar o risco** da mesa.

**Objetivo**
- **Zerar** PUT e CALL **vendidas**  
- **Realizar** o lucro obtido (diferença entre o crédito recebido e o custo para recomprar as pernas)

**No home broker (passo a passo)**
1. Aba **Opções** → **Minhas posições**.  
2. Localize **PUT** e **CALL** vendidas do seu strangle.  
3. **Compre** a mesma quantidade para **zerar as duas pernas**.  
   - Ordem: **“Comprar 2 contratos da PUT 5,50”** e **“Comprar 2 contratos da CALL 6,50”**.  
4. Confira o **resultado financeiro** no extrato:  
   - **Lucro** ≈ **crédito recebido** − **custo de recomprar** as duas opções.

**Dicas rápidas**
- Encerrar com a **meta** evita que um movimento brusco **devolva** o lucro.  
- Se quiser, pode **fechar só uma perna** primeiro (ex.: **CALL**) e avaliar a outra depois — mas isso **muda o perfil de risco**.

---

**Mini-resumo para colar no seu caderno 📒**
- **Rolar (⏳):** **comprar** as opções atuais (zerar) → **vender** nova **PUT+CALL** no próximo vencimento (ajuste strikes conforme seu preset).  
- **Travar CALL (🔺):** **comprar** a CALL vendida (zerar CALL) → opcionalmente **vender** outra CALL **mais acima** (mesmo vencimento ou próximo).  
- **Encerrar (🎯):** **comprar** PUT e CALL (zerar ambas) → **realizar lucro**.

---

## 14) Dúvidas frequentes (FAQ)
**Q1. O app funciona sem Delta e IV?**  
A: Sim. Se **Delta** não vier, o filtro por |Δ| é ignorado. Se **IV** não vier, usa-se **HV20** como proxy de σ.

**Q2. Por que às vezes não há sugestões?**  
A: Filtros “duros” podem eliminar tudo. Tente um **preset menos rígido** ou **desmarque o filtro por Δ**.

**Q3. O preço do yfinance está diferente da cotação agora?**  
A: O app usa **Close mais recente**. Se quiser “spot” ao vivo, atualize cache (até 5 min) ou insira manualmente no código (futuro: campo de override).

**Q4. Quantos contratos por lote?**  
A: **100 ações** por contrato (padrão B3). O app usa esse número para **prêmio total** e **cobertura**.

**Q5. A conta da cobertura é automática?**  
A: Sim:  
- **CALL coberta** → exige **ações** em carteira (100 por contrato × nº de lotes).  
- **PUT coberta** → exige **caixa** = `strike da PUT × 100 × nº de lotes`.  
Os cartões mostram **✅/❌** e os **valores exigidos**.

---

## 15) Limitações & avisos importantes
- **Educacional**: o app é **didático** e **não é recomendação** de investimento.  
- **Modelo teórico**: probabilidades vêm de **Black–Scholes** (com IV ou HV20). Não capturam “caudas gordas”, microestrutura e eventos.  
- **Dados externos**: dependemos de **opcoes.net** (tabela copiada), **dadosdemercado.com.br** (lista de tickers) e **yfinance** (spot). Qualquer inconsistência nessas fontes impacta a análise.  
- **Execução real**: verifique **lotes, taxas, margens** e **liquidez** na sua corretora.

---

## 16) Glossário rápido
- **OTM (Out of The Money)**: opção fora do dinheiro (PUT com strike abaixo do spot; CALL com strike acima).  
- **Prêmio/Crédito**: valor recebido ao vender as opções.  
- **Break-even**: preço no vencimento que zera o resultado do strangle.  
- **PoE (Probabilidade de Exercício)**: chance de a opção terminar **dentro** do dinheiro.  
- **p_inside**: chance estimada de o preço **ficar entre Kp e Kc** no vencimento.  
- **Rolar**: encerrar as opções atuais e abrir novas, geralmente no **próximo vencimento** (e, às vezes, com **novos strikes**).  
- **CALL coberta**: venda de CALL protegida por **ações**.  
- **PUT coberta**: venda de PUT protegida por **caixa** (capaz de comprar as ações no strike).

---

### Contato & feedback
Achou um bug? Tem uma ideia de melhoria? Anote os detalhes (ticker, vencimento, trecho da chain colada) e descreva o comportamento esperado vs observado. Isso acelera qualquer ajuste.

**Bons estudos e bons trades — com disciplina e cobertura!**
