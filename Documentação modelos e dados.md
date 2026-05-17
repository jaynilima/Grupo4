# Pairs Trading na B3

Arbitragem estatística aplicada ao mercado acionário brasileiro.  
Pipeline completo implementado do zero: dados brutos da Economatica → universo líquido anual → cointegração de Engle-Granger → spread OLS → z-score → alertas operacionais.

---

## Índice

1. [Contexto e Motivação](#1-contexto-e-motivação)
2. [A Estratégia](#2-a-estratégia)
3. [Como Rodar](#3-como-rodar)
4. [Pipeline — O Que Foi Construído](#4-pipeline--o-que-foi-construído)
   - 4.1 Dados e Limpeza
   - 4.2 Universo Anual — Top 100 por Volume
   - 4.3 Pré-triagem por Correlação
   - 4.4 Teste de Cointegração (Engle-Granger)
   - 4.5 Modelo OLS — Spread
   - 4.6 Z-Score — Sinal Operacional
   - 4.7 Regras de Entrada e Saída
5. [Decisões Metodológicas](#5-decisões-metodológicas)
6. [Resultados](#6-resultados)
   - 6.1 Resumo por Janela Anual
   - 6.2 Pares Mais Frequentes
   - 6.3 Pares Mais Fortes
   - 6.4 Alertas Ativos (snapshot)
7. [Outputs Gerados](#7-outputs-gerados)
8. [Estrutura da Pasta](#8-estrutura-da-pasta)
9. [Próximas Etapas](#9-próximas-etapas)
10. [Referências](#10-referências)

---

## 1. Contexto e Motivação

Este projeto implementa uma estratégia quantitativa de **pairs trading** para o mercado brasileiro, com base no arcabouço de cointegração de Engle-Granger (1987) e na literatura empírica de Gatev, Goetzmann e Rouwenhorst (2006).

O objetivo central é responder: **existem pares de ações na B3 com relações estatísticas estáveis o suficiente para que seus desvios temporários sejam explorados como oportunidade de arbitragem?**

O projeto parte dos dados brutos da Economatica, cobre o período de **2015 a 2025** e produz, ao final, uma lista de pares cointegrados por janela anual, com seus parâmetros OLS, séries de spread e z-score diários, e alertas operacionais em tempo real.

---

## 2. A Estratégia

**Pairs trading** é uma estratégia *market-neutral*: compra-se o ativo relativamente barato e vende-se o relativamente caro ao mesmo tempo, eliminando a exposição direcional ao mercado. O retorno vem exclusivamente da **convergência do spread** entre os dois ativos.

### Lógica fundamental

```
Dois ativos cointegrados compartilham uma tendência de longo prazo comum.
Quando o spread entre eles se afasta da média histórica além de um limiar
estatístico, espera-se que ele reverta — gerando a oportunidade.
```

### Fluxo de um trade

```
  Spread abre (|Z| ≥ 2)
        │
        ├── Z > +2 → A está caro relativo a B
        │           → SHORT: vende A, compra β unidades de B
        │
        └── Z < -2 → A está barato relativo a B
                    → LONG: compra A, vende β unidades de B

  Spread fecha (|Z| ≤ 0,5)
        │
        └── Encerra a posição → lucro vem da convergência
```

---

## 3. Como Rodar

### Pré-requisitos

```bash
pip install -r requirements.txt
```

```
pandas>=1.5  numpy>=1.23  matplotlib>=3.6  statsmodels>=0.13  scikit-learn>=1.1
```

### Executar

1. Coloque o arquivo CSV na **mesma pasta** do `rodar.py`
2. Confirme o nome do arquivo na linha `NOME_CSV = ...` no topo do script
3. Execute:

```bash
python rodar.py
```

O script exibe um banner de verificação antes de processar qualquer dado:

```
==================================================================
  PAIRS TRADING B3
==================================================================

  VERIFIQUE O CAMINHO DO ARQUIVO CSV:

  [OK] Arquivo encontrado:
       C:\Users\...\dados_economatica_B3 (2).csv

  Saida (CSVs + graficos):
       C:\Users\...\output

  Parametros:
       Anos    : 2015 a 2025
       Top N   : 100 ativos/ano por volume
       Corr min: 0.8
       p-value : < 0.05 (Engle-Granger)

  Pressione ENTER para continuar (ou Ctrl+C para cancelar)...
```

Se aparecer `[ERRO]`, o CSV não foi encontrado. O script procura automaticamente nas pastas: mesma pasta do script, Desktop, Documents e Downloads.

**Tempo de execução:** ~80–120 segundos.  
**Resultado:** `output/` com 6 CSVs e `output/graficos/` com 12 gráficos.

### Parâmetros ajustáveis (topo do `rodar.py`)

```python
NOME_CSV     = "dados_economatica_B3 (2).csv"
ANO_INICIO   = 2015
ANO_FIM      = 2025
TOP_N        = 100        # ativos selecionados por janela anual
MIN_PREGOES  = 0.60       # % mínimo de pregões no ano
CORR_MIN     = 0.80       # correlação mínima para pré-triagem
PVALUE_MAX   = 0.05       # p-value máximo Engle-Granger
```

### Formato esperado do CSV

Arquivo texto separado por vírgula, encoding `latin-1`, com cabeçalho na primeira linha:

| Coluna | Descrição |
|---|---|
| `Ativo` | Ticker com sufixo (ex: `PETR4<XBSP>`) |
| `Data` | Data no formato DD/MM/AAAA |
| `Fechamento` | Preço de fechamento **ajustado** por proventos |
| `Volume_BRL_k` | Volume financeiro em R$ mil |

As demais colunas do export Economatica (Abertura, Mínimo, Máximo, Médio, Q_Negs, Q_Títulos) são ignoradas.

---

## 4. Pipeline — O Que Foi Construído

O script `rodar.py` executa as etapas abaixo em sequência, para cada janela anual de 2015 a 2025.

### 4.1 Dados e Limpeza

**Entrada:** CSV bruto da Economatica com ~973.000 linhas e 650 ativos únicos.

**Tratamentos aplicados:**

| Tratamento | Detalhe |
|---|---|
| Sufixo do ticker | Remove `<XBSP>` e espaços em branco |
| Preços ausentes | Remove linhas com `Fechamento = "-"` ou nulo |
| Conversão numérica | Converte `Fechamento` e `Volume_BRL_k` para `float64` |
| Forward fill | Preenche até 3 dias úteis consecutivos (feriados, suspensões temporárias) |
| Leitura em chunks | Processa o CSV em blocos de 200.000 linhas para controle de memória RAM |

**Por que forward fill de 3 dias?**  
Feriados locais e suspensões temporárias de negociação criam lacunas pontuais nas séries. Propagar o último preço válido por até 3 dias é prática padrão na literatura e não distorce as correlações de longo prazo. Lacunas maiores que 3 dias sinalizam ativo sem liquidez suficiente e são tratadas na etapa seguinte.

### 4.2 Universo Anual — Top 100 por Volume

Para cada ano, o pipeline seleciona os ativos que vão ser usados para formar pares naquela janela.

**Processo:**
1. Calcula o **volume financeiro total** de cada ativo no ano
2. Filtra ativos com menos de `60%` dos pregões disponíveis no ano (evita ativos com negociação intermitente)
3. Seleciona os **100 com maior volume**
4. Constrói a matriz de preços `(dias_do_ano × 100_ativos)`

```
Pares candidatos por janela = 100 × 99 / 2 = 4.950
```

**Por que recomputar o universo a cada ano?**  
Uma lista fixa de ativos para todo o período 2015–2025 capturaria ativos que eram líquidos em 2015 mas saíram de relevância, ou ignoraria IPOs importantes que vieram depois. Recompor o universo anualmente garante que o modelo sempre opera sobre os ativos mais negociados *do momento*.

### 4.3 Pré-triagem por Correlação

Antes de rodar o teste de cointegração (computacionalmente intenso), descarta-se os pares sem relação linear relevante.

```
|ρ( log P_A , log P_B )| ≥ 0,80
```

A correlação é calculada sobre os **log-preços** (não os retornos), porque cointegração é uma propriedade dos níveis das séries.

**Efeito:** reduz os 4.950 candidatos para ~300–1.900 por ano, dependendo do grau de correlação do mercado naquele período.

### 4.4 Teste de Cointegração — Engle-Granger (1987)

Para cada par que passou no filtro de correlação, estima-se a regressão OLS:

```
log(P_A,t) = α + β · log(P_B,t) + ε_t
```

Aplica-se o **Augmented Dickey-Fuller (ADF)** no resíduo `ε_t`:

```
H₀: ε_t possui raiz unitária  →  par NÃO cointegrado
H₁: ε_t é estacionário        →  par cointegrado (spread reverte à média)

Critério: p-value ADF < 0,05
```

**Por que Engle-Granger e não Johansen?**  
Para pares (dois ativos), o Engle-Granger é equivalente ao Johansen e mais simples de implementar e interpretar. Johansen é mais adequado quando o número de variáveis no sistema é maior que dois.

**Por que log-preços e não preços?**  
Log-preços têm variância mais estável ao longo do tempo e a relação linear `log(A) = α + β·log(B)` pode ser interpretada diretamente como uma relação de potência entre os preços (`A = e^α · B^β`), que é economicamente mais natural do que uma relação linear nos níveis.

### 4.5 Modelo OLS — Spread

Para cada par cointegrado (A, B) na janela do ano Y, estimam-se os parâmetros `α` e `β` via OLS e calcula-se o **spread** diário:

```
Spread_t = log(P_A,t) − α − β · log(P_B,t)
```

O spread representa o **desvio da relação de equilíbrio de longo prazo** entre A e B. Como o par é cointegrado, esse spread é estacionário — oscila em torno de uma média com variância finita.

**Parâmetros salvos por par:**

| Parâmetro | Interpretação |
|---|---|
| `α` (alpha) | Intercepto — diferença de nível entre os log-preços |
| `β` (beta) | Hedge ratio — quantas unidades de B para cada unidade de A |
| `μ_spread` | Média histórica do spread na janela de formação |
| `σ_spread` | Desvio-padrão do spread na janela de formação |

### 4.6 Z-Score — Sinal Operacional

O spread é normalizado pelo seu histórico para produzir um sinal comparável entre pares com escalas diferentes:

```
         Spread_t − μ_spread
Z_t  =  ─────────────────────
              σ_spread
```

O z-score responde: *"quantos desvios-padrão o spread está afastado da sua média histórica?"*

```
 Z-score
   │
+3 ┤ ··················· ← Stop-loss superior (divergência)
   │
+2 ┤ ━━━━━━━━━━━━━━━━━━━ ← Entrada SHORT (A caro relativo a B)
   │
+1 ┤
   │
 0 ┤ ─────────────────── ← Equilíbrio histórico
   │
-1 ┤
   │
-2 ┤ ━━━━━━━━━━━━━━━━━━━ ← Entrada LONG (A barato relativo a B)
   │
-3 ┤ ··················· ← Stop-loss inferior (divergência)
```

`μ` e `σ` são estimados na janela de formação (ano corrente) e mantidos fixos. No `zscores.csv` de saída, cada par usa os parâmetros do **último ano em que foi validado como cointegrado**.

### 4.7 Regras de Entrada e Saída

| Evento | Condição | Ação |
|---|---|---|
| **Entrada LONG** | Z_t ≤ −2,0 | Compra A + Vende β unidades de B |
| **Entrada SHORT** | Z_t ≥ +2,0 | Vende A + Compra β unidades de B |
| **Saída — convergência** | \|Z_t\| ≤ 0,5 | Encerra posição (lucro) |
| **Stop — tempo** | Posição aberta > 30 pregões | Saída forçada (spread não normalizou) |
| **Stop — divergência** | \|Z_t\| ≥ 3,0 | Saída forçada (spread piorando) |

**Limiares alternativos para calibração no backtesting:**

| Perfil | Entrada | Saída | Efeito esperado |
|---|---|---|---|
| Agressivo | \|Z\| ≥ 1,5 | \|Z\| ≤ 0,5 | Mais trades, menor retorno por trade |
| **Moderado** (referência) | **\|Z\| ≥ 2,0** | **\|Z\| ≤ 0,5** | Padrão da literatura |
| Conservador | \|Z\| ≥ 2,5 | \|Z\| ≤ 0,5 | Menos trades, maior convicção estatística |

---

## 5. Decisões Metodológicas

Decisões tomadas durante o desenvolvimento e o raciocínio por trás de cada uma.

### Universo anual vs. universo fixo

**Decisão:** recomputar o top 100 por volume a cada ano.

**Motivo:** um universo fixo para 2015–2025 capturaria ativos que eram líquidos em um período mas não em outro (ex: SMLS3 era líquida até 2019 quando foi incorporada pela RAIL3; BHIA3 só ganhou volume depois de 2020). O universo anual reflete a liquidez real de cada época e evita que o modelo opere sobre ativos ilíquidos.

### Pré-triagem por correlação nos log-preços

**Decisão:** filtrar pares com `|ρ| ≥ 0,80` nos log-preços antes do ADF.

**Motivo:** o teste ADF é computacionalmente intenso (~5–50ms por par). Com 4.950 candidatos por ano e 11 anos, rodar ADF em todos seria ~55.000 testes. A correlação de Pearson nos log-preços é calculada em milissegundos para todos os pares simultaneamente (via `DataFrame.corr()`) e serve como filtro eficiente: pares sem correlação relevante não serão cointegrados.

### Forward fill limitado a 3 dias

**Decisão:** preencher lacunas de até 3 pregões consecutivos, nunca mais.

**Motivo:** lacunas de 1–3 dias são quase sempre feriados ou suspensões técnicas. Propagar por mais de 3 dias começaria a mascarar períodos de iliquidez real, distorcendo volumes e correlações. Lacunas maiores fazem o ativo não atingir o filtro de 60% dos pregões e ser descartado naturalmente.

### Parâmetros fixos na janela de formação

**Decisão:** estimar `α`, `β`, `μ`, `σ` nos dados do ano de formação e mantê-los fixos.

**Motivo:** usar parâmetros móveis (janela rolante) introduz o risco de overfitting e torna o modelo mais complexo de auditar. Parâmetros fixos na janela de formação seguem a abordagem clássica da literatura (Gatev et al., 2006) e são suficientes para uma primeira análise robusta. A Fase 4 (walk-forward) abordará o problema da estabilidade dos parâmetros.

### Regressão OLS simples vs. OLS com constante

**Decisão:** usar `LinearRegression` do scikit-learn (inclui intercepto automaticamente).

**Motivo:** sem intercepto, o modelo forçaria a relação a passar pela origem, o que não é economicamente justificado (os log-preços têm níveis diferentes). O intercepto `α` captura a diferença estrutural de nível entre os dois ativos.

---

## 6. Resultados

*Pipeline executado em 16/05/2026 — dados de 2015-01-02 a 2025-12-30.*

### 6.1 Resumo por Janela Anual

| Ano | Ativos | Cand. correlação | Pares cointegrados | Pregões |
|---|---|---|---|---|
| 2015 | 99 | 598 | **162** | 229 |
| 2016 | 100 | 1.905 | **256** | 241 |
| 2017 | 99 | 1.101 | **174** | 170 |
| 2018 | 98 | 472 | **94** | 225 |
| 2019 | 100 | 1.167 | **223** | 248 |
| 2020 | 100 | 1.225 | **313** | 200 |
| 2021 | 99 | 526 | **110** | 186 |
| 2022 | 99 | 463 | **109** | 249 |
| 2023 | 99 | 620 | **82** | 248 |
| 2024 | 99 | 373 | **78** | 216 |
| 2025 | 98 | 1.259 | **148** | 219 |
| **Total** | — | **9.709** | **1.749** | — |

> **1.749 registros** = par × ano. Um mesmo par pode aparecer em múltiplas janelas. **1.559 pares únicos** ao longo de todo o período.

**Leituras dos dados:**

- **2020** tem o maior número de pares cointegrados (313): o choque da COVID-19 comprimiu os ativos em movimentos conjuntos, aumentando artificialmente a correlação e a cointegração. Atenção redobrada no backtesting para esse período.
- **2016** tem o maior número de candidatos por correlação (1.905): ano de tendência fortemente direcional (impeachment + ciclo de queda de juros), que empurrou muitas séries na mesma direção — alta correlação espúria de tendência.
- **2024** tem o menor número de pares cointegrados (78): mercado mais disperso, sem tendência dominante clara; relações de cointegração mais difíceis de estabelecer.

### 6.2 Pares Mais Frequentes

Pares que aparecem como cointegrados em mais janelas anuais (mais estáveis estruturalmente):

| Par | Janelas | p-value mín. | Correlação média | Lógica econômica |
|---|---|---|---|---|
| IGTA3 × MULT3 | **5** | 0,0007 | 0,973 | Ambas são FIIs/incorporadoras de shopping centers |
| CPFE3 × EQTL3 | **4** | 0,0057 | 0,932 | Distribuidoras de energia — regulação e tarifa idênticas |
| COGN3 × LREN3 | **4** | 0,0096 | 0,925 | Varejo e serviços ao consumidor |
| BBDC3 × MULT3 | **4** | 0,0123 | 0,896 | — |
| PETR3 × PETR4 | **3** | 0,0011 | 0,996 | ON e PN da mesma empresa (Petrobras) |
| BBAS3 × CPFE3 | **3** | 0,0016 | 0,902 | — |
| CPFE3 × VBBR3 | **3** | 0,0024 | 0,860 | — |
| MULT3 × RENT3 | **3** | 0,0034 | 0,920 | Blue chips de consumo com perfil de crescimento similar |
| ITSA4 × ITUB4 | **3** | 0,0037 | 0,990 | Holding (Itaúsa) e banco controlado (Itaú Unibanco) |
| ITSA4 × MULT3 | **3** | 0,0044 | 0,910 | — |

Pares como `PETR3 × PETR4` e `ITSA4 × ITUB4` são os "pares naturais" clássicos da B3: mesma empresa em classes diferentes ou holding e controlada. São os candidatos mais fortes para o backtesting.

### 6.3 Pares Mais Fortes (menor p-value)

| Par | Ano formação | Correlação | p-value ADF | β | σ_spread |
|---|---|---|---|---|---|
| PCAR3 × YDUQ3 | 2021 | 0,803 | ~0,000000 | 1,161 | 0,117 |
| CSMG3 × CTIP3 | 2016 | 0,972 | ~0,000000 | 5,471 | 0,089 |
| IGTA3 × RUMO3 | 2016 | 0,833 | ~0,000000 | 0,312 | 0,080 |
| RADL3 × SMLS3 | 2016 | 0,953 | ~0,000000 | 0,704 | 0,053 |
| BBSE3 × EQTL3 | 2019 | 0,974 | 0,000001 | 1,019 | 0,026 |
| MULT3 × RUMO3 | 2016 | 0,811 | 0,000002 | 0,317 | 0,088 |
| EMBJ3 × RUMO3 | 2016 | −0,894 | 0,000003 | −0,530 | 0,103 |
| SBFG3 × YDUQ3 | 2020 | 0,946 | 0,000005 | 0,862 | 0,063 |
| AMER3 × YDUQ3 | 2017 | 0,976 | 0,000013 | 0,854 | 0,052 |
| LREN3 × RUMO3 | 2016 | 0,854 | 0,000018 | 0,339 | 0,080 |

> **Nota:** β negativo (ex: EMBJ3 × RUMO3) indica relação inversa entre os log-preços — quando um sobe, o outro tende a cair. A posição continua sendo hedge-neutral, porém com lógica econômica mais difícil de explicar. Atenção na interpretação do sinal de entrada.

### 6.4 Alertas Ativos — Snapshot de 12/11/2025

Estado dos z-scores com os parâmetros do último ano de formação de cada par:

| Direção | Par | Z-Score | Interpretação |
|---|---|---|---|
| SHORT | GOAU4 × SLCE3 | +3,15 | Acima do stop — monitorar |
| SHORT | MOVI3 × SMFT3 | +2,95 | Próximo ao stop |
| SHORT | ECOR3 × TIMS3 | +2,85 | Próximo ao stop |
| SHORT | ECOR3 × MULT3 | +2,67 | Sinal ativo |
| SHORT | BPAC11 × TIMS3 | +2,66 | Sinal ativo |
| SHORT | CEAB3 × PSSA3 | +2,47 | Sinal ativo |
| SHORT | EGIE3 × SRNA3 | +2,39 | Sinal ativo |
| SHORT | DIRR3 × TIMS3 | +2,26 | Sinal ativo |
| SHORT | EGIE3 × HYPE3 | +2,14 | Sinal ativo |
| SHORT | B3SA3 × YDUQ3 | +2,07 | Sinal ativo |
| SHORT | BPAC11 × NEOE3 | +2,03 | Sinal ativo |
| SHORT | ECOR3 × NEOE3 | +2,01 | Sinal ativo |
| LONG | CURY3 × ECOR3 | −2,03 | Sinal ativo |
| LONG | SRNA3 × TOTS3 | −2,18 | Sinal ativo |
| LONG | COGN3 × CPLE3 | −2,21 | Sinal ativo |
| LONG | ITSA4 × SBSP3 | −2,22 | Sinal ativo |
| LONG | ITUB4 × SBSP3 | −2,24 | Sinal ativo |
| LONG | CEAB3 × EGIE3 | −2,60 | Sinal ativo |

**12 sinais SHORT + 6 sinais LONG** ativos em 12/11/2025, de 148 pares com z-score calculado no período.

---

## 7. Outputs Gerados

Todos os arquivos são criados automaticamente em `output/` ao rodar `rodar.py`.

### CSVs — `output/`

| Arquivo | Colunas principais | Descrição |
|---|---|---|
| `resumo_janelas.csv` | Ano, N_Ativos, Candidatos_Corr, Pares_Cointegrados, Dias_Pregao, Volume_Total_MM, Top1, Top2, Top3 | Uma linha por janela anual |
| `pares_por_ano.csv` | Ano, Ativo_A, Ativo_B, Correlacao, P_Value, Alpha, Beta, Spread_Mu, Spread_Sigma | 1.749 registros — um por (par, ano) |
| `pares_frequentes.csv` | Ativo_A, Ativo_B, N_Anos, P_Value_Min, P_Value_Med, Corr_Med | Pares únicos ordenados por frequência |
| `melhores_pares.csv` | Ativo_A, Ativo_B, Ano, Correlacao, P_Value, Alpha, Beta, Spread_Sigma | Pares únicos ordenados por menor p-value |
| `persistencia_ativos.csv` | Ativo, Janelas, Rank_Medio, Volume_Acum_MM | Frequência de cada ativo no top 100 |
| `zscores.csv` | Data × Par → Z-Score | Série diária de z-score — parâmetros do último ano de formação |

### Gráficos — `output/graficos/`

| Arquivo | Descrição |
|---|---|
| `01_universo_anual.png` | Volume R$ Bi por ano + pares cointegrados vs candidatos por ano + pregões |
| `02_persistencia_ativos.png` | Top 30 ativos com mais aparições no top 100 ao longo das janelas |
| `03_pares_cointegrados.png` | Histograma de p-values + scatter correlação × p-value + contagem de pares por ano |
| `04_pares_ranking.png` | Top 20 mais frequentes (barra) + top 20 menor p-value (barra) |
| `05_parametros_ols.png` | Distribuição de β + distribuição de σ_spread + boxplot de β por ano |
| `06_zscore_top6.png` | Série histórica completa de z-score dos 6 melhores pares (menor p-value) |
| `07_spread_top4.png` | Spread bruto (log-preços) dos 4 melhores pares com bandas ±2σ |
| `08_par_destaque.png` | Análise detalhada do par #1: spread + z-score completo + detalhe 2024–atual |
| `09_sinais_temporais.png` | Evolução mensal dos sinais LONG e SHORT ativos ao longo do período |
| `10_snapshot_zscores.png` | Distribuição de todos os z-scores na data mais recente (barra + histograma) |
| `11_heatmap_pares.png` | Mapa de calor: top 30 pares × anos (verde = cointegrado, branco = ausente) |
| `12_pipeline.png` | Diagrama completo do pipeline com fases concluídas e próximas etapas |

---

## 8. Estrutura da Pasta

```
versao_consolidada/
│
├── rodar.py              ← Script único — executa todo o pipeline
├── requirements.txt      ← Dependências Python
├── README.md             ← Este arquivo
├── .gitignore            ← Exclui CSV e graficos do controle de versão
│
└── output/               ← Criada automaticamente ao rodar o script
    ├── resumo_janelas.csv
    ├── pares_por_ano.csv
    ├── pares_frequentes.csv
    ├── melhores_pares.csv
    ├── persistencia_ativos.csv
    ├── zscores.csv
    └── graficos/
        ├── 01_universo_anual.png
        ├── 02_persistencia_ativos.png
        ├── 03_pares_cointegrados.png
        ├── 04_pares_ranking.png
        ├── 05_parametros_ols.png
        ├── 06_zscore_top6.png
        ├── 07_spread_top4.png
        ├── 08_par_destaque.png
        ├── 09_sinais_temporais.png
        ├── 10_snapshot_zscores.png
        ├── 11_heatmap_pares.png
        └── 12_pipeline.png
```

O arquivo CSV de dados **não é versionado** (`.gitignore`) por ser grande demais para o git. Os gráficos gerados também são excluídos por serem reproduzíveis ao rodar `rodar.py`.

---

## 9. Próximas Etapas

### Fase 3 — Backtesting

Simular as regras de operação sobre os z-scores históricos de cada par em cada janela:

```python
# Para cada par e cada janela:
#   1. Varre a série de z-scores cronologicamente
#   2. Quando |Z| >= limiar_entrada → abre posição (registra data, z, direção)
#   3. Quando |Z| <= limiar_saida, |Z| >= stop ou t > 30d → fecha posição
#   4. Calcula: retorno, duração, resultado (win/loss)
```

**Métricas alvo por par e por janela:**

| Métrica | O que responde |
|---|---|
| Retorno médio por operação | A estratégia é lucrativa em média? |
| Win rate | % de trades que convergiram com lucro |
| Sharpe Ratio anualizado | Retorno ajustado ao risco |
| Drawdown máximo | Pior sequência de perdas acumuladas |
| Tempo médio de normalização | Quantos dias o spread leva para convergir? |
| Frequência de operações | Quantas oportunidades surgem por ano? |

**Perguntas que o backtesting deve responder:**

1. A estratégia gera retorno médio positivo no mercado brasileiro?
2. Qual limiar (1,5 / 2,0 / 2,5) maximiza o Sharpe?
3. Os pares mais frequentes (IGTA3×MULT3, PETR3×PETR4) são os mais lucrativos?
4. O desempenho foi estável entre 2015–2025 ou concentrado em períodos específicos?
5. 2020 (COVID) distorce os resultados? O modelo funciona retirando aquele ano?
6. Quais setores performam melhor?

### Fase 4 — Walk-Forward Out-of-Sample

Reformular o backtesting como walk-forward genuíno (sem look-ahead):

- **Janela de formação:** ano Y → estima α, β, μ, σ
- **Janela de operação:** ano Y+1 → aplica os parâmetros do ano anterior para calcular z-scores e simular trades

Isso elimina o viés de look-ahead e produz resultados realistas e comparáveis à literatura. É a etapa necessária para validar o modelo para uso operacional.

### Fase 5 — Monitoramento Operacional

- **Atualização diária:** pipeline para baixar os preços do dia, recalcular z-scores e gerar a lista de alertas
- **Revalidação anual dos pares:** no início de cada ano, reestimar os parâmetros de cointegração sobre os dados do ano anterior e atualizar a lista de pares válidos
- **Dashboard:** tela com posições abertas, tempo desde a entrada, P&L acumulado por par

---

## 10. Referências

- **Gatev, E., Goetzmann, W. N., & Rouwenhorst, K. G. (2006).** Pairs Trading: Performance of a Relative-Value Arbitrage Rule. *Review of Financial Studies*, 19(3), 797–827.
- **Engle, R. F., & Granger, C. W. J. (1987).** Co-integration and Error Correction: Representation, Estimation, and Testing. *Econometrica*, 55(2), 251–276.
- **Vidyamurthy, G. (2004).** *Pairs Trading: Quantitative Methods and Analysis*. Wiley Finance.
- **Avellaneda, M., & Lee, J. H. (2010).** Statistical Arbitrage in the U.S. Equities Market. *Quantitative Finance*, 10(7), 761–782.
- **Said, S. E., & Dickey, D. A. (1984).** Testing for Unit Roots in Autoregressive-Moving Average Models of Unknown Order. *Biometrika*, 71(3), 599–607.

---

*Dados: Economatica / B3 · Período: 2015-01-02 a 2025-12-30 · 11 janelas anuais · 1.559 pares únicos cointegrados*
