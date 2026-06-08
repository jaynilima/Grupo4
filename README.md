# Pairs Trading no Mercado Acionário Brasileiro
### Estudo da aplicabilidade da estratégia com Filtro de Kalman

**FEA.dev** · Jayni Bitencourt Lima · Victor Braga · 

## Sumário

1. [O que é Pairs Trading](#1-o-que-é-pairs-trading)
2. [Objetivo](#2-objetivo)
3. [Metodologia](#3-metodologia)
4. [Resultados do Backtest](#4-resultados-do-backtest)
5. [Análise Crítica](#5-análise-crítica)
6. [Estrutura do Repositório](#6-estrutura-do-repositório)
7. [Referências](#7-referências)

---

## 1. O que é Pairs Trading

Pairs trading é uma estratégia de **arbitragem estatística** que opera sobre a relação de preços entre dois ativos com vínculo econômico comprovado. Quando essa relação se rompe temporariamente, a estratégia compra o ativo "barato" e vende o ativo "caro" simultaneamente, apostando na convergência posterior.

Por operar **long e short ao mesmo tempo**, a estratégia é *market-neutral*: o retorno não depende da direção geral do mercado, mas apenas de o spread entre os dois ativos voltar ao padrão histórico.

O critério estatístico que sustenta a estratégia é a **cointegração** (Engle & Granger, 1987): dois ativos são cointegrados quando, apesar de cada preço individualmente seguir um passeio aleatório, a diferença entre eles é **estacionária** — oscila em torno de uma média com variância estável.

### Vínculos econômicos válidos

| Tipo | Exemplo B3 |
|------|-----------|
| Mesma empresa, classes diferentes (ON/PN) | PETR3 / PETR4 · GGBR3 / GGBR4 |
| Holding e controlada | BRAP4 / VALE3 · ITSA4 / ITUB4 |
| Mesmo setor e modelo de negócio | ITUB4 / BBDC4 · JBSS3 / MRFG3 |
| Mesma cadeia produtiva | VALE3 / CSNA3 · SUZB3 / KLBN11 |
| Fator macro comum | Exportadoras dolarizadas · Empresas sensíveis à Selic |
| Arbitragem direta | Ação ON na B3 vs. ADR na NYSE |

---

## 2. Objetivo

Analisar se a estratégia de pairs trading pode ser aplicada ao mercado acionário brasileiro a partir de dados históricos da B3, transformando relações estatísticas em **regras quantitativas de entrada e saída** e avaliando a performance financeira por meio de backtest walk-forward.

**Objetivos específicos:**

- Selecionar dinamicamente ativos líquidos da B3 em janelas móveis bienais
- Identificar pares cointegrados combinando distância mínima + teste de Engle-Granger
- Estimar dinamicamente o hedge ratio com **Filtro de Kalman**
- Gerar sinais de operação a partir do z-score do spread dinâmico
- Avaliar desempenho financeiro, risco e consistência temporal da estratégia
- Comparar performance entre setores da B3

---

## 3. Metodologia

O projeto está organizado em **seis notebooks** que formam um pipeline sequencial:

```
Dados brutos (Economatica)
        │
        ▼
[NB 01] Tratamento e limpeza
        │  preços ajustados, padronização, formato Parquet
        ▼
[NB 02] Enriquecimento cadastral e setorial
        │  id_papel (ISIN ou ticker), setor/subsetor/segmento
        ▼
[NB 03] Seleção dinâmica de ativos líquidos
        │  janelas de 504 pregões (~2 anos), score de liquidez por percentil
        ▼
[NB 04] Formação dos pares
        │  pré-filtro por distância mínima → teste de cointegração (p ≤ 5%)
        │  top 20 pares por janela, ordenados por p-valor e distância
        ▼
[NB 05] Modelo com Filtro de Kalman
        │  beta dinâmico, spread dinâmico, z-score adaptativo
        │  sinais: LONG / SHORT / NEUTRO
        ▼
[NB 06] Backtest walk-forward bienal
         métricas, curvas de PnL, drawdown, análise por setor/ciclo/par
```

### Regras do backtest

| Parâmetro | Valor |
|-----------|-------|
| Entrada | \|z\| ≥ 2,0 |
| Saída por convergência | \|z\| ≤ 0,5 |
| Stop por divergência | \|z\| ≥ 3,0 |
| Stop por tempo | 30 pregões |
| Custo operacional | 10 bps/lado → 20 bps round-trip |

### Walk-forward bienal

```
Ciclo 1:  Formação 2010–2011  →  Trading 2012–2013
Ciclo 2:  Formação 2012–2013  →  Trading 2014–2015
Ciclo 3:  Formação 2014–2015  →  Trading 2016–2017
Ciclo 4:  Formação 2016–2017  →  Trading 2018–2019
Ciclo 5:  Formação 2018–2019  →  Trading 2020–2021
Ciclo 6:  Formação 2020–2021  →  Trading 2022–2023
Ciclo 7:  Formação 2022–2023  →  Trading 2024–2025
```

---

## 4. Resultados do Backtest

### Métricas consolidadas (2012–2025)

| Trades | Win Rate | Sharpe | MaxDD (log-spread) | Duração Média | Retorno a.a. |
|--------|----------|--------|-------------------|---------------|--------------|
| 1.047  | 53,9%    | 0,24   | −2,049            | 4,8 pregões   | +0,92%       |

> Simulação monetária: R$ 1.000 iniciais → **R$ 1.136,38** (5% do capital por par, sem alavancagem, sem CDI do caixa não alocado).

### Performance por ciclo bienal

| Ciclo de trading | Sharpe | Win Rate | Resultado |
|-----------------|--------|----------|-----------|
| 2012–2013 | −0,007 | 63,6% | ⚪ Neutro |
| 2014–2015 | 1,31   | 49,1% | 🟢 Positivo |
| 2016–2017 | **1,87** | 65,0% | 🟢 Melhor ciclo |
| 2018–2019 | 1,49   | 50,0% | 🟢 Positivo |
| 2020–2021 | 0,32   | 54,0% | 🟢 Positivo |
| 2022–2023 | −0,13  | 44,5% | 🔴 Negativo |
| 2024–2025 | −1,64  | 46,6% | 🔴 Pior ciclo |

### Top 5 pares por Sharpe (out-of-sample)

| Par | Sharpe | Win Rate | Trades |
|-----|--------|----------|--------|
| AXIA3 × AXIA6   | 45,87 | 100% | 2  |
| GRND3 × KROT11  | 35,82 | 100% | 2  |
| VIVT3 × VIVT4   | 31,94 | 100% | 4  |
| LAME4 × RENT3   | 25,28 | 100% | 3  |
| BBDC3 × IGTA3   | 23,66 | 100% | 2  |

### Razões de saída

| Razão | Trades | % |
|-------|--------|---|
| Convergência (sinal funcionou) | 598 | 57,1% |
| Stop por divergência (spread continuou afastando) | 435 | 41,5% |
| Stop por tempo | 14 | 1,3% |

### Performance por setor

| Setor | Trades | Win Rate | Sharpe | PnL Total |
|-------|--------|----------|--------|-----------|
| Comunicações | 13 | 76,9% | 7,79 | +0,356 |
| Financeiro | 235 | 57,0% | 1,15 | +1,904 |
| Bens Industriais | 126 | 54,0% | 1,32 | +1,473 |
| Materiais Básicos | 69 | 56,5% | 0,91 | +0,130 |
| Consumo Não Cíclico | 11 | 45,5% | 0,65 | +0,033 |
| Consumo Cíclico | 433 | 51,3% | −0,18 | −0,369 |
| Utilidade Pública | 155 | 52,9% | −1,19 | −0,809 |

---

## 5. Análise Crítica

### ✅ O que funcionou

- Pares com **vínculos econômicos sólidos** (ON/PN, holding-controlada) foram os mais consistentes e com Sharpe individual muito elevado
- Ciclos 2014–2019 entregaram **Sharpe > 1,0** em todos os períodos, validando a metodologia em condições de mercado favoráveis
- O Filtro de Kalman adaptou o hedge ratio dinamicamente, capturando mudanças graduais na relação entre os ativos

### ⚠️ Limitações identificadas

**Instabilidade estrutural** — Cointegração estimada em janela histórica não é invariante. A taxa de 41,5% de stops por divergência indica que parte dos pares perdeu a propriedade de reversão no período de trading, especialmente pós-2022.

**Viés de seleção** — A escolha dos top-20 pares por ranking favorece os pares que "mais pareciam" cointegrados no treino, sem garantia out-of-sample. O "efeito publicação" de Gatev et al. (2006) — queda da rentabilidade após divulgação da estratégia — pode estar em curso no Brasil à medida que estratégias quant se tornam mais prevalentes.

**Custos e liquidez** — O modelo assume 20 bps flat por operação, sem modelar bid-ask real nem impacto de mercado. Para pares menos líquidos o custo efetivo pode ser substancialmente maior, especialmente em saídas por stop (quando o mercado tende a ser mais ilíquido).

**Caudas pesadas** — 65% dos pares exibem curtose > 3; alguns ultrapassam 40. Modelos de risco gaussianos subestimam o risco real. Limites de posição baseados em VaR normal são inadequados.

**Ausência de benchmark** — O retorno de +0,92% a.a. sobre o capital total parece modesto frente à Selic. A comparação justa incluiria CDI sobre os 95% não alocados + receita de aluguel da ponta short — ambos omitidos na simulação.

### 🔧 Próximos passos sugeridos

- Renovação contínua dos testes de cointegração **durante** o período de trading
- Limiares de entrada/stop **adaptativos** por volatilidade realizada ou quantis empíricos
- Teto de concentração por setor e por par
- Dimensionamento de risco por CVaR histórico
- Comparação formal com benchmarks (CDI, IBOVESPA, carteira 60/40)
- Análise por clusters estatísticos e regimes de mercado

---

## 6. Estrutura do Repositório

```
.
├── 01_tratar_dados_economatica_B3.ipynb   # Limpeza e padronização da base bruta
├── 02_dado_economatica_B3_addsetores.ipynb # Enriquecimento cadastral e setorial
├── 03_universo_ativos_líquidos.ipynb       # Seleção dinâmica de ativos líquidos
├── 04_formação_pares.ipynb                 # Formação dos pares (distância + cointegração)
├── 05_modelo.ipynb                         # Filtro de Kalman e geração de sinais
├── 06_backtest.ipynb                       # Backtest e métricas de performance
│
├── backtest/
│   ├── graficos/                           # 9 gráficos do backtest
│   ├── metricas_ano.csv                    # Performance por ano
│   ├── metricas_ciclo.csv                  # Performance por ciclo bienal
│   ├── metricas_par.csv                    # Performance por par
│   ├── metricas_setor.csv                  # Performance por setor
│   ├── resumo_ciclos.csv                   # Resumo dos ciclos walk-forward
│   ├── skewness_diagnostico.csv            # Diagnóstico estatístico dos z-scores
│   └── trades.csv                          # Base completa de trades
│
├── liquidez_historica.parquet              # Universo líquido por janela mensal
└── pares_top20_cointegracao.parquet        # Top 20 pares por janela de formação
```

---

## 7. Referências

ALEXANDER, C. Optimal hedging using cointegration. *Philosophical Transactions of the Royal Society of London A*, v. 357, n. 1758, p. 2039–2058, 1999.

DICKEY, D. A.; FULLER, W. A. Distribution of the estimators for autoregressive time series with a unit root. *Journal of the American Statistical Association*, v. 74, n. 366a, p. 427–431, 1979.

ENGLE, R. F.; GRANGER, C. W. J. Co-integration and error correction: representation, estimation, and testing. *Econometrica*, v. 55, n. 2, p. 251–276, 1987.

GATEV, E.; GOETZMANN, W. N.; ROUWENHORST, K. G. Pairs trading: performance of a relative-value arbitrage rule. *Review of Financial Studies*, v. 19, n. 3, p. 797–827, 2006.

HAMILTON, J. D. *Time Series Analysis*. Princeton: Princeton University Press, 1994.

KALMAN, R. E. A new approach to linear filtering and prediction problems. *Journal of Basic Engineering*, v. 82, n. 1, p. 35–45, 1960.

POLE, A. *Statistical Arbitrage: Algorithmic Trading Insights and Techniques*. Hoboken: Wiley, 2007.

VIDYAMURTHY, G. *Pairs Trading: Quantitative Methods and Analysis*. Hoboken: Wiley, 2004.
