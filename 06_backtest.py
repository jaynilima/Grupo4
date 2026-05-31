"""
Pairs Trading B3 — Backtest Walk-Forward Bienal
================================================
Ciclo:
  Formação : 504 pregões até 31/Dez do ano Y  (Y ímpar: 2011, 2013, 2015 ...)
             → liquidez (nb03) + distância mínima (nb04)
             → Engle-Granger + Kalman → estado final (theta, P, R)
  Trading  : 2 anos completos (Y+1 e Y+2)
             → Kalman continua ONLINE a cada pregão
             → aplica thresholds de entrada / saída / stop

Exemplo:
  Form 2010-2011 → Trade 2012-2013
  Form 2012-2013 → Trade 2014-2015
  Form 2014-2015 → Trade 2016-2017
  Form 2016-2017 → Trade 2018-2019
  Form 2018-2019 → Trade 2020-2021
  Form 2020-2021 → Trade 2022-2023
  Form 2022-2023 → Trade 2024-2025

Outputs: output/backtest/
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from scipy.stats import skew, kurtosis
from statsmodels.tsa.stattools import coint

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURAÇÃO
# ─────────────────────────────────────────────────────────────────────────────

NOME_CSV         = "dados_economatica_B3.csv"
NOME_CSV_SETORES = "economatica_B3_setores.csv"

ANO_DADOS_INI    = 2010    # início do carregamento (precisa de 2 anos antes do 1º trade)
ANO_FORM_INI     = 2011    # 1º ano de corte de formação (usa dados 2010-2011)
ANO_TRADE_FIM    = 2025    # último ano de trading

JANELA_FORMACAO  = 504     # pregões por janela ≈ 2 anos (notebook 03/04)
COBERTURA_MIN    = 0.80
OBS_MINIMAS_PAR  = int(JANELA_FORMACAO * COBERTURA_MIN)  # 403
MAX_CAND_DIST    = 500
TOP_N_PARES      = 20
PVALUE_MAX       = 0.05
RAPIDO           = False   # True → maxlag=5 no ADF

# Kalman
KALMAN_DELTA     = 1e-5
KALMAN_VE        = None
KALMAN_N_WARM    = 50

# Thresholds de trading
ENTRADA_Z        = 2.0
SAIDA_Z          = 0.5
STOP_Z           = 3.0
STOP_DIAS        = 30
CUSTO_BPS        = 10      # por lado (entrada + saída = 2 lados)

FILTRAR_SETOR    = True    # mesmo que 05_modelo.py — pares do mesmo setor

# Estrutura dos ciclos: PASSO=2 → bienal (padrão metodológico)
PASSO_CICLO      = 2      # anos entre rebalanceamentos de formação
ANOS_TRADE       = 2      # anos de trading por ciclo


# ─────────────────────────────────────────────────────────────────────────────
#  LOCALIZAÇÃO
# ─────────────────────────────────────────────────────────────────────────────

def _encontrar(nome):
    raiz = os.path.dirname(os.path.abspath(__file__))
    casa = os.path.expanduser("~")
    for pasta in [raiz,
                  os.path.join(casa, "Desktop"),
                  os.path.join(casa, "Documents"),
                  os.path.join(casa, "Downloads")]:
        p = os.path.join(pasta, nome)
        if os.path.isfile(p):
            return p
    return None

CSV_PATH     = _encontrar(NOME_CSV)
SETORES_PATH = _encontrar(NOME_CSV_SETORES)
PASTA_SCRIPT = os.path.dirname(os.path.abspath(__file__))
BT_DIR       = os.path.join(PASTA_SCRIPT, "output", "backtest")
BT_CHARTS    = os.path.join(BT_DIR, "graficos")
os.makedirs(BT_CHARTS, exist_ok=True)

SEP = "=" * 66

# geração de ciclos parametrizada
CICLOS = []
ano_form = ANO_FORM_INI
while True:
    ano_t_ini = ano_form + 1
    ano_t_fim = ano_form + ANOS_TRADE
    if ano_t_fim > ANO_TRADE_FIM:
        break
    CICLOS.append((ano_form, ano_t_ini, ano_t_fim))
    ano_form += PASSO_CICLO

modo_str = f"anual" if PASSO_CICLO == 1 else f"bienal"
print(f"\n{SEP}\n  BACKTEST WALK-FORWARD ({modo_str.upper()}) — Kalman Pairs Trading B3\n{SEP}")
if not CSV_PATH:
    print(f"  [ERRO] CSV não encontrado: {NOME_CSV}")
    sys.exit(1)
print(f"  [OK] Dados  : {CSV_PATH}")
if SETORES_PATH:
    print(f"  [OK] Setores: {SETORES_PATH}")
print(f"\n  Ciclos {modo_str}s ({len(CICLOS)} total):")
for f, t1, t2 in CICLOS:
    suf = f"{t1}" if t1 == t2 else f"{t1}–{t2}"
    print(f"    Form {f-1}–{f}  →  Trade {suf}")
print(f"\n  Janela  : {JANELA_FORMACAO} pregões  |  Top-{TOP_N_PARES}/ciclo")
print(f"  Entrada |Z|≥{ENTRADA_Z}  Saída |Z|≤{SAIDA_Z}  "
      f"Stop |Z|≥{STOP_Z} ou >{STOP_DIAS}d")
print(f"  Custo   : {CUSTO_BPS}bps/lado  |  Kalman δ={KALMAN_DELTA:.0e}")
print(f"{SEP}\n")


# ─────────────────────────────────────────────────────────────────────────────
#  PALETA
# ─────────────────────────────────────────────────────────────────────────────

C = dict(azul="#1f4e79", azul2="#2e75b6", verde="#1e7e34",
         verm="#c0392b", laran="#e67e22", cinza="#7f8c8d",
         amar="#f1c40f", roxo="#8e44ad", fundo="#f8f9fa", grade="#dde1e7")
plt.rcParams.update({
    "figure.facecolor": C["fundo"], "axes.facecolor": C["fundo"],
    "axes.grid": True, "grid.color": C["grade"], "grid.linewidth": 0.6,
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 12, "axes.labelsize": 10,
})


# ─────────────────────────────────────────────────────────────────────────────
#  1. SETORES
# ─────────────────────────────────────────────────────────────────────────────

def _carregar_setores(caminho):
    if not caminho:
        return {}
    mapa = {}
    try:
        with open(caminho, encoding="latin-1") as f:
            f.readline()
            for linha in f:
                p = linha.strip().split('","')
                if len(p) < 10:
                    continue
                ticker = p[0].lstrip('"').replace("<XBSP>", "").strip()
                def _v(s):
                    s = s.rstrip('"').strip()
                    return None if s in ("-", "") else s
                if ticker:
                    mapa[ticker] = {"setor":    _v(p[7]),
                                    "subsetor": _v(p[8])}
    except Exception:
        pass
    return mapa

SETORES = _carregar_setores(SETORES_PATH)


# ─────────────────────────────────────────────────────────────────────────────
#  2. DADOS BRUTOS
# ─────────────────────────────────────────────────────────────────────────────

print(f"[1/4] Carregando dados ({ANO_DADOS_INI}–{ANO_TRADE_FIM})...")
t0 = time.time()

_COLS = ["Ativo", "Data", "Fechamento", "Abertura", "Minimo",
         "Maximo", "Medio", "Q_Negs", "Volume_BRL_k", "Q_Titulos_k"]

chunks = []
for _ck in pd.read_csv(CSV_PATH, encoding="latin-1", names=_COLS,
                       header=0, chunksize=200_000, low_memory=False):
    _ck["Ativo"] = (_ck["Ativo"].astype(str)
                    .str.replace("<XBSP>", "", regex=False).str.strip())
    _ck["Data"]  = pd.to_datetime(_ck["Data"], errors="coerce")
    _ck = _ck[_ck["Data"].dt.year.between(ANO_DADOS_INI, ANO_TRADE_FIM)].copy()
    for col in ("Fechamento", "Volume_BRL_k", "Q_Negs"):
        _ck[col] = pd.to_numeric(
            _ck[col].astype(str).str.replace(",", ".", regex=False)
                    .replace("-", np.nan), errors="coerce")
    _ck = _ck.dropna(subset=["Fechamento"])
    _ck = _ck[_ck["Fechamento"] > 0]
    _ck["Volume_BRL_k"] = _ck["Volume_BRL_k"].fillna(0.0)
    _ck["Q_Negs"]       = _ck["Q_Negs"].fillna(0.0)
    chunks.append(_ck[["Ativo", "Data", "Fechamento", "Volume_BRL_k", "Q_Negs"]])

df_raw = pd.concat(chunks, ignore_index=True)
del chunks

# calendário de pregões
calendario = np.array(sorted(df_raw["Data"].dropna().unique()),
                      dtype="datetime64[ns]")
cal_list   = list(pd.DatetimeIndex(calendario))
cal_pos    = {d: i for i, d in enumerate(cal_list)}

print(f"      {len(df_raw):,} linhas | "
      f"{df_raw['Data'].min().date()}–{df_raw['Data'].max().date()} | "
      f"{df_raw['Ativo'].nunique()} ativos | {time.time()-t0:.0f}s\n")


# ─────────────────────────────────────────────────────────────────────────────
#  FUNÇÕES DO PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def selecionar_liquidez(df_janela, tot_preg):
    """Replica notebook 03: cobertura ≥80%, volume>0, q_negs>0, remove Q25."""
    stats = df_janela.groupby("Ativo").agg(
        Dias     =("Data",        "nunique"),
        Vol_Med  =("Volume_BRL_k","median"),
        Negs_Med =("Q_Negs",     "median"),
    ).reset_index()
    stats = stats[
        (stats["Dias"] / tot_preg >= COBERTURA_MIN) &
        (stats["Vol_Med"]  > 0) &
        (stats["Negs_Med"] > 0)
    ]
    if len(stats) >= 4:
        stats = stats[
            (stats["Vol_Med"]  >= stats["Vol_Med"].quantile(0.25)) &
            (stats["Negs_Med"] >= stats["Negs_Med"].quantile(0.25))
        ]
    return stats["Ativo"].tolist()


def distancia_candidatos(precos_raw, cols):
    """Replica notebook 04: distância pairwise sem imputação."""
    precos_dist = precos_raw[cols]
    primeiro = precos_dist.apply(
        lambda c: c.dropna().iloc[0] if c.dropna().shape[0] > 0 else np.nan
    )
    norm_arr = (precos_dist / primeiro).values.astype(np.float64)
    N = len(cols)
    i_idx, j_idx = np.triu_indices(N, k=1)
    dists = np.full(len(i_idx), np.inf)
    for k, (ii, jj) in enumerate(zip(i_idx, j_idx)):
        ca, cb = norm_arr[:, ii], norm_arr[:, jj]
        mask   = ~(np.isnan(ca) | np.isnan(cb))
        if mask.sum() < OBS_MINIMAS_PAR:
            continue
        dists[k] = np.sum((ca[mask] - cb[mask]) ** 2)
    validos = np.where(np.isfinite(dists))[0]
    n_cand  = min(MAX_CAND_DIST, len(validos))
    if n_cand == 0:
        return []
    top_rel = np.argpartition(dists[validos], n_cand - 1)[:n_cand]
    top_rel = top_rel[np.argsort(dists[validos][top_rel])]
    top_pos = validos[top_rel]
    return [(cols[i_idx[p]], cols[j_idx[p]], float(dists[p])) for p in top_pos]


def kalman_formacao(log_a, log_b, delta=1e-5, ve=None, n_warm=50):
    """
    Kalman Filter na janela de formação.
    Retorna estado final (theta, P, R) para continuar online no trading.
    """
    T  = len(log_a)
    Q  = delta * np.eye(2)
    n_w = max(2, min(n_warm, T // 4))
    xw, yw = log_b[:n_w], log_a[:n_w]
    sb, sa   = xw.sum(), yw.sum()
    sbb, sab = (xw * xw).sum(), (yw * xw).sum()
    den = n_w * sbb - sb * sb
    if abs(den) > 1e-12 and n_w >= 5:
        beta0  = (n_w * sab - sb * sa) / den
        alpha0 = (sa - beta0 * sb) / n_w
        resid  = yw - alpha0 - beta0 * xw
        R = max(resid.var(), 1e-8) if ve is None else max(ve, 1e-8)
    else:
        alpha0, beta0, R = 0.0, 1.0, 0.001

    theta = np.array([alpha0, beta0], dtype=np.float64)
    P     = np.eye(2, dtype=np.float64) * max(R, 1e-4)
    for t in range(T):
        H      = np.array([1.0, log_b[t]])
        P_pred = P + Q
        nu     = log_a[t] - (H @ theta)
        S      = float(H @ P_pred @ H) + R
        K      = (P_pred @ H) / S
        theta  = theta + K * nu
        P      = (np.eye(2) - np.outer(K, H)) @ P_pred
    return theta.copy(), P.copy(), R


def kalman_online(log_a_t, log_b_t, theta, P, R, delta):
    """
    Um passo Kalman online — usado a cada pregão do trading.
    Retorna: z_t, nu_t, beta_t, theta_t+1, P_t+1
    """
    Q      = delta * np.eye(2)
    H      = np.array([1.0, log_b_t])
    P_pred = P + Q
    nu     = log_a_t - (H @ theta)
    S      = float(H @ P_pred @ H) + R
    K      = (P_pred @ H) / S
    theta  = theta + K * nu
    P      = (np.eye(2) - np.outer(K, H)) @ P_pred
    z      = nu / max(S ** 0.5, 1e-10)
    return float(z), float(nu), float(theta[1]), theta, P


# ─────────────────────────────────────────────────────────────────────────────
#  3. WALK-FORWARD BIENAL
# ─────────────────────────────────────────────────────────────────────────────

print("[2/4] Walk-forward bienal...")

todos_trades   = []
resumo_ciclos  = []
diag_zscore    = []

for (ano_form, ano_t1, ano_t2) in CICLOS:

    t_ciclo = time.time()
    print(f"\n  Ciclo Form {ano_form-1}–{ano_form} → Trade {ano_t1}–{ano_t2}")

    # ── A. JANELA DE FORMAÇÃO: 504 pregões até 31/Dez/ano_form ─────────────
    datas_ate = [d for d in cal_list if d.year <= ano_form]
    if not datas_ate:
        continue
    data_fim_form = max(datas_ate)
    pos_fim = cal_pos.get(data_fim_form, len(cal_list) - 1)
    pos_ini = max(0, pos_fim - JANELA_FORMACAO + 1)
    data_ini_form = cal_list[pos_ini]

    df_form  = df_raw[(df_raw["Data"] >= data_ini_form) &
                      (df_raw["Data"] <= data_fim_form)]
    tot_preg = df_form["Data"].nunique()

    print(f"    Form: {data_ini_form.date()} → {data_fim_form.date()} "
          f"({tot_preg} pregões)")

    # liquidez (notebook 03)
    tickers = selecionar_liquidez(df_form, tot_preg)
    if len(tickers) < 2:
        print("    [AVISO] Ativos líquidos insuficientes")
        continue

    # matrizes de preços
    df_p       = df_form[df_form["Ativo"].isin(tickers)]
    precos_raw = (df_p.pivot_table(index="Data", columns="Ativo",
                                   values="Fechamento", aggfunc="last")
                      .sort_index().ffill(limit=3))
    precos_raw = precos_raw.dropna(axis=1, thresh=int(tot_preg * COBERTURA_MIN))
    precos     = precos_raw.dropna()
    cols       = precos.columns.tolist()
    N          = len(cols)
    if N < 2:
        continue

    logP = np.log(np.clip(precos.values.astype(np.float64), 1e-10, None))

    # distância mínima (notebook 04)
    candidatos = distancia_candidatos(precos_raw, cols)
    if FILTRAR_SETOR:
        candidatos = [(a, b, d) for a, b, d in candidatos
                      if SETORES.get(a, {}).get("setor") is not None
                      and SETORES.get(a, {}).get("setor") ==
                          SETORES.get(b, {}).get("setor")]

    # cointegração Engle-Granger + Kalman na formação
    col_idx  = {c: i for i, c in enumerate(cols)}
    pares_ok = []
    for a, b, dist in candidatos:
        ia, ib = col_idx[a], col_idx[b]
        ya, xb = logP[:, ia], logP[:, ib]
        try:
            if RAPIDO:
                _, pval, _ = coint(ya, xb, maxlag=5, autolag=None)
            else:
                _, pval, _ = coint(ya, xb)
        except Exception:
            continue
        if pval > PVALUE_MAX:
            continue

        theta0, P0, R0 = kalman_formacao(
            ya, xb, delta=KALMAN_DELTA,
            ve=KALMAN_VE, n_warm=KALMAN_N_WARM
        )
        pares_ok.append({
            "a": a, "b": b, "dist": dist, "pval": pval,
            "theta": theta0, "P": P0, "R": R0,
            "setor_a": SETORES.get(a, {}).get("setor"),
            "setor_b": SETORES.get(b, {}).get("setor"),
        })

    pares_ok.sort(key=lambda x: (x["pval"], x["dist"]))
    pares_ok = pares_ok[:TOP_N_PARES]

    print(f"    {len(tickers)} ativos líquidos | "
          f"{len(candidatos)} candidatos dist | "
          f"{len(pares_ok)} pares validados")

    if not pares_ok:
        continue

    # ── B. JANELA DE TRADING: 2 anos (ano_t1 e ano_t2) ────────────────────
    tickers_trade = list({p["a"] for p in pares_ok} |
                         {p["b"] for p in pares_ok})
    df_trade = df_raw[
        (df_raw["Data"].dt.year.between(ano_t1, ano_t2)) &
        (df_raw["Ativo"].isin(tickers_trade))
    ]
    if df_trade.empty:
        continue

    precos_trade = (df_trade.pivot_table(index="Data", columns="Ativo",
                                          values="Fechamento", aggfunc="last")
                             .sort_index().ffill(limit=3))
    datas_trade  = precos_trade.index.tolist()

    if len(datas_trade) < 10:
        continue

    custo_total = 2 * CUSTO_BPS / 10_000   # entrada + saída, 2 lados

    n_trades_ciclo = 0

    # ── C. SIMULAÇÃO DIA A DIA ────────────────────────────────────────────
    for par in pares_ok:
        a, b = par["a"], par["b"]
        if a not in precos_trade.columns or b not in precos_trade.columns:
            continue

        # estado inicial = estado FINAL da formação (Kalman continua de onde parou)
        theta = par["theta"].copy()
        P     = par["P"].copy()
        R     = par["R"]

        posicao      = None
        entrada_z    = None
        entrada_dia  = None
        entrada_loga = None
        entrada_logb = None
        entrada_beta = None
        dias_em_pos  = 0

        z_hist = []   # para diagnóstico de distribuição

        for dia in datas_trade:
            pa = precos_trade.at[dia, a] if a in precos_trade.columns else np.nan
            pb = precos_trade.at[dia, b] if b in precos_trade.columns else np.nan
            if np.isnan(pa) or np.isnan(pb) or pa <= 0 or pb <= 0:
                z_hist.append(np.nan)
                continue

            log_a_t = np.log(pa)
            log_b_t = np.log(pb)

            # passo Kalman online
            z_t, nu_t, beta_t, theta, P = kalman_online(
                log_a_t, log_b_t, theta, P, R, KALMAN_DELTA
            )
            z_hist.append(z_t)

            if posicao is not None:
                dias_em_pos += 1

            # ── regras de trading ─────────────────────────────────────────
            if posicao is None:
                if z_t >= ENTRADA_Z:
                    posicao      = "short"
                    entrada_z    = z_t
                    entrada_dia  = dia
                    entrada_loga = log_a_t
                    entrada_logb = log_b_t
                    entrada_beta = beta_t
                    dias_em_pos  = 0
                elif z_t <= -ENTRADA_Z:
                    posicao      = "long"
                    entrada_z    = z_t
                    entrada_dia  = dia
                    entrada_loga = log_a_t
                    entrada_logb = log_b_t
                    entrada_beta = beta_t
                    dias_em_pos  = 0
            else:
                fechar = (
                    abs(z_t) <= SAIDA_Z or
                    abs(z_t) >= STOP_Z  or
                    dias_em_pos >= STOP_DIAS
                )
                if fechar:
                    ret_a      = log_a_t - entrada_loga
                    ret_b      = log_b_t - entrada_logb
                    spread_ret = ret_a - entrada_beta * ret_b
                    if posicao == "short":
                        spread_ret = -spread_ret
                    pnl_liq = spread_ret - custo_total

                    razao = "convergencia"
                    if abs(z_t) >= STOP_Z:
                        razao = "stop_divergencia"
                    elif dias_em_pos >= STOP_DIAS:
                        razao = "stop_tempo"

                    todos_trades.append({
                        "ciclo_form":   f"{ano_form-1}-{ano_form}",
                        "ano_trade":    dia.year,
                        "Ativo_A":      a,
                        "Ativo_B":      b,
                        "Setor_A":      par["setor_a"],
                        "Setor_B":      par["setor_b"],
                        "Mesmo_Setor":  par["setor_a"] == par["setor_b"]
                                        if par["setor_a"] else False,
                        "P_Value":      par["pval"],
                        "Distancia":    par["dist"],
                        "direcao":      posicao,
                        "data_entrada": entrada_dia,
                        "data_saida":   dia,
                        "dias_abertos": dias_em_pos,
                        "z_entrada":    entrada_z,
                        "z_saida":      z_t,
                        "beta_entrada": entrada_beta,
                        "beta_saida":   beta_t,
                        "pnl_bruto":    round(spread_ret,        6),
                        "pnl_liquido":  round(pnl_liq,           6),
                        "razao_saida":  razao,
                    })
                    posicao     = None
                    dias_em_pos = 0
                    n_trades_ciclo += 1

        # diagnóstico z-score no período de trading
        zs = [z for z in z_hist if not np.isnan(z)]
        if len(zs) > 20:
            diag_zscore.append({
                "ciclo_form": f"{ano_form-1}-{ano_form}",
                "ano_t1":     ano_t1, "ano_t2": ano_t2,
                "Ativo_A":    a, "Ativo_B": b,
                "Setor_A":    par["setor_a"],
                "z_media":    float(np.mean(zs)),
                "z_std":      float(np.std(zs)),
                "z_skew":     float(skew(zs)),
                "z_kurt":     float(kurtosis(zs)),
                "pval_form":  par["pval"],
            })

    resumo_ciclos.append({
        "ciclo_form":  f"{ano_form-1}-{ano_form}",
        "trade_ini":   ano_t1,
        "trade_fim":   ano_t2,
        "n_pares":     len(pares_ok),
        "n_trades":    n_trades_ciclo,
        "n_ativos_liq": len(tickers),
    })

    print(f"    {n_trades_ciclo} trades | {time.time()-t_ciclo:.0f}s")

print(f"\n  Total: {len(todos_trades)} trades em {len(CICLOS)} ciclos\n")


# ─────────────────────────────────────────────────────────────────────────────
#  4. MÉTRICAS E CSVs
# ─────────────────────────────────────────────────────────────────────────────

print("[3/4] Calculando métricas e salvando CSVs...")

trades_df = pd.DataFrame(todos_trades)
diag_df   = pd.DataFrame(diag_zscore)
res_df    = pd.DataFrame(resumo_ciclos)

if trades_df.empty:
    print("  [AVISO] Nenhum trade gerado. Verifique parâmetros ou dados.")
    sys.exit(0)

trades_df["data_entrada"] = pd.to_datetime(trades_df["data_entrada"])
trades_df["data_saida"]   = pd.to_datetime(trades_df["data_saida"])


def metricas_grupo(g):
    n     = len(g)
    n_win = (g["pnl_liquido"] > 0).sum()
    pnl   = g["pnl_liquido"]
    sharpe = (pnl.mean() / pnl.std() * np.sqrt(252)) if pnl.std() > 0 else 0
    # drawdown em unidades absolutas (cumsum, não cumprod)
    cum_pnl = pnl.cumsum()
    dd      = (cum_pnl - cum_pnl.cummax()).min()
    return pd.Series({
        "N_Trades":    n,
        "Win_Rate":    round(n_win / n, 4) if n > 0 else 0,
        "PnL_Medio":   round(pnl.mean(), 6),
        "PnL_Total":   round(pnl.sum(),  6),
        "Sharpe":      round(sharpe, 4),
        "MaxDD":       round(dd, 4),
        "Dur_Media":   round(g["dias_abertos"].mean(), 1),
        "PnL_Bruto":   round(g["pnl_bruto"].sum(), 6),
        "Custo_Total": round((g["pnl_bruto"] - g["pnl_liquido"]).sum(), 6),
    })


metr_par    = (trades_df.groupby(["Ativo_A","Ativo_B"])
               .apply(metricas_grupo).reset_index()
               .sort_values("Sharpe", ascending=False))

metr_ciclo  = (trades_df.groupby("ciclo_form")
               .apply(metricas_grupo).reset_index())

metr_ano    = (trades_df.groupby("ano_trade")
               .apply(metricas_grupo).reset_index()
               .rename(columns={"ano_trade": "Ano"}))

metr_setor  = None
if "Setor_A" in trades_df.columns and trades_df["Setor_A"].notna().any():
    metr_setor = (trades_df.dropna(subset=["Setor_A"])
                  .groupby("Setor_A").apply(metricas_grupo)
                  .reset_index().rename(columns={"Setor_A": "Setor"})
                  .sort_values("Sharpe", ascending=False))

# PnL diário: média por par (para Sharpe) e soma total (para equity curve)
retorno_diario = (trades_df.groupby("data_saida")["pnl_liquido"]
                  .mean().rename("ret"))
# equity curve = soma cumulativa do PnL diário (log-spread, não composto)
equity_curve   = (trades_df.groupby("data_saida")["pnl_liquido"]
                  .sum().cumsum())
sharpe_total   = (retorno_diario.mean() / retorno_diario.std() * np.sqrt(252)
                  if retorno_diario.std() > 0 else 0)
# drawdown em unidades absolutas de log-spread (não faz sentido % aqui)
dd_series      = equity_curve - equity_curve.cummax()
max_dd         = dd_series.min()   # ex: -2.05 log-spread units
win_rate       = (trades_df["pnl_liquido"] > 0).mean()

# CSVs
para_salvar = {
    "trades.csv":              trades_df,
    "metricas_par.csv":        metr_par,
    "metricas_ciclo.csv":      metr_ciclo,
    "metricas_ano.csv":        metr_ano,
    "resumo_ciclos.csv":       res_df,
    "skewness_diagnostico.csv": diag_df,
}
if metr_setor is not None:
    para_salvar["metricas_setor.csv"] = metr_setor

for nome, df_out in para_salvar.items():
    df_out.to_csv(os.path.join(BT_DIR, nome), index=False)

print(f"      {len(trades_df)} trades | Sharpe={sharpe_total:.2f} | "
      f"WinRate={win_rate:.1%} | MaxDD={max_dd:.4f} (log-spread)")
print(f"      CSVs em: output/backtest/")


# ─────────────────────────────────────────────────────────────────────────────
#  5. GRÁFICOS
# ─────────────────────────────────────────────────────────────────────────────

print("\n[4/4] Gerando gráficos...")

def _salvar(fig, nome):
    fig.savefig(os.path.join(BT_CHARTS, nome), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"      {nome}")


# BT-01: retorno cumulativo + drawdown
fig, axes = plt.subplots(2, 1, figsize=(15, 9),
                         gridspec_kw={"height_ratios": [3, 1]},
                         sharex=True)
fig.suptitle(
    f"PnL Acumulado — Walk-Forward Bienal Kalman  (market-neutral)\n"
    f"Sharpe={sharpe_total:.2f}  |  MaxDD={max_dd:.4f} log-spread  |  "
    f"WinRate={win_rate:.1%}  |  Trades={len(trades_df)}",
    fontweight="bold", fontsize=12)

ax = axes[0]
ax.plot(equity_curve.index, equity_curve.values,
        color=C["azul2"], linewidth=1.8, label="Estratégia (pairs trading)")
ax.fill_between(equity_curve.index, equity_curve.values, 0,
                where=(equity_curve.values >= 0),
                alpha=0.14, color=C["verde"])
ax.fill_between(equity_curve.index, equity_curve.values, 0,
                where=(equity_curve.values < 0),
                alpha=0.14, color=C["verm"])
ax.axhline(0, linestyle=":", color=C["cinza"], linewidth=0.9)
cores_ciclo = [C["azul2"], C["roxo"], C["verde"], C["laran"],
               C["amar"], C["cinza"], C["verm"]]
for i, (_, t1, t2) in enumerate(CICLOS):
    ax.axvspan(pd.Timestamp(f"{t1}-01-01"),
               pd.Timestamp(f"{t2}-12-31"),
               alpha=0.04, color=cores_ciclo[i % len(cores_ciclo)])
    ax.text(pd.Timestamp(f"{(t1+t2)//2}-06-01"), 0.05,
            f"C{i+1}", fontsize=7.5, color=C["cinza"],
            ha="center", va="bottom",
            transform=ax.get_xaxis_transform())
ax.set_ylabel("PnL Acumulado (log-spread)")
ax.legend(loc="upper left", fontsize=9)

ax2 = axes[1]
ax2.fill_between(dd_series.index, dd_series.values, 0,
                 alpha=0.65, color=C["verm"])
ax2.plot(dd_series.index, dd_series.values,
         color=C["verm"], linewidth=0.9)
ax2.axhline(0, linestyle="-", color=C["cinza"], linewidth=0.7)
ax2.set_ylabel("Drawdown (log-spread)")
ax2.set_title(f"Drawdown  |  Máximo = {max_dd:.4f}", fontsize=9)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax2.xaxis.set_major_locator(mdates.YearLocator())
plt.tight_layout()
_salvar(fig, "bt_01_retorno_acumulado.png")


# BT-02: Sharpe por par (top 20)
top20 = metr_par.head(20).copy()
top20["Par"] = top20["Ativo_A"] + " × " + top20["Ativo_B"]
cbs = [C["verde"] if s > 1.0 else C["azul2"] if s > 0.5
       else C["amar"] if s > 0 else C["verm"]
       for s in top20["Sharpe"]]
fig, ax = plt.subplots(figsize=(12, 7))
ax.barh(top20["Par"][::-1], top20["Sharpe"][::-1],
        color=list(reversed(cbs)), edgecolor="white")
ax.axvline(0, linestyle="-",  color=C["cinza"], linewidth=0.8)
ax.axvline(1, linestyle="--", color=C["verde"], linewidth=1.0, label="Sharpe=1")
ax.set_title("Sharpe por Par — Walk-Forward Out-of-Sample",
             fontweight="bold")
ax.set_xlabel("Sharpe Ratio Anualizado")
for v, yp in zip(top20["Sharpe"][::-1], range(len(top20))):
    ax.text(v + 0.01, yp, f"{v:.2f}", va="center", fontsize=8)
ax.legend()
plt.tight_layout()
_salvar(fig, "bt_02_sharpe_pares.png")


# BT-03: distribuição de retornos e duração
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Distribuição de Retornos por Trade", fontsize=13, fontweight="bold")
sk_t = skew(trades_df["pnl_liquido"].dropna())
kt_t = kurtosis(trades_df["pnl_liquido"].dropna())
axes[0].hist(trades_df["pnl_liquido"], bins=40,
             color=C["azul2"], edgecolor="white")
axes[0].axvline(0, linestyle="--", color=C["verm"], linewidth=1.5)
axes[0].axvline(trades_df["pnl_liquido"].mean(), linestyle="-",
                color=C["verde"], linewidth=1.5,
                label=f"Média={trades_df['pnl_liquido'].mean():.4f}")
axes[0].set_title(f"PnL Líquido  |  Skew={sk_t:.2f}  Kurt={kt_t:.2f}")
axes[0].legend()
axes[1].hist(trades_df["dias_abertos"], bins=30,
             color=C["roxo"], edgecolor="white")
axes[1].axvline(trades_df["dias_abertos"].mean(), linestyle="--",
                color=C["laran"], linewidth=1.5,
                label=f"Média={trades_df['dias_abertos'].mean():.1f}d")
axes[1].set_title("Duração dos Trades (pregões)")
axes[1].legend()
plt.tight_layout()
_salvar(fig, "bt_03_distribuicao_retornos.png")


# BT-04: drawdown
fig, ax = plt.subplots(figsize=(15, 4))
ax.fill_between(dd_series.index, dd_series.values, 0,
                alpha=0.6, color=C["verm"])
ax.plot(dd_series.index, dd_series.values, color=C["verm"], linewidth=1.0)
ax.axhline(0, linestyle="-", color=C["cinza"], linewidth=0.8)
ax.set_title(f"Drawdown  |  Máximo={max_dd:.4f} (log-spread)", fontweight="bold")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.xaxis.set_major_locator(mdates.YearLocator())
plt.tight_layout()
_salvar(fig, "bt_04_drawdown.png")


# BT-05: performance anual absoluta (PnL + win rate)
anos_str  = metr_ano["Ano"].astype(str).tolist()
ret_strat = metr_ano["PnL_Total"].tolist()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Performance por Ano de Trading", fontsize=13, fontweight="bold")

cbs_pnl = [C["verde"] if v > 0 else C["verm"] for v in ret_strat]
axes[0].bar(anos_str, ret_strat, color=cbs_pnl, edgecolor="white")
axes[0].axhline(0, linestyle="-", color=C["cinza"], linewidth=0.8)
for i, v in enumerate(ret_strat):
    axes[0].text(i, v + (0.01 if v >= 0 else -0.03),
                 f"{v:+.2f}", ha="center", fontsize=8, fontweight="bold",
                 color=C["azul"])
axes[0].set_title("PnL Líquido por Ano (log-spread)")
axes[0].set_ylabel("PnL Líquido Total")
plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha="right")

cbs_wr = [C["verde"] if w_ >= 0.55 else C["azul2"] if w_ >= 0.50
          else C["verm"] for w_ in metr_ano["Win_Rate"]]
axes[1].bar(anos_str, metr_ano["Win_Rate"], color=cbs_wr, edgecolor="white")
axes[1].axhline(0.50, linestyle="--", color=C["laran"],
                linewidth=1.5, label="50%")
axes[1].axhline(0.55, linestyle=":",  color=C["verde"],
                linewidth=1.2, label="55%")
for i, v in enumerate(metr_ano["Win_Rate"]):
    axes[1].text(i, v + 0.005, f"{v:.0%}", ha="center", fontsize=8)
axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
axes[1].set_title("Win Rate por Ano"); axes[1].legend()
plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha="right")
plt.tight_layout()
_salvar(fig, "bt_05_performance_ano.png")


# BT-06: diagnóstico de skewness dos z-scores Kalman no trading
if not diag_df.empty:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Diagnóstico de Distribuição — Z-Scores Kalman (período de trading)",
                 fontsize=13, fontweight="bold")
    axes[0].hist(diag_df["z_skew"].dropna(), bins=25,
                 color=C["roxo"], edgecolor="white")
    axes[0].axvline(0, linestyle="--", color=C["cinza"], linewidth=1.5,
                    label="Skew=0 (ideal)")
    axes[0].axvline(diag_df["z_skew"].mean(), linestyle="-",
                    color=C["verm"], linewidth=1.5,
                    label=f"Média={diag_df['z_skew'].mean():.2f}")
    pct_assimetrico = (diag_df["z_skew"].abs() > 0.5).mean()
    axes[0].set_title(f"Skewness  |  {pct_assimetrico:.0%} pares com |skew|>0.5")
    axes[0].legend()
    axes[1].hist(diag_df["z_kurt"].dropna(), bins=25,
                 color=C["laran"], edgecolor="white")
    axes[1].axvline(0, linestyle="--", color=C["cinza"], linewidth=1.5,
                    label="Kurt=0 (Normal)")
    axes[1].axvline(diag_df["z_kurt"].mean(), linestyle="-",
                    color=C["verm"], linewidth=1.5,
                    label=f"Média={diag_df['z_kurt'].mean():.2f}")
    pct_heavy = (diag_df["z_kurt"] > 1.0).mean()
    axes[1].set_title(f"Curtose  |  {pct_heavy:.0%} pares com caudas pesadas")
    axes[1].legend()
    plt.tight_layout()
    _salvar(fig, "bt_06_skewness_zscore.png")


# BT-07: razões de saída
fig, ax = plt.subplots(figsize=(8, 5))
razoes = trades_df["razao_saida"].value_counts()
crs = [C["verde"] if r == "convergencia" else
       C["verm"]  if r == "stop_divergencia" else C["amar"]
       for r in razoes.index]
ax.bar(razoes.index, razoes.values, color=crs, edgecolor="white")
for i, (r, v) in enumerate(razoes.items()):
    ax.text(i, v + 0.5, str(v), ha="center", fontsize=10, fontweight="bold")
ax.set_title("Razões de Saída dos Trades", fontweight="bold")
ax.set_ylabel("N de trades")
plt.tight_layout()
_salvar(fig, "bt_07_razoes_saida.png")


# BT-08: performance por ciclo
if len(metr_ciclo) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Performance por Ciclo Bienal", fontsize=13, fontweight="bold")
    cbs_c = [C["verde"] if s > 0.5 else C["azul2"] if s > 0
             else C["verm"] for s in metr_ciclo["Sharpe"]]
    axes[0].bar(metr_ciclo["ciclo_form"], metr_ciclo["Sharpe"],
                color=cbs_c, edgecolor="white")
    axes[0].axhline(0, linestyle="-",  color=C["cinza"], linewidth=0.8)
    axes[0].axhline(1, linestyle="--", color=C["verde"], linewidth=1.0)
    axes[0].set_title("Sharpe por Ciclo de Formação")
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=30, ha="right")
    axes[1].bar(metr_ciclo["ciclo_form"], metr_ciclo["Win_Rate"],
                color=C["azul2"], edgecolor="white")
    axes[1].axhline(0.5, linestyle="--", color=C["laran"], linewidth=1.5)
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    axes[1].set_title("Win Rate por Ciclo")
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=30, ha="right")
    plt.tight_layout()
    _salvar(fig, "bt_08_performance_ciclo.png")


# ─────────────────────────────────────────────────────────────────────────────
#  RESUMO FINAL
# ─────────────────────────────────────────────────────────────────────────────

def _sinal(v):
    return "+" if v >= 0 else ""

# ── métricas absolutas da estratégia ─────────────────────────────────────
_pnl     = trades_df["pnl_liquido"]
_wins    = _pnl[_pnl > 0]
_losses  = _pnl[_pnl < 0]
_gross_p = _wins.sum()
_gross_l = _losses.abs().sum()
_pf      = _gross_p / _gross_l if _gross_l > 0 else np.inf
_payoff  = _wins.mean() / _losses.abs().mean() if len(_losses) > 0 else np.inf
_n_anos  = (trades_df["data_saida"].max() - trades_df["data_entrada"].min()).days / 365.25
_pnl_ann = _pnl.sum() / _n_anos if _n_anos > 0 else 0
_calmar  = _pnl_ann / abs(max_dd) if max_dd != 0 else np.inf

print(f"\n{SEP}")
print("  RESUMO BACKTEST WALK-FORWARD BIENAL — Kalman Pairs Trading B3")
print(SEP)
print(f"\n  Período     : {CICLOS[0][1]}–{ANO_TRADE_FIM}  "
      f"({len(CICLOS)} ciclos bienais  |  formação 504 pregões)")
print(f"  Kalman δ    : {KALMAN_DELTA:.0e}  |  warm-start: {KALMAN_N_WARM} obs  "
      f"|  Filtro setor: {'sim' if FILTRAR_SETOR else 'não'}")

print(f"\n  {'─'*44}")
print(f"  {'VOLUME':<30}")
print(f"  {'─'*44}")
print(f"  {'Trades totais':<30} {len(trades_df):>10,}")
print(f"  {'Trades/ano (média)':<30} {len(trades_df)/_n_anos:>10.0f}")
print(f"  {'Duração média (pregões)':<30} {trades_df['dias_abertos'].mean():>10.1f}")
print(f"  {'Duração mediana (pregões)':<30} {trades_df['dias_abertos'].median():>10.1f}")

print(f"\n  {'─'*44}")
print(f"  {'RETORNO':<30}")
print(f"  {'─'*44}")
print(f"  {'PnL líquido total':<30} {_pnl.sum():>+10.4f}")
print(f"  {'PnL líquido anualizado':<30} {_pnl_ann:>+10.4f}")
print(f"  {'PnL médio por trade':<30} {_pnl.mean():>+10.4f}")
print(f"  {'Custo total':<30} {(_pnl - trades_df['pnl_bruto']).abs().sum():>10.4f}")
print(f"  {'PnL bruto total':<30} {trades_df['pnl_bruto'].sum():>+10.4f}")

print(f"\n  {'─'*44}")
print(f"  {'RISCO / QUALIDADE':<30}")
print(f"  {'─'*44}")
print(f"  {'Win Rate':<30} {win_rate:>10.1%}")
print(f"  {'Profit Factor':<30} {_pf:>10.2f}")
print(f"  {'Payoff Ratio (win/loss)':<30} {_payoff:>10.2f}")
print(f"  {'Sharpe (anualizado)':<30} {sharpe_total:>10.2f}")
print(f"  {'Calmar Ratio':<30} {_calmar:>10.2f}")
print(f"  {'MaxDD (log-spread)':<30} {max_dd:>10.4f}")
print(f"  {'Ganho médio (wins)':<30} {_wins.mean():>+10.4f}")
print(f"  {'Perda média (losses)':<30} {-_losses.abs().mean():>+10.4f}")

print(f"\n  POR ANO:")
print(f"\n  {'Ano':<6} {'N':>5} {'WinRate':>8} {'PF':>6} {'PnL Total':>10} {'Sharpe':>7} {'MaxDD':>8}")
print(f"  {'─'*54}")
for _, row in metr_ano.iterrows():
    ano = int(row["Ano"])
    g   = trades_df[trades_df["ano_trade"] == ano]["pnl_liquido"]
    gw  = g[g > 0]; gl = g[g < 0]
    pf  = gw.sum() / gl.abs().sum() if len(gl) > 0 and gl.abs().sum() > 0 else np.inf
    pf_str = f"{pf:.2f}" if np.isfinite(pf) else " ∞"
    flag = " ◄" if row["Win_Rate"] >= 0.55 and row["PnL_Total"] > 0 else ""
    print(f"  {ano:<6} {int(row['N_Trades']):>5} {row['Win_Rate']:>8.1%} "
          f"{pf_str:>6} {_sinal(row['PnL_Total'])}{row['PnL_Total']:>9.3f} "
          f"{row['Sharpe']:>7.2f} {row['MaxDD']:>8.4f}{flag}")

print(f"\n  POR CICLO:")
print(f"\n  {'Ciclo':<12} {'N':>5} {'WinRate':>8} {'PF':>6} {'Sharpe':>7} {'MaxDD':>8}")
print(f"  {'─'*50}")
for _, row in metr_ciclo.iterrows():
    g  = trades_df[trades_df["ciclo_form"] == row["ciclo_form"]]["pnl_liquido"]
    gw = g[g > 0]; gl = g[g < 0]
    pf = gw.sum() / gl.abs().sum() if len(gl) > 0 and gl.abs().sum() > 0 else np.inf
    pf_str = f"{pf:.2f}" if np.isfinite(pf) else " ∞"
    flag = " (*)" if row["Sharpe"] > 1 else "    "
    print(f"  {row['ciclo_form']:<12} {int(row['N_Trades']):>5} "
          f"{row['Win_Rate']:>8.1%} {pf_str:>6} "
          f"{row['Sharpe']:>7.2f}{flag} {row['MaxDD']:>8.4f}")

razoes_pct = trades_df["razao_saida"].value_counts(normalize=True)
print(f"\n  RAZÕES DE SAÍDA:")
for razao, pct in razoes_pct.items():
    barra = "█" * int(pct * 30)
    print(f"    {razao:20s} {pct:5.1%}  {barra}")

if not diag_df.empty:
    n_assim = (diag_df["z_skew"].abs() > 0.5).sum()
    n_heavy = (diag_df["z_kurt"] > 1.0).sum()
    print(f"\n  Z-SCORES (out-of-sample):  "
          f"skew={diag_df['z_skew'].mean():.3f}  "
          f"kurt={diag_df['z_kurt'].mean():.3f}  "
          f"|  {n_assim}/{len(diag_df)} assimétricos  "
          f"|  {n_heavy}/{len(diag_df)} caudas pesadas")

print(f"\n  Gráficos : {BT_CHARTS}")
print(f"  CSVs     : {BT_DIR}")
print(f"{SEP}\n")
