"""
Pairs Trading B3 — Modelo Consolidado v4 (Kalman Filter)
=========================================================
Pipeline completo: dados brutos → liquidez → distância mínima →
                   cointegração Engle-Granger → Kalman Filter → z-score dinâmico

Metodologia:
  Gatev, Goetzmann & Rouwenhorst (2006) — pré-filtro por distância mínima,
    janelas mensais de 504 pregões
  Engle & Granger (1987) — cointegração para validação do par
  Avellaneda & Lee (2010) — Kalman Filter para beta dinâmico (substituição do
    OLS estático)

Kalman State Space:
  Observação : log(A_t) = alpha_t + beta_t * log(B_t) + ε_t,  ε ~ N(0, R)
  Transição  : [alpha, beta]_t = [alpha, beta]_{t-1} + δ_t,   δ ~ N(0, Q)
  → beta se adapta ao longo do tempo; z-score = inovação / sqrt(S)
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from statsmodels.tsa.stattools import coint

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURAÇÃO  — edite apenas aqui
# ─────────────────────────────────────────────────────────────────────────────

NOME_CSV         = "dados_economatica_B3.csv"
NOME_CSV_SETORES = "economatica_B3_setores.csv"

ANO_INICIO       = 2013    # carrega dados desde aqui (precisa de 2 anos de história)
ANO_FIM          = 2025
JANELA_FORMACAO  = 504     # pregões por janela ≈ 2 anos (Gatev et al.)
COBERTURA_MIN    = 0.80    # cobertura mínima de pregões por ativo
OBS_MINIMAS_PAR  = int(JANELA_FORMACAO * COBERTURA_MIN)  # = 403 (notebook 04)
MAX_CAND_DIST    = 500     # top N candidatos por distância mínima
TOP_N_PARES      = 20      # top N pares cointegrados por janela
PVALUE_MAX       = 0.05    # p-value máximo Engle-Granger
RAPIDO           = False   # True → maxlag=5 no ADF (mais rápido, menos preciso)

# ── Kalman Filter (Avellaneda & Lee, 2010) ──────────────────────────────────
KALMAN_DELTA     = 1e-5    # velocidade de adaptação de alpha/beta (maior = adapta mais)
KALMAN_VE        = None    # variância R: None = estimada do warm-start OLS, float = fixo
KALMAN_N_WARM    = 50      # obs para warm-start OLS (inicialização do estado)

GERAR_GRAFICOS   = True    # False → pula gráficos (~40% mais rápido)
FILTRAR_SETOR    = True   # True → só pares do mesmo setor econômico


# ─────────────────────────────────────────────────────────────────────────────
#  LOCALIZAÇÃO DOS ARQUIVOS
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
OUTPUT_DIR   = os.path.join(PASTA_SCRIPT, "output")
CHARTS_DIR   = os.path.join(OUTPUT_DIR, "graficos")
os.makedirs(CHARTS_DIR, exist_ok=True)

SEP = "=" * 66
print(f"\n{SEP}\n  PAIRS TRADING B3  —  v4 (Kalman Filter)\n{SEP}")
if CSV_PATH:
    print(f"  [OK] Dados  : {CSV_PATH}")
else:
    print(f"  [ERRO] CSV não encontrado: {NOME_CSV}")
    sys.exit(1)
if SETORES_PATH:
    print(f"  [OK] Setores: {SETORES_PATH}")
else:
    print(f"  [AVISO] Setores não encontrado — pares sem tag setorial")
print(f"  Saída       : {OUTPUT_DIR}")
print(f"  Anos        : {ANO_INICIO}–{ANO_FIM}  |  Janela: {JANELA_FORMACAO} pregões")
print(f"  Dist top-{MAX_CAND_DIST}  |  Coint top-{TOP_N_PARES}/janela  |  p<{PVALUE_MAX}")
print(f"  Kalman δ={KALMAN_DELTA:.0e}  |  warm={KALMAN_N_WARM}  |  "
      f"gráficos: {'sim' if GERAR_GRAFICOS else 'não'}")
print(f"{SEP}\n")


# ─────────────────────────────────────────────────────────────────────────────
#  PALETA VISUAL
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
    """Dict: ticker → {setor, subsetor, segmento}."""
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
                    mapa[ticker] = {
                        "setor":    _v(p[7]),
                        "subsetor": _v(p[8]),
                        "segmento": _v(p[9]),
                    }
    except Exception as exc:
        print(f"  [AVISO] Erro ao ler setores: {exc}")
    return mapa

print("[1/4] Carregando setores...")
SETORES   = _carregar_setores(SETORES_PATH)
n_c_setor = sum(1 for v in SETORES.values() if v.get("setor"))
print(f"      {len(SETORES)} tickers | {n_c_setor} com setor definido\n")


# ─────────────────────────────────────────────────────────────────────────────
#  2. DADOS BRUTOS
# ─────────────────────────────────────────────────────────────────────────────

print(f"[2/4] Carregando {NOME_CSV}...")
t0 = time.time()

_COLS = ["Ativo", "Data", "Fechamento", "Abertura", "Minimo",
         "Maximo", "Medio", "Q_Negs", "Volume_BRL_k", "Q_Titulos_k"]

chunks = []
for _ck in pd.read_csv(CSV_PATH, encoding="latin-1", names=_COLS,
                       header=0, chunksize=200_000, low_memory=False):
    _ck["Ativo"] = (_ck["Ativo"].astype(str)
                    .str.replace("<XBSP>", "", regex=False).str.strip())
    _ck["Data"]  = pd.to_datetime(_ck["Data"], errors="coerce")
    _ck = _ck[_ck["Data"].dt.year.between(ANO_INICIO, ANO_FIM)].copy()
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

print(f"      {len(df_raw):,} linhas | "
      f"{df_raw['Data'].min().date()}–{df_raw['Data'].max().date()} | "
      f"{df_raw['Ativo'].nunique()} ativos | {time.time()-t0:.0f}s\n")


# ─────────────────────────────────────────────────────────────────────────────
#  3. KALMAN FILTER  (Avellaneda & Lee, 2010)
# ─────────────────────────────────────────────────────────────────────────────

def kalman_spread(log_a, log_b,
                  delta=1e-5, ve=None, n_warm=50):
    """
    Kalman Filter para beta dinâmico em pairs trading.

    State space:
        obs:   log_a_t = alpha_t + beta_t * log_b_t + ε_t,  ε ~ N(0, R)
        trans: theta_t = theta_{t-1} + δ_t,                  δ ~ N(0, Q)
        theta = [alpha, beta]

    Warm start: OLS nos primeiros n_warm obs inicializa theta_0 e R.

    Returns
    -------
    alphas : (T,) alpha dinâmico (intercepto)
    betas  : (T,) beta  dinâmico (hedge ratio)
    z_arr  : (T,) z-score = inovação / sqrt(variância da inovação)
    innov  : (T,) inovações brutas (spread no espaço log)
    S_arr  : (T,) variância da inovação (para análise de incerteza)
    """
    T  = len(log_a)
    Q  = delta * np.eye(2)          # ruído do estado (random walk)

    # ── warm start: OLS nos primeiros n_warm obs ─────────────────────────
    n_w = max(2, min(n_warm, T // 4))
    xw, yw = log_b[:n_w], log_a[:n_w]
    sb, sa   = xw.sum(), yw.sum()
    sbb, sab = (xw * xw).sum(), (yw * xw).sum()
    den      = n_w * sbb - sb * sb

    if abs(den) > 1e-12 and n_w >= 5:
        beta0  = (n_w * sab - sb * sa) / den
        alpha0 = (sa - beta0 * sb) / n_w
        resid  = yw - alpha0 - beta0 * xw
        R      = max(resid.var(), 1e-8) if ve is None else max(ve, 1e-8)
    else:
        alpha0, beta0 = 0.0, 1.0
        R = 0.001 if ve is None else ve

    theta = np.array([alpha0, beta0], dtype=np.float64)
    # P_0 pequeno → confiança moderada no warm start
    P     = np.eye(2, dtype=np.float64) * max(R, 1e-4)

    alphas = np.empty(T, dtype=np.float64)
    betas  = np.empty(T, dtype=np.float64)
    z_arr  = np.empty(T, dtype=np.float64)
    innov  = np.empty(T, dtype=np.float64)
    S_arr  = np.empty(T, dtype=np.float64)

    for t in range(T):
        H = np.array([1.0, log_b[t]])       # vetor de observação (1 × 2)

        # predict
        P_pred = P + Q                       # P não tem termo F pois F = I

        # inovação
        nu = log_a[t] - (H @ theta)
        S  = float(H @ P_pred @ H) + R       # variância escalar

        # Kalman gain
        K = (P_pred @ H) / S                 # shape (2,)

        # update
        theta = theta + K * nu
        P     = (np.eye(2) - np.outer(K, H)) @ P_pred

        alphas[t] = theta[0]
        betas[t]  = theta[1]
        innov[t]  = nu
        S_arr[t]  = S
        z_arr[t]  = nu / max(S ** 0.5, 1e-10)

    return alphas, betas, z_arr, innov, S_arr


# ─────────────────────────────────────────────────────────────────────────────
#  4. PIPELINE DE JANELAS MENSAIS
# ─────────────────────────────────────────────────────────────────────────────

print("[3/4] Pipeline mensal (janelas de 504 pregões + Kalman Filter)...")
t_pipe = time.time()

# calendário de pregões
calendario = np.array(sorted(df_raw["Data"].dropna().unique()),
                      dtype="datetime64[ns]")
cal_pd   = pd.DatetimeIndex(calendario)
cal_list = list(cal_pd)
cal_pos  = {d: i for i, d in enumerate(cal_list)}

# datas de rebalanceamento: último pregão de cada mês
datas_fim = (
    pd.Series(cal_pd).to_frame("data").set_index("data")
    .resample("ME").last().dropna().index
)
data_min = pd.Timestamp(calendario[JANELA_FORMACAO - 1])
datas_fim = datas_fim[datas_fim >= data_min]

print(f"      {len(datas_fim)} janelas | "
      f"{datas_fim[0].date()} → {datas_fim[-1].date()}\n")

# acumuladores
resumo_janelas   = []
pares_por_janela = []
zscore_global    = {}   # (a, b) → params + Series

for k, data_fim in enumerate(datas_fim):

    t_j     = time.time()
    pos_fim = cal_pos.get(data_fim, len(cal_list) - 1)
    pos_ini = max(0, pos_fim - JANELA_FORMACAO + 1)
    data_ini = cal_list[pos_ini]

    df_j     = df_raw[(df_raw["Data"] >= data_ini) &
                      (df_raw["Data"] <= data_fim)]
    tot_preg = df_j["Data"].nunique()

    # ── seleção de liquidez (notebook 03) ─────────────────────────────────
    stats = df_j.groupby("Ativo").agg(
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

    tickers = stats["Ativo"].tolist()
    if len(tickers) < 2:
        continue

    # ── matriz de preços ──────────────────────────────────────────────────
    df_p = df_j[df_j["Ativo"].isin(tickers)]

    # precos_raw: ffill sem dropna global → preserva NaN pairwise (distância)
    precos_raw = (df_p.pivot_table(index="Data", columns="Ativo",
                                   values="Fechamento", aggfunc="last")
                      .sort_index().ffill(limit=3))
    precos_raw = precos_raw.dropna(axis=1,
                                   thresh=int(tot_preg * COBERTURA_MIN))

    # precos: alinhada (dropna global) → OLS warm start + Kalman + coint
    precos = precos_raw.dropna()
    cols   = precos.columns.tolist()
    idx    = precos.index
    N      = len(cols)
    if N < 2:
        continue

    mat  = precos.values.astype(np.float64)   # (T, N)
    logP = np.log(np.clip(mat, 1e-10, None))  # (T, N)

    # ── distância mínima (Gatev et al.) — pairwise sem imputação ─────────
    # normalização por-coluna: preço / 1º preço válido (notebook 04)
    precos_dist = precos_raw[cols]
    primeiro_valido = precos_dist.apply(
        lambda c: c.dropna().iloc[0] if c.dropna().shape[0] > 0 else np.nan
    )
    norm_arr = (precos_dist / primeiro_valido).values.astype(np.float64)

    i_idx, j_idx = np.triu_indices(N, k=1)
    dists = np.full(len(i_idx), np.inf)
    for kk, (ii, jj) in enumerate(zip(i_idx, j_idx)):
        col_a = norm_arr[:, ii]
        col_b = norm_arr[:, jj]
        mask  = ~(np.isnan(col_a) | np.isnan(col_b))
        if mask.sum() < OBS_MINIMAS_PAR:        # check pairwise (notebook 04)
            continue
        dists[kk] = np.sum((col_a[mask] - col_b[mask]) ** 2)

    validos = np.where(np.isfinite(dists))[0]
    n_cand  = min(MAX_CAND_DIST, len(validos))
    if n_cand == 0:
        continue
    top_rel  = np.argpartition(dists[validos], n_cand - 1)[:n_cand]
    top_rel  = top_rel[np.argsort(dists[validos][top_rel])]
    top_pos  = validos[top_rel]
    candidatos = [(cols[i_idx[p]], cols[j_idx[p]], float(dists[p]))
                  for p in top_pos]

    if FILTRAR_SETOR:
        candidatos = [
            (a, b, d) for a, b, d in candidatos
            if SETORES.get(a, {}).get("setor") is not None
            and SETORES.get(a, {}).get("setor") ==
                SETORES.get(b, {}).get("setor")
        ]

    # ── cointegração Engle-Granger + Kalman Filter ────────────────────────
    col_idx  = {c: i for i, c in enumerate(cols)}
    pares_ok = []

    for a, b, dist in candidatos:
        ia, ib = col_idx[a], col_idx[b]
        ya, xb = logP[:, ia], logP[:, ib]

        # validação do par: Engle-Granger na janela de formação
        try:
            if RAPIDO:
                _, pval, _ = coint(ya, xb, maxlag=5, autolag=None)
            else:
                _, pval, _ = coint(ya, xb)
        except Exception:
            continue
        if pval > PVALUE_MAX:
            continue

        # ── Kalman Filter — beta dinâmico ─────────────────────────────────
        alphas, betas, z_arr, innov, S_arr = kalman_spread(
            ya, xb,
            delta=KALMAN_DELTA,
            ve=KALMAN_VE,
            n_warm=KALMAN_N_WARM,
        )

        # métricas do beta dinâmico
        beta_final  = float(betas[-1])
        beta_mean   = float(betas.mean())
        beta_std    = float(betas.std())
        alpha_final = float(alphas[-1])
        z_final     = float(z_arr[-1])

        # spread bruto = inovações acumuladas ao longo da janela
        spread_series = pd.Series(innov, index=idx, name=f"{a}__{b}")
        z_series      = pd.Series(z_arr, index=idx,  name=f"{a}__{b}")
        beta_series   = pd.Series(betas, index=idx,  name=f"{a}__{b}")

        inf_a = SETORES.get(a, {})
        inf_b = SETORES.get(b, {})
        set_a = inf_a.get("setor")
        set_b = inf_b.get("setor")
        msm   = (set_a is not None and set_a == set_b)
        corr  = float(np.corrcoef(ya, xb)[0, 1])

        pares_ok.append({
            "a": a, "b": b, "dist": dist, "pval": pval, "corr": corr,
            "alpha_final": alpha_final, "beta_final": beta_final,
            "beta_mean": beta_mean, "beta_std": beta_std,
            "z_final": z_final,
            "z_series": z_series, "spread_series": spread_series,
            "beta_series": beta_series,
            "set_a": set_a, "set_b": set_b, "msm": msm,
            "sub_a": inf_a.get("subsetor"), "sub_b": inf_b.get("subsetor"),
        })

    # ordena por p-value + distância como tie-breaker (notebook 04)
    pares_ok.sort(key=lambda x: (x["pval"], x["dist"]))
    pares_ok = pares_ok[:TOP_N_PARES]

    # ── salva resultados da janela ─────────────────────────────────────────
    resumo_janelas.append({
        "data_fim_janela":   data_fim,
        "N_Ativos":          N,
        "Candidatos_Dist":   len(candidatos),
        "Pares_Cointegrados": len(pares_ok),
        "Pregoes_Janela":    tot_preg,
    })

    for p in pares_ok:
        a, b = p["a"], p["b"]
        pares_por_janela.append({
            "data_fim_janela": data_fim,
            "Ativo_A":        a,           "Ativo_B":        b,
            "Distancia":      round(p["dist"],       6),
            "Correlacao":     round(p["corr"],       4),
            "P_Value":        round(p["pval"],       6),
            "Alpha_Kal":      round(p["alpha_final"],6),
            "Beta_Kal_Final": round(p["beta_final"], 6),
            "Beta_Kal_Media": round(p["beta_mean"],  6),
            "Beta_Kal_Std":   round(p["beta_std"],   6),
            "Z_Score_Atual":  round(p["z_final"],    4),
            "Setor_A":        p["set_a"], "Setor_B":        p["set_b"],
            "Subsetor_A":     p["sub_a"], "Subsetor_B":     p["sub_b"],
            "Mesmo_Setor":    p["msm"],
        })

        key = (a, b) if a < b else (b, a)
        zscore_global[key] = {
            "data_fim_janela": data_fim,
            "Alpha_Kal":      p["alpha_final"],
            "Beta_Kal_Final": p["beta_final"],
            "Beta_Kal_Media": p["beta_mean"],
            "Beta_Kal_Std":   p["beta_std"],
            "Correlacao":     p["corr"],
            "P_Value":        p["pval"],
            "Distancia":      p["dist"],
            "Setor_A":        p["set_a"], "Setor_B": p["set_b"],
            "Mesmo_Setor":    p["msm"],
            "Z":              p["z_series"],      # Kalman z-score
            "S":              p["spread_series"], # inovações brutas
            "Beta_t":         p["beta_series"],   # beta dinâmico
        }

    if (k + 1) % 12 == 0 or k == 0 or k == len(datas_fim) - 1:
        print(f"      {data_fim.strftime('%Y-%m')} | "
              f"{N:3d} ativos | {len(candidatos):4d} cand | "
              f"{len(pares_ok):2d} pares | {time.time()-t_j:.1f}s")

print(f"\n      {len(pares_por_janela)} pares-janela | "
      f"{len(zscore_global)} pares únicos | "
      f"{time.time()-t_pipe:.0f}s total pipeline\n")


# ─────────────────────────────────────────────────────────────────────────────
#  5. CONSOLIDAÇÃO E CSVs
# ─────────────────────────────────────────────────────────────────────────────

n_steps = 5 if GERAR_GRAFICOS else 4
print(f"[4/{n_steps}] Consolidando e salvando CSVs...")

resumo_df = pd.DataFrame(resumo_janelas)
pares_df  = pd.DataFrame(pares_por_janela)

freq_pares = (pares_df.groupby(["Ativo_A", "Ativo_B"])
              .agg(N_Janelas      =("data_fim_janela", "count"),
                   P_Value_Min    =("P_Value",         "min"),
                   P_Value_Med    =("P_Value",         "mean"),
                   Dist_Med       =("Distancia",       "mean"),
                   Corr_Med       =("Correlacao",      lambda x: x.abs().mean()),
                   Beta_Kal_Medio =("Beta_Kal_Media",  "mean"),
                   Beta_Kal_Std   =("Beta_Kal_Std",    "mean"),
                   Mesmo_Setor    =("Mesmo_Setor",     "first"),
                   Setor_A        =("Setor_A",         "first"),
                   Setor_B        =("Setor_B",         "first"))
              .reset_index()
              .sort_values(["N_Janelas", "P_Value_Min"],
                           ascending=[False, True]))

melhores = (pares_df.sort_values("P_Value")
            .drop_duplicates(subset=["Ativo_A", "Ativo_B"])
            .reset_index(drop=True))

ativ_a = pares_df[["data_fim_janela","Ativo_A"]].rename(
             columns={"Ativo_A": "Ativo"})
ativ_b = pares_df[["data_fim_janela","Ativo_B"]].rename(
             columns={"Ativo_B": "Ativo"})
persistencia = (pd.concat([ativ_a, ativ_b])
                .drop_duplicates()
                .groupby("Ativo")["data_fim_janela"]
                .count().reset_index(name="N_Janelas")
                .sort_values("N_Janelas", ascending=False))

# z-score Kalman wide (último período de formação de cada par)
z_wide = (pd.DataFrame({f"{a}__{b}": v["Z"]
                         for (a, b), v in zscore_global.items()})
            .sort_index())

for nome, df_out in {
    "resumo_janelas.csv":      resumo_df,
    "pares_por_janela.csv":    pares_df,
    "pares_frequentes.csv":    freq_pares,
    "melhores_pares.csv":      melhores,
    "persistencia_ativos.csv": persistencia,
    "zscores_kalman.csv":      z_wide,
}.items():
    df_out.to_csv(os.path.join(OUTPUT_DIR, nome),
                  index=(nome == "zscores_kalman.csv"))

print(f"      {len(pares_df)} pares-janela | "
      f"{len(freq_pares)} únicos | "
      f"{len(zscore_global)} séries z-score Kalman")
print(f"      CSVs em: output/")

if not GERAR_GRAFICOS:
    print("\n  [info] GERAR_GRAFICOS=False — pulando gráficos.")
else:

    # ─────────────────────────────────────────────────────────────────────────
    #  6. GRÁFICOS
    # ─────────────────────────────────────────────────────────────────────────

    print(f"\n[5/{n_steps}] Gerando gráficos...")

    def _salvar(fig, nome):
        fig.savefig(os.path.join(CHARTS_DIR, nome),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"      {nome}")

    pares_z = sorted(zscore_global.items(), key=lambda x: x[1]["P_Value"])
    pares_df["Ano"] = pd.DatetimeIndex(pares_df["data_fim_janela"]).year
    resumo_df["Ano"] = pd.DatetimeIndex(resumo_df["data_fim_janela"]).year

    res_ano = resumo_df.groupby("Ano").agg(
        Candidatos_Dist    =("Candidatos_Dist",    "mean"),
        Pares_Cointegrados =("Pares_Cointegrados", "sum"),
        N_Ativos           =("N_Ativos",           "mean"),
    ).reset_index()

    # G1 — overview anual
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle(f"Overview Anual — Janelas de {JANELA_FORMACAO} Pregões + Kalman",
                 fontsize=13, fontweight="bold")
    axes[0].bar(res_ano["Ano"].astype(str), res_ano["N_Ativos"],
                color=C["azul2"], edgecolor="white")
    axes[0].set_title("Média Ativos Líquidos/Janela")
    axes[0].set_ylabel("N ativos")
    axes[1].bar(res_ano["Ano"].astype(str), res_ano["Candidatos_Dist"],
                color=C["azul2"], edgecolor="white", alpha=0.4,
                label=f"Cand. dist. (top-{MAX_CAND_DIST})")
    axes[1].bar(res_ano["Ano"].astype(str), res_ano["Pares_Cointegrados"],
                color=C["verde"], edgecolor="white",
                label=f"Cointegrados (p<{PVALUE_MAX})")
    axes[1].set_title("Candidatos vs Pares Cointegrados")
    axes[1].set_ylabel("N pares"); axes[1].legend()
    msm_pct = (pares_df.groupby("Ano")["Mesmo_Setor"].mean() * 100
               ).reindex(res_ano["Ano"])
    axes[2].bar(msm_pct.index.astype(str), msm_pct.values,
                color=C["roxo"], edgecolor="white", alpha=0.85)
    axes[2].axhline(msm_pct.mean(), linestyle="--", color=C["laran"],
                    linewidth=1.5, label=f"Média: {msm_pct.mean():.0f}%")
    axes[2].set_title("% Pares Mesmo Setor")
    axes[2].set_ylabel("%"); axes[2].set_ylim(0, 100); axes[2].legend()
    plt.tight_layout()
    _salvar(fig, "01_overview_anual.png")

    # G2 — persistência dos ativos
    top30 = persistencia.head(30)
    n_tot = resumo_df["data_fim_janela"].nunique()
    cbs   = [C["verde"] if j >= n_tot * 0.8 else C["azul2"] if j >= n_tot * 0.5
             else C["amar"] if j >= n_tot * 0.3 else C["cinza"]
             for j in top30["N_Janelas"]]
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(top30["Ativo"][::-1], top30["N_Janelas"][::-1],
            color=list(reversed(cbs)), edgecolor="white", linewidth=0.6)
    ax.axvline(n_tot, linestyle="--", color=C["verm"], linewidth=1.3,
               label=f"Total: {n_tot} janelas")
    ax.set_xlabel("Janelas mensais com presença em algum par")
    ax.set_title("Persistência dos Ativos nos Pares", fontweight="bold")
    for v, yp in zip(top30["N_Janelas"][::-1], range(len(top30))):
        ax.text(v + 0.3, yp, str(v), va="center", fontsize=8.5)
    patches = [mpatches.Patch(color=C["verde"], label="≥80% janelas"),
               mpatches.Patch(color=C["azul2"], label="≥50%"),
               mpatches.Patch(color=C["amar"],  label="≥30%"),
               mpatches.Patch(color=C["cinza"], label="<30%")]
    ax.legend(handles=patches, loc="lower right")
    plt.tight_layout()
    _salvar(fig, "02_persistencia_ativos.png")

    # G3 — pares por janela mensal
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    fig.suptitle("Pares Cointegrados por Janela Mensal",
                 fontsize=13, fontweight="bold")
    axes[0].fill_between(resumo_df["data_fim_janela"],
                         resumo_df["Pares_Cointegrados"],
                         alpha=0.4, color=C["verde"])
    axes[0].plot(resumo_df["data_fim_janela"],
                 resumo_df["Pares_Cointegrados"],
                 color=C["verde"], linewidth=1.2)
    axes[0].axhline(resumo_df["Pares_Cointegrados"].mean(), linestyle="--",
                    color=C["laran"], linewidth=1.5,
                    label=f"Média: {resumo_df['Pares_Cointegrados'].mean():.1f}")
    axes[0].set_ylabel("Pares cointegrados"); axes[0].legend()
    axes[1].plot(resumo_df["data_fim_janela"], resumo_df["N_Ativos"],
                 color=C["azul2"], linewidth=1.2)
    axes[1].fill_between(resumo_df["data_fim_janela"],
                         resumo_df["N_Ativos"], alpha=0.3, color=C["azul2"])
    axes[1].set_ylabel("Ativos líquidos")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    plt.tight_layout()
    _salvar(fig, "03_pares_por_janela.png")

    # G4 — ranking de pares
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Ranking dos Pares — Frequência e Força",
                 fontweight="bold", fontsize=13)
    top_freq = freq_pares.head(20).copy()
    top_freq["Par"] = top_freq["Ativo_A"] + " × " + top_freq["Ativo_B"]
    max_jan = freq_pares["N_Janelas"].max()
    cbs_f   = [C["verde"] if n >= max_jan * 0.8 else C["azul2"] if n >= max_jan * 0.5
               else C["amar"] for n in top_freq["N_Janelas"]]
    axes[0].barh(top_freq["Par"][::-1], top_freq["N_Janelas"][::-1],
                 color=list(reversed(cbs_f)), edgecolor="white")
    axes[0].set_xlabel("Janelas cointegrado")
    axes[0].set_title("Top 20 — Mais Frequentes")
    for v, yp in zip(top_freq["N_Janelas"][::-1], range(len(top_freq))):
        axes[0].text(v + 0.1, yp, str(v), va="center", fontsize=8.5)
    top_forte = melhores.head(20).copy()
    top_forte["Par"] = top_forte["Ativo_A"] + " × " + top_forte["Ativo_B"]
    cbs_s = [C["verde"] if pv <= 0.01 else C["azul2"] if pv <= 0.03
             else C["amar"] for pv in top_forte["P_Value"]]
    axes[1].barh(top_forte["Par"][::-1],
                 top_forte["Correlacao"].abs()[::-1],
                 color=list(reversed(cbs_s)), edgecolor="white")
    axes[1].set_xlim(0.75, 1.02)
    axes[1].set_xlabel("|Correlação|")
    axes[1].set_title("Top 20 — Menor P-value")
    for i, (_, row) in enumerate(top_forte[::-1].iterrows()):
        axes[1].text(abs(row["Correlacao"]) + 0.002, i,
                     f"p={row['P_Value']:.4f}", va="center",
                     fontsize=7.5, color=C["azul"])
    plt.tight_layout()
    _salvar(fig, "04_pares_ranking.png")

    # G5 — beta dinâmico (Kalman) — TOP 4 PARES
    top4 = pares_z[:4]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        f"Beta Dinâmico (Kalman Filter) — Top 4 Pares\n"
        f"δ={KALMAN_DELTA:.0e}  |  warm-start={KALMAN_N_WARM} obs",
        fontsize=13, fontweight="bold")
    for i, ((at, bt), info) in enumerate(top4):
        ax   = axes.flatten()[i]
        beta = info["Beta_t"].dropna()
        ax.plot(beta.index, beta.values,
                color=C["azul2"], linewidth=1.0, label="β(t)")
        bm = beta.mean(); bs = beta.std()
        ax.axhline(bm, linestyle="--", color=C["cinza"],
                   linewidth=1.2, label=f"Média={bm:.3f}")
        ax.axhline(bm + bs, linestyle=":", color=C["laran"],
                   linewidth=0.9, label=f"±1σ={bs:.3f}")
        ax.axhline(bm - bs, linestyle=":", color=C["laran"], linewidth=0.9)
        ax.fill_between(beta.index, bm - bs, bm + bs,
                        alpha=0.10, color=C["laran"])
        setor_tag = f" [{info['Setor_A']}]" if info.get("Setor_A") else ""
        ax.set_title(
            f"{at} × {bt}{setor_tag}\n"
            f"p={info['P_Value']:.4f}  dist={info['Distancia']:.3f}  "
            f"β_final={info['Beta_Kal_Final']:.3f}  "
            f"({pd.Timestamp(info['data_fim_janela']).strftime('%Y-%m')})",
            fontsize=9, fontweight="bold")
        ax.set_ylabel("β dinâmico (hedge ratio)")
        ax.legend(fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.tight_layout()
    _salvar(fig, "05_beta_dinamico_kalman.png")

    # G6 — z-score Kalman top 6 pares
    top6 = pares_z[:6]
    fig, axes = plt.subplots(3, 2, figsize=(16, 13))
    fig.suptitle("Z-Score Kalman — Top 6 Pares (menor p-value)\n"
                 "Vermelho: SHORT (+2σ) | Verde: LONG (-2σ)",
                 fontsize=13, fontweight="bold")
    for i, ((at, bt), info) in enumerate(top6):
        ax  = axes.flatten()[i]
        z   = info["Z"].dropna()
        nl  = (z <= -2).sum(); ns = (z >= 2).sum()
        pct = (nl + ns) / len(z) * 100
        ax.plot(z.index, z.values, color=C["azul2"],
                linewidth=0.85, alpha=0.9)
        for lvl, col, ls in [(2, C["verm"], "--"), (-2, C["verde"], "--"),
                              (3, C["verm"], ":"),  (-3, C["verde"], ":"),
                              (0, C["cinza"], "-")]:
            ax.axhline(lvl, linestyle=ls, color=col,
                       linewidth=1.2 if abs(lvl) == 2 else 0.8, alpha=0.85)
        ax.fill_between(z.index, 2,  z.values,
                        where=(z.values >= 2),  alpha=0.2, color=C["verm"])
        ax.fill_between(z.index, -2, z.values,
                        where=(z.values <= -2), alpha=0.2, color=C["verde"])
        setor_tag = f" [{info['Setor_A']}]" if info.get("Setor_A") else ""
        ax.set_title(
            f"{at} × {bt}{setor_tag}\n"
            f"|r|={abs(info['Correlacao']):.3f}  "
            f"p={info['P_Value']:.4f}  "
            f"β_final={info['Beta_Kal_Final']:.3f}  "
            f"({pd.Timestamp(info['data_fim_janela']).strftime('%Y-%m')})\n"
            f"LONG:{nl}d  SHORT:{ns}d  {pct:.1f}% ativo",
            fontsize=9, fontweight="bold")
        ax.set_ylabel("Z-Score (Kalman)"); ax.set_ylim(-5, 5)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.text(z.index[-1], 2.1,  "+2σ", color=C["verm"],  fontsize=8)
        ax.text(z.index[-1], -2.4, "-2σ", color=C["verde"], fontsize=8)
    plt.tight_layout()
    _salvar(fig, "06_zscore_kalman_top6.png")

    # G7 — spread bruto (inovações Kalman) top 4
    top4 = pares_z[:4]
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    fig.suptitle("Spread Kalman (Inovações) — Top 4 Pares",
                 fontsize=13, fontweight="bold")
    for i, ((at, bt), info) in enumerate(top4):
        ax = axes.flatten()[i]
        s  = info["S"].dropna()
        mu = s.mean(); sig = s.std()
        ax.plot(s.index, s.values, color=C["azul2"],
                linewidth=0.8, alpha=0.9)
        ax.axhline(mu, linestyle="-", color=C["cinza"],
                   linewidth=1.2, label=f"μ={mu:.4f}")
        ax.axhline(mu + 2 * sig, linestyle="--", color=C["verm"],
                   linewidth=1.2, label="+2σ")
        ax.axhline(mu - 2 * sig, linestyle="--", color=C["verde"],
                   linewidth=1.2, label="-2σ")
        ax.fill_between(s.index, mu + 2 * sig, s.values,
                        where=(s.values >= mu + 2 * sig),
                        alpha=0.2, color=C["verm"])
        ax.fill_between(s.index, mu - 2 * sig, s.values,
                        where=(s.values <= mu - 2 * sig),
                        alpha=0.2, color=C["verde"])
        ax.set_title(f"Spread: {at} × {bt}  "
                     f"(β_med={info['Beta_Kal_Media']:.3f}  "
                     f"β_std={info['Beta_Kal_Std']:.3f})",
                     fontweight="bold", fontsize=10)
        ax.set_ylabel("Inovação Kalman (log)"); ax.legend(fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.tight_layout()
    _salvar(fig, "07_spread_kalman_top4.png")

    # G8 — par destaque (melhor p-value)
    (ad, bd), id_ = pares_z[0]
    zd = id_["Z"].dropna()
    sd = id_["S"].dropna()
    bd_t = id_["Beta_t"].dropna()

    fig = plt.figure(figsize=(15, 12))
    fig.suptitle(
        f"Par Destaque: {ad} × {bd}\n"
        f"|r|={abs(id_['Correlacao']):.4f}  "
        f"p={id_['P_Value']:.4f}  "
        f"β_final={id_['Beta_Kal_Final']:.3f}  "
        f"δ={KALMAN_DELTA:.0e}  "
        f"janela: {pd.Timestamp(id_['data_fim_janela']).strftime('%Y-%m')}",
        fontsize=12, fontweight="bold")
    gs = GridSpec(4, 1, hspace=0.5, figure=fig)

    ax1 = fig.add_subplot(gs[0])
    ax1.plot(sd.index, sd.values, color=C["azul2"], linewidth=0.9)
    mu_d = sd.mean(); sg_d = sd.std()
    ax1.axhline(mu_d, linestyle="--", color=C["cinza"], linewidth=1)
    ax1.fill_between(sd.index, mu_d + 2 * sg_d, sd.values,
                     where=(sd.values >= mu_d + 2 * sg_d),
                     alpha=0.2, color=C["verm"])
    ax1.fill_between(sd.index, mu_d - 2 * sg_d, sd.values,
                     where=(sd.values <= mu_d - 2 * sg_d),
                     alpha=0.2, color=C["verde"])
    ax1.set_ylabel("Spread (inovação)")
    ax1.set_title("Inovações Kalman (spread dinâmico)")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.xaxis.set_major_locator(mdates.YearLocator())

    ax2 = fig.add_subplot(gs[1])
    ax2.plot(bd_t.index, bd_t.values, color=C["roxo"], linewidth=1.0)
    bm = bd_t.mean(); bs = bd_t.std()
    ax2.axhline(bm, linestyle="--", color=C["cinza"], linewidth=1,
                label=f"Média β={bm:.3f}")
    ax2.fill_between(bd_t.index, bm - bs, bm + bs,
                     alpha=0.15, color=C["roxo"])
    ax2.set_ylabel("β dinâmico")
    ax2.set_title(f"Hedge Ratio Dinâmico (Kalman)  ±1σ={bs:.3f}")
    ax2.legend(fontsize=9)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.xaxis.set_major_locator(mdates.YearLocator())

    ax3 = fig.add_subplot(gs[2])
    ax3.plot(zd.index, zd.values, color=C["azul"], linewidth=0.9, alpha=0.9)
    for lvl, col, ls in [(2, C["verm"], "--"), (-2, C["verde"], "--"),
                          (3, C["verm"], ":"),  (-3, C["verde"], ":"),
                          (0, C["cinza"], "-")]:
        ax3.axhline(lvl, linestyle=ls, color=col,
                    linewidth=1.2 if abs(lvl) <= 2 else 0.9)
    ax3.fill_between(zd.index, 2,  zd.values,
                     where=(zd.values >= 2),  alpha=0.2, color=C["verm"])
    ax3.fill_between(zd.index, -2, zd.values,
                     where=(zd.values <= -2), alpha=0.2, color=C["verde"])
    nl = (zd <= -2).sum(); ns = (zd >= 2).sum()
    ax3.set_ylabel("Z-Score"); ax3.set_ylim(-5, 5)
    ax3.set_title(f"Z-Score Kalman — LONG:{nl}d  SHORT:{ns}d")
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax3.xaxis.set_major_locator(mdates.YearLocator())

    ax4 = fig.add_subplot(gs[3])
    zr  = zd[zd.index >= pd.Timestamp("2024-01-01")]
    ax4.plot(zr.index, zr.values, color=C["azul"], linewidth=1.2)
    for lvl, col, ls in [(2, C["verm"], "--"), (-2, C["verde"], "--"),
                          (0, C["cinza"], "-")]:
        ax4.axhline(lvl, linestyle=ls, color=col, linewidth=1.2)
    ax4.fill_between(zr.index, 2,  zr.values,
                     where=(zr.values >= 2),  alpha=0.25, color=C["verm"])
    ax4.fill_between(zr.index, -2, zr.values,
                     where=(zr.values <= -2), alpha=0.25, color=C["verde"])
    ax4.set_ylabel("Z-Score"); ax4.set_ylim(-4, 4)
    ax4.set_title(
        f"Detalhe 2024–atual  |  Z atual={zd.iloc[-1]:+.2f}  "
        f"β atual={bd_t.iloc[-1]:.3f}  ({zd.index[-1].date()})")
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%b/%Y"))
    ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

    plt.savefig(os.path.join(CHARTS_DIR, "08_par_destaque_kalman.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("      08_par_destaque_kalman.png")

    # G9 — sinais ativos ao longo do tempo
    if not z_wide.empty:
        sl   = (z_wide <= -2).sum(axis=1)
        ss   = (z_wide >= 2).sum(axis=1)
        sm_l = sl.resample("ME").mean()
        sm_s = ss.resample("ME").mean()
        sm_t = (sl + ss).resample("ME").mean()
        fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
        fig.suptitle("Sinais Ativos Kalman (|Z| ≥ 2) ao Longo do Tempo",
                     fontsize=13, fontweight="bold")
        axes[0].fill_between(sm_t.index, sm_t.values,
                             alpha=0.5, color=C["laran"])
        axes[0].plot(sm_t.index, sm_t.values,
                     color=C["laran"], linewidth=1.5)
        axes[0].axhline(sm_t.mean(), linestyle="--", color=C["azul"],
                        linewidth=1.2,
                        label=f"Média: {sm_t.mean():.1f}/mês")
        axes[0].set_ylabel("N pares"); axes[0].legend()
        axes[0].set_title("Total Sinais Ativos (Kalman Z-Score)")
        axes[1].fill_between(sm_l.index, sm_l.values,
                             alpha=0.5, color=C["verde"], label="LONG (Z≤-2)")
        axes[1].fill_between(sm_s.index, sm_s.values,
                             alpha=0.5, color=C["verm"],  label="SHORT (Z≥+2)")
        axes[1].plot(sm_l.index, sm_l.values, color=C["verde"], linewidth=1.2)
        axes[1].plot(sm_s.index, sm_s.values, color=C["verm"],  linewidth=1.2)
        axes[1].set_ylabel("N pares"); axes[1].legend()
        axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        axes[1].xaxis.set_major_locator(mdates.YearLocator())
        plt.tight_layout()
        _salvar(fig, "09_sinais_temporais.png")

    # G10 — snapshot atual
    if not z_wide.empty:
        ultimo_z = z_wide.iloc[-1].dropna().sort_values()
        nl_now   = (ultimo_z <= -2).sum()
        ns_now   = (ultimo_z >= 2).sum()
        data_ref = z_wide.index[-1].strftime("%d/%m/%Y")
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f"Snapshot Kalman Z-Scores — {data_ref}",
                     fontsize=13, fontweight="bold")
        cbs_snap = [C["verm"] if z >= 2 else C["verde"] if z <= -2
                    else C["cinza"] for z in ultimo_z.values]
        axes[0].bar(range(len(ultimo_z)), ultimo_z.values,
                    color=cbs_snap, edgecolor="none", width=1.0)
        axes[0].axhline(2,  linestyle="--", color=C["verm"],
                        linewidth=1.2, label="+2σ (SHORT)")
        axes[0].axhline(-2, linestyle="--", color=C["verde"],
                        linewidth=1.2, label="-2σ (LONG)")
        axes[0].axhline(0,  linestyle="-",  color=C["cinza"], linewidth=0.7)
        axes[0].set_title(
            f"{len(ultimo_z)} pares | SHORT:{ns_now}  LONG:{nl_now}  "
            f"Neutro:{len(ultimo_z)-nl_now-ns_now}")
        axes[0].set_xlabel("Pares (ordenados por z-score Kalman)")
        axes[0].set_ylabel("Z-Score"); axes[0].legend()
        axes[1].hist(ultimo_z.values, bins=20,
                     color=C["azul2"], edgecolor="white")
        axes[1].axvline(2,  linestyle="--", color=C["verm"],
                        linewidth=1.5, label="+2σ")
        axes[1].axvline(-2, linestyle="--", color=C["verde"],
                        linewidth=1.5, label="-2σ")
        axes[1].axvline(0,  linestyle="-",  color=C["cinza"], linewidth=0.8)
        axes[1].set_title("Distribuição Z-Scores Kalman Atuais")
        axes[1].legend()
        plt.tight_layout()
        _salvar(fig, "10_snapshot_zscores.png")

    # G11 — heatmap pares × ano
    if not pares_df.empty:
        pares_df["Par"] = pares_df["Ativo_A"] + "_" + pares_df["Ativo_B"]
        top_hm = freq_pares.head(25).copy()
        top_hm["Par"] = top_hm["Ativo_A"] + "_" + top_hm["Ativo_B"]
        hm_data  = pares_df[pares_df["Par"].isin(top_hm["Par"].values)]
        pivot_hm = hm_data.pivot_table(index="Par", columns="Ano",
                                       values="P_Value", aggfunc="min")
        pivot_hm = pivot_hm.reindex(
            [p for p in top_hm["Par"].tolist() if p in pivot_hm.index])
        pivot_hm.index = [p.replace("_", " × ") for p in pivot_hm.index]
        fig, ax = plt.subplots(figsize=(14, max(6, len(pivot_hm) * 0.4)))
        im = ax.imshow(pivot_hm.values, aspect="auto", cmap="RdYlGn_r",
                       vmin=0, vmax=PVALUE_MAX,
                       extent=[-0.5, pivot_hm.shape[1] - 0.5,
                                pivot_hm.shape[0] - 0.5, -0.5])
        plt.colorbar(im, ax=ax, label="P-value mín. do ano")
        ax.set_xticks(range(len(pivot_hm.columns)))
        ax.set_xticklabels(pivot_hm.columns.astype(str), rotation=45)
        ax.set_yticks(range(len(pivot_hm.index)))
        ax.set_yticklabels(pivot_hm.index, fontsize=9)
        ax.set_title(f"Heatmap — Top {len(pivot_hm)} Pares Mais Frequentes",
                     fontweight="bold")
        ax.set_xlabel("Ano")
        for i in range(pivot_hm.shape[0]):
            for j in range(pivot_hm.shape[1]):
                val = pivot_hm.iloc[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                            fontsize=6.5,
                            color="white" if val < 0.02 else "black")
        plt.tight_layout()
        _salvar(fig, "11_heatmap_pares.png")

    # G12 — setores
    if "Setor_A" in pares_df.columns and pares_df["Setor_A"].notna().any():
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle("Distribuição por Setor Econômico",
                     fontsize=13, fontweight="bold")
        msm_df = pares_df[pares_df["Mesmo_Setor"] == True]
        if not msm_df.empty and msm_df["Setor_A"].notna().any():
            cnt = msm_df["Setor_A"].value_counts().head(10)
            axes[0].pie(cnt.values, labels=cnt.index,
                        autopct="%1.0f%%", startangle=90,
                        colors=plt.cm.Set3.colors[:len(cnt)])
            axes[0].set_title(
                f"Pares Mesmo Setor ({msm_df.shape[0]} pares-janela)")
        todos_set = pd.concat([
            pares_df[["Setor_A","Ano"]].rename(columns={"Setor_A":"Setor"}),
            pares_df[["Setor_B","Ano"]].rename(columns={"Setor_B":"Setor"}),
        ]).dropna(subset=["Setor"])
        todos_set = todos_set[todos_set["Setor"] != "Outros"]
        pivot_set = todos_set.groupby(["Ano","Setor"]).size().unstack(fill_value=0)
        pivot_set.plot(kind="bar", ax=axes[1], colormap="Set3",
                       edgecolor="white", linewidth=0.5)
        axes[1].set_title("Participação Setorial × Ano")
        axes[1].set_xlabel("Ano"); axes[1].set_ylabel("N ativos em pares")
        axes[1].legend(title="Setor", fontsize=7, ncol=2)
        plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha="right")
        plt.tight_layout()
        _salvar(fig, "12_setores.png")


# ─────────────────────────────────────────────────────────────────────────────
#  RESUMO FINAL
# ─────────────────────────────────────────────────────────────────────────────

if not z_wide.empty:
    ult_z = z_wide.iloc[-1].dropna().sort_values()
    nl_s  = (ult_z <= -2).sum()
    ns_s  = (ult_z >= 2).sum()
else:
    ult_z = pd.Series(dtype=float)
    nl_s = ns_s = 0

print(f"\n{SEP}")
print("RESUMO FINAL — Kalman Filter")
print(SEP)
print(f"\n  Período   : {ANO_INICIO}–{ANO_FIM}")
print(f"  Janelas   : {len(resumo_df)} mensais de {JANELA_FORMACAO} pregões")
print(f"  Kalman    : δ={KALMAN_DELTA:.0e}  warm={KALMAN_N_WARM}  "
      f"RAPIDO={'sim' if RAPIDO else 'não'}")
print(f"\n  Totais    : {len(pares_df)} pares-janela | "
      f"{len(freq_pares)} únicos | "
      f"{len(zscore_global)} séries z-score Kalman")

print(f"\n  PARES MAIS FREQUENTES (top 10):")
print("  " + freq_pares.head(10)[
    ["Ativo_A","Ativo_B","N_Janelas","P_Value_Min",
     "Beta_Kal_Medio","Beta_Kal_Std","Mesmo_Setor","Setor_A"]]
    .to_string(index=False).replace("\n", "\n  "))

print(f"\n  PARES MAIS FORTES (menor p-value | top 10):")
print("  " + melhores.head(10)[
    ["Ativo_A","Ativo_B","P_Value","Correlacao",
     "Beta_Kal_Final","Beta_Kal_Std","Setor_A","Setor_B"]]
    .to_string(index=False).replace("\n", "\n  "))

if len(ult_z) > 0:
    data_ult = z_wide.index[-1].date()
    print(f"\n  SNAPSHOT KALMAN ({data_ult}) — "
          f"LONG:{nl_s} | SHORT:{ns_s} | "
          f"Neutro:{len(ult_z)-nl_s-ns_s}")
    alertas = ult_z[ult_z.abs() >= 2]
    if len(alertas) > 0:
        print()
        for par_n, zv in alertas.items():
            partes  = par_n.split("__")
            direcao = "LONG " if zv <= 0 else "SHORT"
            # busca beta atual se disponível
            key = (partes[0], partes[1]) if partes[0] < partes[1] \
                  else (partes[1], partes[0])
            beta_now = zscore_global.get(key, {}).get("Beta_t")
            beta_str = (f"  β={float(beta_now.iloc[-1]):.3f}"
                        if beta_now is not None and len(beta_now) > 0
                        else "")
            print(f"    {direcao}  {partes[0]:8s} × {partes[1]:8s}"
                  f"  Z={zv:+.2f}{beta_str}")

print(f"\n  Gráficos : {CHARTS_DIR}")
print(f"  CSVs     : {OUTPUT_DIR}")
print(f"{SEP}\n")
