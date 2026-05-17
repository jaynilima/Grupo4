"""
╔══════════════════════════════════════════════════════════════╗
║         PAIRS TRADING B3 — Script Principal (PC)            ║
║                                                              ║
║  Execute: python rodar.py                                    ║
║  Saída  : pasta output/  (CSVs + graficos/)                  ║
╚══════════════════════════════════════════════════════════════╝
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
from matplotlib.ticker import FuncFormatter
from itertools import combinations
from statsmodels.tsa.stattools import coint
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ════════════════════════════════════════════════════════════════
#  CONFIGURACAO — UNICO LUGAR PARA EDITAR
# ════════════════════════════════════════════════════════════════

# Coloque aqui o nome do seu arquivo CSV (sem precisar do caminho
# completo se ele estiver na mesma pasta que este script).
# Exemplos:
#   NOME_CSV = "dados_economatica_B3 (2).csv"
#   NOME_CSV = "minha_base.csv"
NOME_CSV = "dados_economatica_B3 (2).csv"

ANO_INICIO   = 2015
ANO_FIM      = 2025
TOP_N        = 100
MIN_PREGOES  = 0.60
CORR_MIN     = 0.80
PVALUE_MAX   = 0.05


# ════════════════════════════════════════════════════════════════
#  LOCALIZACAO DO CSV — busca automatica
# ════════════════════════════════════════════════════════════════

def encontrar_csv(nome):
    """Procura o CSV na pasta do script e nas pastas comuns do usuario."""
    pasta_script = os.path.dirname(os.path.abspath(__file__))
    candidatos = [
        os.path.join(pasta_script, nome),
        os.path.join(os.path.expanduser("~"), "Desktop", nome),
        os.path.join(os.path.expanduser("~"), "Documents", nome),
        os.path.join(os.path.expanduser("~"), "Downloads", nome),
        os.path.join(os.path.expanduser("~"), "pairs trading", nome),
        os.path.join(os.path.expanduser("~"), nome),
    ]
    for p in candidatos:
        if os.path.isfile(p):
            return p
    return None

CSV_PATH = encontrar_csv(NOME_CSV)

PASTA_SCRIPT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR   = os.path.join(PASTA_SCRIPT, "output")
CHARTS_DIR   = os.path.join(OUTPUT_DIR, "graficos")
os.makedirs(CHARTS_DIR, exist_ok=True)

SEP  = "=" * 66
SEP2 = "-" * 66


# ════════════════════════════════════════════════════════════════
#  BANNER INICIAL — VERIFIQUE O CAMINHO
# ════════════════════════════════════════════════════════════════

print()
print(SEP)
print("  PAIRS TRADING B3")
print(SEP)
print()
print("  VERIFIQUE O CAMINHO DO ARQUIVO CSV:")
print()

if CSV_PATH:
    print(f"  [OK] Arquivo encontrado:")
    print(f"       {CSV_PATH}")
else:
    print(f"  [ERRO] Arquivo NAO encontrado: {NOME_CSV}")
    print()
    print("  O que fazer:")
    print(f"  1. Abra este script: rodar.py")
    print(f"  2. Na linha  NOME_CSV = ...  coloque o nome exato do seu CSV")
    print(f"  3. Coloque o CSV na mesma pasta deste script:")
    print(f"     {PASTA_SCRIPT}")
    print()
    print("  Pastas onde o script procurou:")
    pasta_script = os.path.dirname(os.path.abspath(__file__))
    for p in [pasta_script,
              os.path.join(os.path.expanduser("~"), "Desktop"),
              os.path.join(os.path.expanduser("~"), "Documents"),
              os.path.join(os.path.expanduser("~"), "Downloads")]:
        print(f"     {p}")
    print()
    print(SEP)
    sys.exit(1)

print()
print("  Saida (CSVs + graficos):")
print(f"       {OUTPUT_DIR}")
print()
print("  Parametros:")
print(f"       Anos    : {ANO_INICIO} a {ANO_FIM}")
print(f"       Top N   : {TOP_N} ativos/ano por volume")
print(f"       Corr min: {CORR_MIN}")
print(f"       p-value : < {PVALUE_MAX} (Engle-Granger)")
print()
print(SEP)

# Pausa para o usuario confirmar o caminho
if sys.stdin.isatty():
    try:
        input("\n  Pressione ENTER para continuar (ou Ctrl+C para cancelar)...\n")
    except (KeyboardInterrupt, EOFError):
        print("\n  Cancelado pelo usuario.")
        sys.exit(0)
    sys.exit(0)


# ════════════════════════════════════════════════════════════════
#  PALETA E ESTILO
# ════════════════════════════════════════════════════════════════

C = dict(azul="#1f4e79", azul2="#2e75b6", verde="#1e7e34",
         verm="#c0392b", laran="#e67e22", cinza="#7f8c8d",
         amar="#f1c40f", roxo="#8e44ad", fundo="#f8f9fa", grade="#dde1e7")

plt.rcParams.update({
    "figure.facecolor": C["fundo"], "axes.facecolor": C["fundo"],
    "axes.grid": True, "grid.color": C["grade"], "grid.linewidth": 0.6,
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 12, "axes.labelsize": 10,
})


# ════════════════════════════════════════════════════════════════
#  1. CARREGAMENTO E LIMPEZA
# ════════════════════════════════════════════════════════════════

print(f"[1/4] Carregando dados...")
t0 = time.time()

COL_NAMES = ["Ativo", "Data", "Fechamento", "Abertura", "Minimo",
             "Maximo", "Medio", "Q_Negs", "Volume_BRL_k", "Q_Titulos_k"]

chunks = []
for chunk in pd.read_csv(
        CSV_PATH, encoding="latin-1", names=COL_NAMES,
        header=0, chunksize=200_000, low_memory=False):

    chunk["Ativo"] = chunk["Ativo"].str.replace("<XBSP>", "", regex=False).str.strip()
    chunk["Data"]  = pd.to_datetime(chunk["Data"], errors="coerce")
    chunk = chunk[chunk["Data"].dt.year.between(ANO_INICIO, ANO_FIM)].copy()

    for col in ["Fechamento", "Volume_BRL_k"]:
        chunk[col] = pd.to_numeric(
            chunk[col].astype(str).str.replace(",", ".", regex=False)
                       .replace("-", np.nan), errors="coerce")

    chunk = chunk.dropna(subset=["Fechamento", "Volume_BRL_k"])
    chunk["Ano"] = chunk["Data"].dt.year
    chunks.append(chunk[["Ativo", "Data", "Ano", "Fechamento", "Volume_BRL_k"]])

df_raw = pd.concat(chunks, ignore_index=True)
del chunks

print(f"      {len(df_raw):,} linhas | "
      f"{df_raw['Data'].min().date()} a {df_raw['Data'].max().date()} | "
      f"{df_raw['Ativo'].nunique()} ativos | {time.time()-t0:.0f}s")


# ════════════════════════════════════════════════════════════════
#  2. PIPELINE POR JANELA ANUAL
# ════════════════════════════════════════════════════════════════

print(f"\n[2/4] Pipeline por janela anual...")

anos = sorted(a for a in df_raw["Ano"].unique()
              if ANO_INICIO <= a <= ANO_FIM)

resumo_janelas = []
todos_top100   = []
pares_por_ano  = []
zscore_global  = {}   # (a,b) -> dict com parametros e serie

for ano in anos:
    t_ano  = time.time()
    df_ano = df_raw[df_raw["Ano"] == ano].copy()

    # ── Top N por volume ────────────────────────────────────────
    stats    = df_ano.groupby("Ativo").agg(
        Dias=("Data", "nunique"),
        Volume=("Volume_BRL_k", "sum"),
        Preco_Ultimo=("Fechamento", "last"),
    ).reset_index()
    max_dias = stats["Dias"].max()
    stats    = stats[stats["Dias"] >= max_dias * MIN_PREGOES]
    top100   = stats.nlargest(TOP_N, "Volume").copy()
    top100["Ano"]    = ano
    top100["Rank"]   = range(1, len(top100)+1)
    top100["Vol_MM"] = top100["Volume"] / 1_000
    todos_top100.append(top100)
    tickers = top100["Ativo"].tolist()

    # ── Matriz de precos ────────────────────────────────────────
    df_f   = df_ano[df_ano["Ativo"].isin(tickers)]
    precos = (df_f.pivot_table(index="Data", columns="Ativo",
                                values="Fechamento", aggfunc="last")
                  .sort_index().ffill(limit=3))
    precos = precos.dropna(axis=1, thresh=int(len(precos)*0.80)).dropna()
    tickers = list(precos.columns)
    logP    = np.log(precos)

    # ── Correlacao ──────────────────────────────────────────────
    corr_mat   = logP.corr()
    candidatos = [(a, b, round(corr_mat.loc[a, b], 4))
                  for a, b in combinations(tickers, 2)
                  if abs(corr_mat.loc[a, b]) >= CORR_MIN]

    # ── Engle-Granger ADF ────────────────────────────────────────
    pares_validos = []
    for a, b, corr in candidatos:
        try:
            _, pval, _ = coint(logP[a].values, logP[b].values)
            if pval <= PVALUE_MAX:
                pares_validos.append((a, b, corr, round(pval, 6)))
        except Exception:
            continue

    # ── OLS -> spread -> z-score ─────────────────────────────────
    for a, b, corr, pval in pares_validos:
        ya = logP[a].values
        xb = logP[b].values.reshape(-1, 1)
        reg   = LinearRegression().fit(xb, ya)
        alpha = reg.intercept_
        beta  = reg.coef_[0]
        spread   = ya - (alpha + beta * xb[:, 0])
        mu_sp    = spread.mean()
        sigma_sp = spread.std()
        if sigma_sp < 1e-10:
            continue
        zscore = (spread - mu_sp) / sigma_sp

        pares_por_ano.append({
            "Ano": ano, "Ativo_A": a, "Ativo_B": b,
            "Correlacao": corr, "P_Value": pval,
            "Alpha": round(alpha, 6), "Beta": round(beta, 6),
            "Spread_Mu": round(mu_sp, 6), "Spread_Sigma": round(sigma_sp, 6),
        })

        key = (a, b) if a < b else (b, a)
        zscore_global[key] = {
            "Ano_Formacao": ano,
            "Alpha": alpha, "Beta": beta,
            "Mu": mu_sp, "Sigma": sigma_sp,
            "Correlacao": corr, "P_Value": pval,
            "Z": pd.Series(zscore, index=precos.index, name=f"{a}__{b}"),
            "S": pd.Series(spread,  index=precos.index, name=f"{a}__{b}"),
        }

    top3 = tickers[:3]
    resumo_janelas.append({
        "Ano": ano, "N_Ativos": len(tickers),
        "Candidatos_Corr": len(candidatos),
        "Pares_Cointegrados": len(pares_validos),
        "Dias_Pregao": len(precos),
        "Volume_Total_MM": top100["Vol_MM"].sum(),
        "Top1": top3[0] if len(top3) > 0 else "",
        "Top2": top3[1] if len(top3) > 1 else "",
        "Top3": top3[2] if len(top3) > 2 else "",
    })
    print(f"      {ano}: {len(tickers):3d} ativos | "
          f"{len(candidatos):4d} cand. | "
          f"{len(pares_validos):3d} pares ADF | "
          f"{time.time()-t_ano:.0f}s")


# ════════════════════════════════════════════════════════════════
#  3. CONSOLIDACAO E SALVAMENTO DE CSVs
# ════════════════════════════════════════════════════════════════

print(f"\n[3/4] Consolidando e salvando CSVs...")

resumo_df    = pd.DataFrame(resumo_janelas)
todos_top_df = pd.concat(todos_top100, ignore_index=True)
pares_df     = pd.DataFrame(pares_por_ano)

persistencia = (todos_top_df.groupby("Ativo")
                .agg(Janelas=("Ano","count"),
                     Rank_Medio=("Rank","mean"),
                     Volume_Acum_MM=("Vol_MM","sum"))
                .reset_index()
                .sort_values(["Janelas","Rank_Medio"], ascending=[False,True]))

freq_pares = (pares_df.groupby(["Ativo_A","Ativo_B"])
              .agg(N_Anos=("Ano","count"),
                   P_Value_Min=("P_Value","min"),
                   P_Value_Med=("P_Value","mean"),
                   Corr_Med=("Correlacao", lambda x: x.abs().mean()))
              .reset_index()
              .sort_values(["N_Anos","P_Value_Min"], ascending=[False,True]))

melhores = (pares_df.sort_values("P_Value")
            .drop_duplicates(subset=["Ativo_A","Ativo_B"])
            .reset_index(drop=True))

# Z-scores wide (ultimo periodo de formacao de cada par)
z_wide = pd.DataFrame({f"{a}__{b}": v["Z"]
                        for (a, b), v in zscore_global.items()}).sort_index()

csvs = {
    "resumo_janelas.csv":     resumo_df,
    "pares_por_ano.csv":      pares_df,
    "pares_frequentes.csv":   freq_pares,
    "melhores_pares.csv":     melhores,
    "persistencia_ativos.csv": persistencia,
    "zscores.csv":            z_wide,
}
for nome, df in csvs.items():
    df.to_csv(os.path.join(OUTPUT_DIR, nome), index=(nome == "zscores.csv"))

print(f"      {len(pares_df)} registros | "
      f"{freq_pares.shape[0]} pares unicos | "
      f"{len(zscore_global)} series de z-score")
print(f"      CSVs salvos em: output/")


# ════════════════════════════════════════════════════════════════
#  4. GRAFICOS
# ════════════════════════════════════════════════════════════════

print(f"\n[4/4] Gerando graficos...")

def salvar(fig, nome):
    path = os.path.join(CHARTS_DIR, nome)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"      {nome}")


# ── G1: Universo anual ───────────────────────────────────────────
df_p = resumo_df.copy()
df_p["Vol_BI"] = df_p["Volume_Total_MM"] / 1_000

fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle("Universo Anual — Top 100 Ativos B3 por Volume",
             fontsize=13, fontweight="bold")

ax = axes[0]
bars = ax.bar(df_p["Ano"].astype(str), df_p["Vol_BI"],
              color=C["azul2"], edgecolor="white", linewidth=0.8, zorder=3)
for bar, v in zip(bars, df_p["Vol_BI"]):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.03,
            f"R${v:.0f}Bi", ha="center", fontsize=7.5,
            fontweight="bold", color=C["azul"])
coef = np.polyfit(range(len(df_p)), df_p["Vol_BI"].values, 2)
ax.plot(df_p["Ano"].astype(str),
        np.polyval(coef, range(len(df_p))),
        "--", color=C["laran"], linewidth=2, label="Tendencia")
ax.set_title("Volume Total Anual (R$ Bilhoes)")
ax.set_ylabel("R$ Bilhoes"); ax.legend()
ax.set_ylim(0, df_p["Vol_BI"].max()*1.15)

ax2 = axes[1]
ax2.bar(df_p["Ano"].astype(str), df_p["Candidatos_Corr"],
        color=C["azul2"], edgecolor="white", alpha=0.4, zorder=2,
        label=f"Cand. corr. (>={CORR_MIN})")
ax2.bar(df_p["Ano"].astype(str), df_p["Pares_Cointegrados"],
        color=C["verde"], edgecolor="white", zorder=3,
        label=f"Cointegrados (p<{PVALUE_MAX})")
for bar, v in zip(ax2.patches[len(df_p):], df_p["Pares_Cointegrados"]):
    if v > 0:
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                 str(v), ha="center", fontsize=8,
                 fontweight="bold", color=C["verde"])
ax2.set_title("Candidatos vs Pares Cointegrados por Ano")
ax2.set_ylabel("N de pares"); ax2.legend()

ax3 = axes[2]
ax3.bar(df_p["Ano"].astype(str), df_p["Dias_Pregao"],
        color=C["roxo"], edgecolor="white", alpha=0.85, zorder=3)
ax3.axhline(df_p["Dias_Pregao"].mean(), linestyle="--", color=C["laran"],
            linewidth=1.5, label=f"Media: {df_p['Dias_Pregao'].mean():.0f}d/ano")
ax3.set_title("Dias de Pregao por Ano")
ax3.set_ylabel("Pregoes"); ax3.set_ylim(150, 260); ax3.legend()

plt.tight_layout()
salvar(fig, "01_universo_anual.png")


# ── G2: Persistencia ────────────────────────────────────────────
n_tot  = len(anos)
top30  = persistencia.head(30)
cbs    = [C["verde"] if j == n_tot else C["azul2"] if j >= n_tot-2
          else C["amar"] if j >= n_tot-4 else C["cinza"]
          for j in top30["Janelas"]]

fig, ax = plt.subplots(figsize=(12, 8))
ax.barh(top30["Ativo"][::-1], top30["Janelas"][::-1],
        color=list(reversed(cbs)), edgecolor="white", linewidth=0.6)
ax.axvline(n_tot, linestyle="--", color=C["verm"], linewidth=1.3,
           label=f"Max: {n_tot} janelas")
ax.set_xlim(0, n_tot+2)
ax.set_xlabel(f"Anos no Top {TOP_N} (de {n_tot} possiveis)")
ax.set_title(f"Persistencia dos Ativos no Top {TOP_N} por Volume\n"
             f"{ANO_INICIO}-{ANO_FIM}  |  Verde=todas as janelas",
             fontweight="bold")
for v, yp in zip(top30["Janelas"][::-1], range(len(top30))):
    ax.text(v+0.1, yp, str(v), va="center", fontsize=8.5)
patches = [mpatches.Patch(color=C["verde"],  label=f"Todas ({n_tot})"),
           mpatches.Patch(color=C["azul2"], label=f">{n_tot-3}"),
           mpatches.Patch(color=C["amar"],  label=f">{n_tot-5}"),
           mpatches.Patch(color=C["cinza"], label="demais")]
ax.legend(handles=patches, loc="lower right")
plt.tight_layout()
salvar(fig, "02_persistencia_ativos.png")


# ── G3: Analise dos pares ────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle(f"Analise de Cointegração — {len(pares_df)} pares-janela | "
             f"{freq_pares.shape[0]} pares unicos",
             fontsize=13, fontweight="bold")

ax = axes[0]
ax.hist(pares_df["P_Value"], bins=20, color=C["azul2"], edgecolor="white")
ax.axvline(PVALUE_MAX, linestyle="--", color=C["verm"], linewidth=1.5,
           label=f"p={PVALUE_MAX}")
ax.set_title("Distribuicao P-values ADF")
ax.set_xlabel("P-value"); ax.set_ylabel("Frequencia"); ax.legend()

ax2 = axes[1]
sc = ax2.scatter(pares_df["Correlacao"].abs(), pares_df["P_Value"],
                 c=pares_df["P_Value"], cmap="RdYlGn_r",
                 alpha=0.5, s=25, edgecolors="none",
                 vmin=0, vmax=PVALUE_MAX)
plt.colorbar(sc, ax=ax2, label="P-value")
ax2.axhline(PVALUE_MAX, linestyle="--", color=C["verm"], linewidth=1.2)
ax2.set_title("|Correlacao| vs P-value")
ax2.set_xlabel("|Correlacao|"); ax2.set_ylabel("P-value ADF")

ax3 = axes[2]
serie_ano = pares_df.groupby("Ano")["Ativo_A"].count()
ax3.bar(serie_ano.index.astype(str), serie_ano.values,
        color=C["verde"], edgecolor="white")
ax3.axhline(serie_ano.mean(), linestyle="--", color=C["laran"],
            linewidth=1.5, label=f"Media: {serie_ano.mean():.0f}/ano")
ax3.set_title("Pares Cointegrados por Ano")
ax3.set_ylabel("N de pares"); ax3.legend()

plt.tight_layout()
salvar(fig, "03_pares_cointegrados.png")


# ── G4: Ranking de pares ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Ranking dos Pares — Frequencia e Forca Estatistica",
             fontweight="bold", fontsize=13)

top_freq = freq_pares.head(20).copy()
top_freq["Par"] = top_freq["Ativo_A"] + " x " + top_freq["Ativo_B"]
cbs_f = [C["verde"] if n == n_tot else C["azul2"] if n >= n_tot-2
         else C["amar"] for n in top_freq["N_Anos"]]

ax = axes[0]
ax.barh(top_freq["Par"][::-1], top_freq["N_Anos"][::-1],
        color=list(reversed(cbs_f)), edgecolor="white")
ax.set_xlim(0, n_tot+1)
ax.set_xlabel("Janelas em que o par foi cointegrado")
ax.set_title(f"Top 20 — Mais Frequentes (max={n_tot} janelas)")
for v, yp in zip(top_freq["N_Anos"][::-1], range(len(top_freq))):
    ax.text(v+0.05, yp, str(v), va="center", fontsize=8.5)

top_forte = melhores.head(20).copy()
top_forte["Par"] = top_forte["Ativo_A"] + " x " + top_forte["Ativo_B"]
cbs_s = [C["verde"] if p <= 0.01 else C["azul2"] if p <= 0.03
         else C["amar"] for p in top_forte["P_Value"]]

ax2 = axes[1]
ax2.barh(top_forte["Par"][::-1], top_forte["Correlacao"].abs()[::-1],
         color=list(reversed(cbs_s)), edgecolor="white")
ax2.set_xlim(0.75, 1.02)
ax2.set_xlabel("|Correlacao|")
ax2.set_title("Top 20 — Menor P-value ADF\n"
              "Verde: p<0.01 | Azul: p<0.03 | Amarelo: p<0.05")
for i, (_, row) in enumerate(top_forte[::-1].iterrows()):
    ax2.text(abs(row["Correlacao"])+0.002, i,
             f"p={row['P_Value']:.4f}", va="center",
             fontsize=7.5, color=C["azul"])

plt.tight_layout()
salvar(fig, "04_pares_ranking.png")


# ── G5: Parametros OLS ──────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Parametros OLS — log(A) = alpha + beta*log(B)",
             fontsize=13, fontweight="bold")

ax = axes[0]
ax.hist(pares_df["Beta"], bins=25, color=C["verde"], edgecolor="white")
ax.axvline(1.0, linestyle="--", color=C["laran"], linewidth=1.5, label="beta=1")
ax.axvline(pares_df["Beta"].median(), linestyle="-", color=C["azul"],
           linewidth=1.5, label=f"Mediana={pares_df['Beta'].median():.2f}")
ax.set_title("Hedge Ratio (beta)"); ax.legend()

ax2 = axes[1]
ax2.hist(pares_df["Spread_Sigma"], bins=25, color=C["laran"], edgecolor="white")
ax2.axvline(pares_df["Spread_Sigma"].mean(), linestyle="--", color=C["azul"],
            linewidth=1.5,
            label=f"Media={pares_df['Spread_Sigma'].mean():.3f}")
ax2.set_title("Desvio-Padrao do Spread (sigma)"); ax2.legend()

ax3 = axes[2]
anos_bp  = sorted(pares_df["Ano"].unique())
data_bp  = [pares_df[pares_df["Ano"]==a]["Beta"].values for a in anos_bp]
bp = ax3.boxplot(data_bp, labels=[str(a) for a in anos_bp], patch_artist=True)
for patch in bp["boxes"]:
    patch.set_facecolor(C["azul2"]); patch.set_alpha(0.7)
ax3.axhline(1.0, linestyle="--", color=C["laran"], linewidth=1.2, label="beta=1")
ax3.set_title("Beta por Ano"); ax3.legend()
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha="right")

plt.tight_layout()
salvar(fig, "05_parametros_ols.png")


# ── G6: Z-score top 6 pares ─────────────────────────────────────
pares_z = sorted(zscore_global.items(), key=lambda x: x[1]["P_Value"])
top6    = pares_z[:6]

fig, axes = plt.subplots(3, 2, figsize=(16, 13))
fig.suptitle("Z-Score — Top 6 Pares (menor p-value ADF)\n"
             "Vermelho: SHORT (+2s) | Verde: LONG (-2s)",
             fontsize=13, fontweight="bold")
axes = axes.flatten()

for i, ((at, bt), info) in enumerate(top6):
    ax  = axes[i]
    z   = info["Z"].dropna()
    nl  = (z <= -2).sum(); ns = (z >= 2).sum()
    pct = (nl+ns)/len(z)*100

    ax.plot(z.index, z.values, color=C["azul2"], linewidth=0.85, alpha=0.9)
    for lvl, col, ls in [(2,C["verm"],"--"),(-2,C["verde"],"--"),
                          (3,C["verm"],":"  ),(-3,C["verde"],":"),
                          (0,C["cinza"],"-")]:
        ax.axhline(lvl, linestyle=ls, color=col,
                   linewidth=1.2 if abs(lvl)==2 else 0.8, alpha=0.85)
    ax.fill_between(z.index, 2,  z.values, where=(z.values>=2),
                    alpha=0.2, color=C["verm"])
    ax.fill_between(z.index, -2, z.values, where=(z.values<=-2),
                    alpha=0.2, color=C["verde"])
    ax.set_title(
        f"{at} x {bt}  |r|={abs(info['Correlacao']):.3f}  "
        f"p={info['P_Value']:.4f}  (formado {info['Ano_Formacao']})\n"
        f"LONG: {nl}d | SHORT: {ns}d | {pct:.1f}% do tempo ativo",
        fontsize=9.5, fontweight="bold")
    ax.set_ylabel("Z-Score"); ax.set_ylim(-5, 5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.text(z.index[-1], 2.1,  "+2s", color=C["verm"],  fontsize=8)
    ax.text(z.index[-1], -2.4, "-2s", color=C["verde"], fontsize=8)

plt.tight_layout()
salvar(fig, "06_zscore_top6.png")


# ── G7: Spread top 4 ────────────────────────────────────────────
top4 = pares_z[:4]

fig, axes = plt.subplots(2, 2, figsize=(16, 9))
fig.suptitle("Spread Bruto (log-precos) — Top 4 Pares\n"
             "Spread = log(A) - alpha - beta*log(B)",
             fontsize=13, fontweight="bold")
axes = axes.flatten()

for i, ((at, bt), info) in enumerate(top4):
    ax  = axes[i]
    s   = info["S"].dropna()
    mu  = s.mean(); sig = s.std()
    ax.plot(s.index, s.values, color=C["azul2"], linewidth=0.8, alpha=0.9)
    ax.axhline(mu,         linestyle="-",  color=C["cinza"], linewidth=1.2,
               label=f"mu={mu:.4f}")
    ax.axhline(mu+2*sig,   linestyle="--", color=C["verm"],  linewidth=1.2,
               label=f"+2s={mu+2*sig:.4f}")
    ax.axhline(mu-2*sig,   linestyle="--", color=C["verde"], linewidth=1.2,
               label=f"-2s={mu-2*sig:.4f}")
    ax.fill_between(s.index, mu+2*sig, s.values,
                    where=(s.values>=mu+2*sig), alpha=0.2, color=C["verm"])
    ax.fill_between(s.index, mu-2*sig, s.values,
                    where=(s.values<=mu-2*sig), alpha=0.2, color=C["verde"])
    ax.set_title(f"Spread: {at} x {bt}  "
                 f"(sigma={sig:.4f} | formado {info['Ano_Formacao']})",
                 fontweight="bold", fontsize=10)
    ax.set_ylabel("Spread (log)"); ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())

plt.tight_layout()
salvar(fig, "07_spread_top4.png")


# ── G8: Par destaque ────────────────────────────────────────────
(ad, bd), id_ = pares_z[0]
zd = id_["Z"].dropna()
sd = id_["S"].dropna()

fig = plt.figure(figsize=(15, 9))
fig.suptitle(
    f"Par Destaque: {ad} x {bd}\n"
    f"|r|={abs(id_['Correlacao']):.4f}  p={id_['P_Value']:.4f}  "
    f"beta={id_['Beta']:.3f}  formado em {id_['Ano_Formacao']}",
    fontsize=12, fontweight="bold")
gs = GridSpec(3, 1, hspace=0.45, figure=fig)

ax1 = fig.add_subplot(gs[0])
mu_d = sd.mean(); sg_d = sd.std()
ax1.plot(sd.index, sd.values, color=C["azul2"], linewidth=0.9)
ax1.axhline(mu_d, linestyle="--", color=C["cinza"], linewidth=1,
            label=f"mu={mu_d:.4f}")
ax1.fill_between(sd.index, mu_d+2*sg_d, sd.values,
                 where=(sd.values>=mu_d+2*sg_d), alpha=0.2, color=C["verm"])
ax1.fill_between(sd.index, mu_d-2*sg_d, sd.values,
                 where=(sd.values<=mu_d-2*sg_d), alpha=0.2, color=C["verde"])
ax1.set_ylabel("Spread (log)"); ax1.legend(fontsize=9)
ax1.set_title("Spread bruto")
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax1.xaxis.set_major_locator(mdates.YearLocator())

ax2 = fig.add_subplot(gs[1])
ax2.plot(zd.index, zd.values, color=C["azul"], linewidth=0.9, alpha=0.9)
for lvl, col, ls in [(2,C["verm"],"--"),(-2,C["verde"],"--"),
                      (3,C["verm"],":"  ),(-3,C["verde"],":"),
                      (0,C["cinza"],"-")]:
    ax2.axhline(lvl, linestyle=ls, color=col,
                linewidth=1.2 if abs(lvl)<=2 else 0.9)
ax2.fill_between(zd.index, 2,  zd.values, where=(zd.values>=2),
                 alpha=0.2, color=C["verm"])
ax2.fill_between(zd.index, -2, zd.values, where=(zd.values<=-2),
                 alpha=0.2, color=C["verde"])
nl = (zd<=-2).sum(); ns = (zd>=2).sum()
ax2.set_ylabel("Z-Score"); ax2.set_ylim(-5, 5)
ax2.set_title(f"Z-Score completo — LONG: {nl}d | SHORT: {ns}d | Neutro: {len(zd)-nl-ns}d")
ax2.text(zd.index[-1], 2.1,  "+2s (SHORT)", color=C["verm"],  fontsize=8)
ax2.text(zd.index[-1], -2.4, "-2s (LONG)",  color=C["verde"], fontsize=8)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax2.xaxis.set_major_locator(mdates.YearLocator())

ax3 = fig.add_subplot(gs[2])
zr = zd[zd.index >= pd.Timestamp("2024-01-01")]
ax3.plot(zr.index, zr.values, color=C["azul"], linewidth=1.2)
for lvl, col, ls in [(2,C["verm"],"--"),(-2,C["verde"],"--"),(0,C["cinza"],"-")]:
    ax3.axhline(lvl, linestyle=ls, color=col, linewidth=1.2)
ax3.fill_between(zr.index, 2,  zr.values, where=(zr.values>=2),  alpha=0.25, color=C["verm"])
ax3.fill_between(zr.index, -2, zr.values, where=(zr.values<=-2), alpha=0.25, color=C["verde"])
ax3.set_ylabel("Z-Score"); ax3.set_ylim(-4, 4)
ax3.set_title(f"Detalhe 2024-atual  |  Z atual = {zd.iloc[-1]:+.2f}  ({zd.index[-1].date()})")
ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b/%Y"))
ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

plt.savefig(os.path.join(CHARTS_DIR, "08_par_destaque.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)
print("      08_par_destaque.png")


# ── G9: Sinais ativos ao longo do tempo ─────────────────────────
sl = (z_wide <= -2).sum(axis=1)
ss = (z_wide >=  2).sum(axis=1)
st = sl + ss
sm_l = sl.resample("ME").mean()
sm_s = ss.resample("ME").mean()
sm_t = st.resample("ME").mean()

fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
fig.suptitle("Sinais Ativos (|Z| >= 2) ao Longo do Tempo",
             fontsize=13, fontweight="bold")

ax = axes[0]
ax.fill_between(sm_t.index, sm_t.values, alpha=0.5, color=C["laran"])
ax.plot(sm_t.index, sm_t.values, color=C["laran"], linewidth=1.5)
ax.axhline(sm_t.mean(), linestyle="--", color=C["azul"], linewidth=1.2,
           label=f"Media: {sm_t.mean():.1f} pares/mes")
ax.set_title("Total de Pares com Sinal Ativo — Media Mensal")
ax.set_ylabel("N de pares"); ax.legend()

ax2 = axes[1]
ax2.fill_between(sm_l.index, sm_l.values, alpha=0.5, color=C["verde"],
                 label="LONG (Z <= -2)")
ax2.fill_between(sm_s.index, sm_s.values, alpha=0.5, color=C["verm"],
                 label="SHORT (Z >= +2)")
ax2.plot(sm_l.index, sm_l.values, color=C["verde"], linewidth=1.2)
ax2.plot(sm_s.index, sm_s.values, color=C["verm"],  linewidth=1.2)
ax2.set_title("LONG vs SHORT — Media Mensal")
ax2.set_ylabel("N de pares"); ax2.legend()
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax2.xaxis.set_major_locator(mdates.YearLocator())

plt.tight_layout()
salvar(fig, "09_sinais_temporais.png")


# ── G10: Snapshot atual ──────────────────────────────────────────
ultimo_z = z_wide.iloc[-1].dropna().sort_values()
nl_now   = (ultimo_z <= -2).sum()
ns_now   = (ultimo_z >=  2).sum()
data_ref = z_wide.index[-1].strftime("%d/%m/%Y")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle(f"Snapshot de Z-Scores — {data_ref}",
             fontsize=13, fontweight="bold")

ax = axes[0]
cbs_snap = [C["verm"] if z >= 2 else C["verde"] if z <= -2 else C["cinza"]
            for z in ultimo_z.values]
ax.bar(range(len(ultimo_z)), ultimo_z.values,
       color=cbs_snap, edgecolor="none", width=1.0)
ax.axhline( 2, linestyle="--", color=C["verm"],  linewidth=1.2,
            label="+2s (SHORT)")
ax.axhline(-2, linestyle="--", color=C["verde"], linewidth=1.2,
            label="-2s (LONG)")
ax.axhline( 0, linestyle="-",  color=C["cinza"], linewidth=0.7)
ax.set_title(f"Todos os {len(ultimo_z)} pares ordenados\n"
             f"SHORT: {ns_now} | Neutro: {len(ultimo_z)-nl_now-ns_now} | LONG: {nl_now}")
ax.set_xlabel("Pares (ordenados por z-score)")
ax.set_ylabel("Z-Score"); ax.legend()

ax2 = axes[1]
ax2.hist(ultimo_z.values, bins=20, color=C["azul2"], edgecolor="white")
ax2.axvline( 2, linestyle="--", color=C["verm"],  linewidth=1.5, label="+2s")
ax2.axvline(-2, linestyle="--", color=C["verde"], linewidth=1.5, label="-2s")
ax2.axvline( 0, linestyle="-",  color=C["cinza"], linewidth=0.8)
ax2.set_title("Distribuicao dos Z-Scores Atuais")
ax2.set_xlabel("Z-Score"); ax2.set_ylabel("Frequencia"); ax2.legend()

plt.tight_layout()
salvar(fig, "10_snapshot_zscores.png")


# ── G11: Heatmap pares x anos ────────────────────────────────────
pares_df["Par"] = pares_df["Ativo_A"] + "_" + pares_df["Ativo_B"]
top_hm = freq_pares.head(30).copy()
top_hm["Par"] = top_hm["Ativo_A"] + "_" + top_hm["Ativo_B"]

hm_data  = pares_df[pares_df["Par"].isin(top_hm["Par"].values)]
pivot_hm = hm_data.pivot_table(index="Par", columns="Ano",
                                values="P_Value", aggfunc="min")
pivot_hm = pivot_hm.reindex(
    [p for p in top_hm["Par"].tolist() if p in pivot_hm.index])
pivot_hm.index = [p.replace("_", " x ") for p in pivot_hm.index]

fig, ax = plt.subplots(figsize=(14, max(6, len(pivot_hm)*0.4)))
im = ax.imshow(pivot_hm.values, aspect="auto", cmap="RdYlGn_r",
               vmin=0, vmax=PVALUE_MAX,
               extent=[-0.5, pivot_hm.shape[1]-0.5,
                        pivot_hm.shape[0]-0.5, -0.5])
plt.colorbar(im, ax=ax, label="P-value ADF (verde = forte)")
ax.set_xticks(range(len(pivot_hm.columns)))
ax.set_xticklabels(pivot_hm.columns.astype(str), rotation=45)
ax.set_yticks(range(len(pivot_hm.index)))
ax.set_yticklabels(pivot_hm.index, fontsize=9)
ax.set_title(f"Heatmap — Top {len(pivot_hm)} Pares Mais Frequentes\n"
             "Verde = cointegrado no ano | Branco = nao apareceu",
             fontweight="bold")
ax.set_xlabel("Ano de Formacao")

for i in range(pivot_hm.shape[0]):
    for j in range(pivot_hm.shape[1]):
        val = pivot_hm.iloc[i, j]
        if not np.isnan(val):
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=6.5,
                    color="white" if val < 0.02 else "black")

plt.tight_layout()
salvar(fig, "11_heatmap_pares.png")


# ── G12: Pipeline ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(15, 7))
ax.set_xlim(0, 14); ax.set_ylim(0, 8); ax.axis("off")

fases = [
    (1.0, 5.8, "DADOS BRUTOS",
     f"Economatica/B3\n{len(df_raw):,.0f} linhas\n{ANO_INICIO}-{ANO_FIM}\n{df_raw['Ativo'].nunique()} ativos",
     C["cinza"]),
    (3.6, 5.8, "LIMPEZA",
     "Remove NaN\nForward fill (3d)\nConversao float64\nSufixo <XBSP>", C["azul"]),
    (6.2, 5.8, "TOP 100 / ANO",
     f"Por janela anual\nFiltro {int(MIN_PREGOES*100)}% pregoes\n{len(anos)} janelas\n4.950 pares/janela",
     C["azul2"]),
    (8.8, 5.8, "PARES / ANO",
     f"Corr >= {CORR_MIN}\nEngle-Granger ADF\np < {PVALUE_MAX}\n{freq_pares.shape[0]} pares unicos",
     C["roxo"]),
    (11.4, 5.8, "SPREAD & Z",
     f"OLS log-precos\nSpread = residuo\nZ = (s-mu)/sigma\n{len(zscore_global)} series diarias",
     C["verde"]),
]
for x, y, titulo, corpo, cor in fases:
    rect = mpatches.FancyBboxPatch(
        (x-1.05, y-1.15), 2.0, 2.1,
        boxstyle="round,pad=0.07",
        facecolor=cor, edgecolor="white", linewidth=2.0, alpha=0.90, zorder=3)
    ax.add_patch(rect)
    ax.text(x, y+0.75, titulo, ha="center", va="center",
            fontsize=8.5, fontweight="bold", color="white", zorder=4)
    ax.text(x, y-0.2,  corpo,  ha="center", va="center",
            fontsize=7.5, color="white", alpha=0.95, zorder=4)
for xa, xb in [(2.05,2.55),(4.65,5.15),(7.25,7.75),(9.85,10.35)]:
    ax.annotate("", xy=(xb, 5.8), xytext=(xa, 5.8),
                arrowprops=dict(arrowstyle="-|>", color=C["azul2"],
                                lw=2.0, mutation_scale=16), zorder=5)

ax.text(7.0, 7.3, "FASES 1 E 2 CONCLUIDAS", ha="center", fontsize=11,
        fontweight="bold", color=C["verde"],
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#e8f5e9",
                  edgecolor=C["verde"], linewidth=1.5))

proximos = [
    (3.5, 2.5, "BACKTESTING",
     "Simular sinais\nSharpe / Drawdown\nRetorno por par/ano"),
    (7.0, 2.5, "CALIBRACAO",
     "Z: 1.5 / 2.0 / 2.5\nStop |Z|>3 e t>30d\nOtimizar por janela"),
    (10.5, 2.5, "MONITORAMENTO",
     "Z-score diario\nAlertas de entrada\nDashboard live"),
]
for x, y, titulo, corpo in proximos:
    rect = mpatches.FancyBboxPatch(
        (x-1.5, y-0.9), 2.9, 1.7,
        boxstyle="round,pad=0.07",
        facecolor=C["laran"], edgecolor="white", linewidth=1.5, alpha=0.80, zorder=3)
    ax.add_patch(rect)
    ax.text(x, y+0.58, titulo, ha="center", fontsize=8.5,
            fontweight="bold", color="white", zorder=4)
    ax.text(x, y-0.2,  corpo,  ha="center", fontsize=7.5,
            color="white", alpha=0.95, zorder=4)

ax.text(7.0, 4.2, "PROXIMAS ETAPAS", ha="center", fontsize=10,
        fontweight="bold", color=C["laran"])
ax.set_title(f"Pipeline Completo — Pairs Trading B3 ({ANO_INICIO}-{ANO_FIM})",
             fontsize=14, fontweight="bold", pad=18, color=C["azul"])
plt.tight_layout()
salvar(fig, "12_pipeline.png")


# ════════════════════════════════════════════════════════════════
#  RESUMO FINAL
# ════════════════════════════════════════════════════════════════

print()
print(SEP)
print("RESUMO FINAL")
print(SEP)
print(f"\n  Periodo    : {ANO_INICIO} a {ANO_FIM} ({len(anos)} janelas anuais)")
print(f"  Ativos/ano : top {TOP_N} por volume financeiro")
print(f"  Corr. min  : |rho| >= {CORR_MIN}")
print(f"  Criterio   : p-value ADF < {PVALUE_MAX}")
print()
print("  JANELAS:")
print("  " + resumo_df[["Ano","N_Ativos","Candidatos_Corr",
                          "Pares_Cointegrados","Dias_Pregao",
                          "Top1","Top2","Top3"]].to_string(index=False)
                                                .replace("\n", "\n  "))

print(f"\n  PARES MAIS FREQUENTES (top 10):")
print("  " + freq_pares.head(10)[["Ativo_A","Ativo_B",
                                   "N_Anos","P_Value_Min","Corr_Med"]]
                        .to_string(index=False)
                        .replace("\n", "\n  "))

print(f"\n  PARES MAIS FORTES (menor p-value | top 10):")
print("  " + melhores.head(10)[["Ativo_A","Ativo_B","Ano",
                                  "Correlacao","P_Value","Beta","Spread_Sigma"]]
                      .to_string(index=False)
                      .replace("\n", "\n  "))

alertas = ultimo_z[ultimo_z.abs() >= 2].sort_values()
print(f"\n  SNAPSHOT ATUAL ({z_wide.index[-1].date()}) — "
      f"LONG: {nl_now} | SHORT: {ns_now} | Neutro: {len(ultimo_z)-nl_now-ns_now}")
if len(alertas) > 0:
    print()
    for par_n, zv in alertas.items():
        partes = par_n.split("__")
        direcao = "LONG " if zv <= 0 else "SHORT"
        print(f"    {direcao}  {partes[0]:8s} x {partes[1]:8s}  Z = {zv:+.2f}")

print()
print(SEP)
print(f"  Graficos : {CHARTS_DIR}")
print(f"  CSVs     : {OUTPUT_DIR}")
print(SEP)
print()
