!pip install yahooquery -q

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
import logging
from yahooquery import Ticker

warnings.filterwarnings('ignore')
logging.getLogger('matplotlib').setLevel(logging.ERROR)

# pares com alta liquidez na B3
pares = [
    ("(i)", "Mesma empresa, classes diferentes",
     "PETR3.SA", "PETR4.SA",
     "Mesmo fluxo de caixa; diferem em direitos políticos."),

    ("(ii)", "Holding e controlada",
     "ITSA4.SA", "ITUB4.SA",
     "Itaúsa é a holding cuja maior posição é o próprio Itaú."),

    ("(iii)", "Mesmo setor — Bancos",
     "BBAS3.SA", "BBDC4.SA",
     "Banco do Brasil x Bradesco: mesma exposição a Selic e spread."),

    ("(iv)", "Mesma cadeia produtiva",
     "VALE3.SA", "CSNA3.SA",
     "Vale produz minério; CSN o consome para fabricar aço."),

    ("(v)", "Fator macro comum (USD)",
     "WEGE3.SA", "SUZB3.SA",
     "WEG e Suzano: Exportadoras massivas, receita ligada ao Dólar."),

    ("(vi)", "Arbitragem Setorial",
     "RENT3.SA", "MOVI3.SA",
     "Localiza x Movida: Fator de risco idêntico (Locação e Seminovos)."),
]

# config visual (estilo Financial Times)
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.edgecolor'] = '#666666'
plt.rcParams['axes.linewidth'] = 0.6
plt.rcParams['figure.facecolor'] = '#fbfaf7'
plt.rcParams['axes.facecolor'] = '#fbfaf7'
plt.rcParams['grid.color'] = '#e5e5e5'
plt.rcParams['grid.linewidth'] = 0.5

inicio = '2020-01-01'
fim = '2024-01-01'

fig, axes = plt.subplots(2, 3, figsize=(17, 10))
axes = axes.flatten()

for ax, (codigo, titulo, t1, t2, desc) in zip(axes, pares):
    
    # 1. puxando os dados do yahoo query (yfinance deu erro)
    try:
        ativos = Ticker(f"{t1} {t2}")
        historico = ativos.history(start=inicio, end=fim)
        
        # o yahooquery traz os dados empilhados. o unstack organiza as colunas.
        # usamos 'adjclose' para descontar dividendos 
        df_bruto = historico['adjclose'].unstack(level=0)
        
        # ffill() preenche buracos de dias sem negociação copiando o dia anterior.
        df = df_bruto.ffill().dropna()
        
    except Exception as e:
        print(f"Deu ruim ao baixar {t1} e {t2}: {e}")
        df = pd.DataFrame()

    # tratamento caso algum par ainda falhe
    if df.empty or len(df) < 50:
        ax.text(0.5, 0.5, "sem dados / erro na base", ha='center', va='center',
                transform=ax.transAxes, color='#666')
        ax.set_title(f"{codigo}  {titulo}", loc='left', fontweight='bold', pad=14)
        continue

    # normaliza tudo pra base 100 no primeiro pregão
    norm = (df / df.iloc[0]) * 100
    s1 = norm[t1]
    s2 = norm[t2]

    # correlação dos retornos diários (rho)
    corr = df.pct_change().corr().iloc[0, 1]

    # cor azul fica pro primeiro ativo, vermelho pro segundo
    azul = '#0a3d62'
    vermelho = '#c44536'

    # faixa cinza entre as duas linhas pra mostrar o spread
    ax.fill_between(norm.index, s1, s2, color='#d4d4d4', alpha=0.55, linewidth=0)

    # linha do 100 (referência)
    ax.axhline(100, color='#666', linewidth=0.5, linestyle='--', alpha=0.4)

    # as duas séries (o replace tira o .SA só pra ficar limpo na legenda)
    ax.plot(norm.index, s1, color=azul, linewidth=1.4, label=t1.replace('.SA', ''))
    ax.plot(norm.index, s2, color=vermelho, linewidth=1.4, label=t2.replace('.SA', ''))

    # marca o ponto de maior diferença entre os dois (onde mais "abriu" o spread)
    idx_max = (s1 - s2).abs().idxmax()
    v1 = s1.loc[idx_max]
    v2 = s2.loc[idx_max]
    ax.plot([idx_max, idx_max], [v1, v2], color='#f0a500', linewidth=1.8, alpha=0.85)
    ax.scatter([idx_max, idx_max], [v1, v2], color='#f0a500', s=22,
               edgecolor='white', linewidth=1.2, zorder=5)

    # titulo do painel
    ax.set_title(f"{codigo}  {titulo}", loc='left', fontweight='bold',
                 fontsize=11, pad=14)

    # rho no canto, com cor variando conforme a força
    if corr > 0.7:
        cor_rho = azul
    elif corr > 0.5:
        cor_rho = '#a06800'
    else:
        cor_rho = '#666'
    ax.text(0.985, 1.04, f"ρ = {corr:.2f}", transform=ax.transAxes,
            ha='right', va='bottom', fontweight='bold', fontsize=10.5, color=cor_rho)

    # descrição econômica + barrinha azul lateral (estilo citação)
    ax.text(0.04, 0.94, desc, transform=ax.transAxes, fontsize=8.5,
            style='italic', color='#666', va='top')
    ax.annotate('', xy=(0.025, 0.83), xytext=(0.025, 0.95),
                xycoords='axes fraction',
                arrowprops=dict(arrowstyle='-', color=azul, linewidth=2.2))

    # ajustes finais
    ax.legend(loc='lower left', frameon=False, fontsize=9, handlelength=1.5)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.tick_params(axis='x', length=0, labelsize=8, colors='#666')
    ax.tick_params(axis='y', length=3, labelsize=8, colors='#666')
    ax.set_ylabel('Base 100', fontsize=8.5, color='#666')
    ax.grid(alpha=0.8)

# titulo geral e subtitulo
fig.suptitle("Lógica econômica dos pares: seis categorias de vínculo",
             fontsize=15.5, fontweight='bold', x=0.06, y=0.985, ha='left')
fig.text(0.06, 0.954,
         f"Preços ajustados (descontando proventos) e normalizados na base 100. ρ = correlação.\n"
         f"Período: {inicio} a {fim}.",
         fontsize=9.5, color='#666', style='italic')

# legenda dos elementos visuais (rodapé)
fig.text(0.06, 0.018,
         "Faixa cinza  =  spread do par.   "
         "Marcador âmbar  =  ponto de máxima divergência.   "
         "Fonte: Dados B3 via yahooquery.",
         fontsize=8.5, color='#666')

plt.tight_layout(rect=[0.04, 0.04, 0.985, 0.92])
plt.show()
