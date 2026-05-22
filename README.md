# Aplicação da estratégia de Pairs Trading no contexto acionário brasileiro

## Contexto 
### O que é Pairs Trading
A literatura (Gatev et al., 2006; Vidyamurthy, 2004) classifica os vínculos econômicos válidos em seis categorias:

(i) Mesma empresa, classes diferentes — ações ON e PN da mesma companhia compartilham o mesmo fluxo de caixa e diferem apenas em direitos políticos. Ex.: PETR3/PETR4, GGBR3/GGBR4.

(ii) Holding e controlada — o valor da holding deriva da participação na controlada, com desconto historicamente estável. Ex.: BRAP4/VALE3, ITSA4/ITUB4.

(iii) Mesmo setor e modelo de negócio — concorrentes sujeitos aos mesmos drivers de demanda, custos e regulação. Ex.: ITUB4/BBDC4 (bancos), JBSS3/MRFG3 (frigoríficos), ASAI3/CRFB3 (atacarejo).

(iv) Mesma cadeia produtiva — empresas em elos diferentes da mesma cadeia, expostas ao mesmo ciclo. Ex.: VALE3/CSNA3 (minério-aço), SUZB3/KLBN11 (papel-celulose).

(v) Fator macro comum — ativos cuja relação vem de exposição compartilhada a um driver externo (câmbio, juros, commodity). Ex.: exportadoras dolarizadas; empresas alavancadas sensíveis à Selic.

(vi) Arbitragem direta — mesmo ativo em mercados diferentes (ações ON na B3 vs. ADRs na NYSE).

### Histórico

A estratégia de pairs trading foi formalizada por Gatev, Goetzmann e Rouwenhorst (2006), mas já era praticada por desks quantitativos desde os anos 1980, com destaque para o grupo de Morgan Stanley liderado por Nunzio Tartaglia. O arcabouço estatístico que sustenta a estratégia — a teoria da cointegração — foi desenvolvido por Engle e Granger (1987), rendendo-lhes o Prêmio Nobel de Economia em 2003.

No Brasil, a pesquisa empírica sobre pairs trading na B3 ainda é escassa em comparação com o mercado americano. O ambiente brasileiro apresenta características que tornam o tema particularmente interessante: a existência estrutural de pares ON/PN da mesma empresa (ex.: PETR3/PETR4), a relação holding-controlada em grandes conglomerados (ex.: ITSA4/ITUB4), e setores com poucas empresas de capital aberto operando em condições quase idênticas (energia elétrica, bancos, frigoríficos). Esses fatores criam vínculos econômicos persistentes que são candidatos naturais à cointegração de longo prazo.

### Importância para o mercado financeiro

Pairs trading é relevante por múltiplas razões. Como estratégia *market-neutral*, gera retorno independente da direção geral do mercado, o que a torna atraente para gestão de risco em carteiras de ações. Do ponto de vista de eficiência de mercado, a estratégia contribui para acelerar a convergência de preços de ativos com fundamentos comuns, reduzindo distorções temporárias. Para analistas e gestores quantitativos, ela oferece um método rigoroso de formalizar relações econômicas já conhecidas informalmente — como o desconto estrutural de uma holding frente à sua controlada — e transformá-las em regras operacionais testáveis.

No contexto brasileiro, a estratégia também serve como instrumento de análise comparativa entre setores: diferentes segmentos da B3 apresentam graus distintos de cointegração, e mapear essas diferenças permite compreender melhor a estrutura de dependência do mercado acionário nacional.

## Objetivo geral 
Analisar se a estratégia de pairs trading pode ser aplicada ao mercado acionário brasileiro a partir de dados históricos da B3. A estratégia consiste em identificar ações que se moveram juntas no passado e verificar se, quando esses pares apresentam um afastamento anormal entre seus preços, o spread tende a retornar ao seu padrão histórico. A partir disso, o estudo busca avaliar se essa normalização pode gerar oportunidade de lucro por meio da compra da ação relativamente barata e da venda da ação relativamente cara, apostando na convergência posterior. Por fim, pretende-se testar se o comportamento histórico dos pares pode ser transformado em uma regra quantitativa de decisão para monitorar futuras oportunidades de entrada e saída no mercado brasileiro. 

## Objetivos específicos 
1) Identificar pares estatisticamente semelhantes na B3
2) Identificar a lógica econômica que relaciona esses ativos 
3) Construir o spread e analisar a partir de qual z-score o desvio deixa de ser uma oscilação normal e passa a representar uma oportunidade de pairs trading
4) Definir uma regra de entrada e saída da operação
5) Medir o tempo e grau de normalização desse spread
6) Avaliar o desempenho financeiro e o risco da estratégia
7) Analisar a diferença do desempenho entre os setores da B3
8) Construção de uma regra de monitoramento futuro. Quando um par validado atinge determinado nível de z-score, o modelo gera um sinal de entrada na operação, ou seja, indicação de compra e venda. Quando o spread retorna para perto da média, gera um sinal de saída da operação, ou seja, vender a ação que havia sido comprada e recomprar a ação que havia sido vendida para obter lucro com essa estratégia. 

## Metodologia

O estudo cobre o período de janeiro de 2015 a dezembro de 2025, utilizando dados históricos de preços ajustados da Economatica/B3. O pipeline foi implementado do zero em Python e executa as seguintes etapas, repetidas para cada janela anual:

**1. Seleção do universo:** para cada ano, selecionam-se os 100 ativos com maior volume financeiro que tenham negociado em pelo menos 60% dos pregões disponíveis naquele período. Isso garante que o modelo opere sempre sobre ativos com liquidez real, incorporando naturalmente IPOs e excluindo papéis que perderam relevância ao longo do tempo.

**2. Pré-triagem por correlação:** calcula-se a correlação de Pearson entre os log-preços de todos os pares possíveis (4.950 combinações por janela). São mantidos apenas os pares com |ρ| ≥ 0,80. A correlação sobre log-preços é a métrica correta para pré-selecionar candidatos à cointegração, pois esta é uma propriedade dos níveis das séries e não dos retornos.

**3. Teste de cointegração de Engle-Granger:** para cada par que passou no filtro de correlação, estima-se a regressão OLS log(A) = α + β·log(B) + ε e aplica-se o teste ADF sobre o resíduo. Pares com p-value ADF < 0,05 são considerados cointegrados — ou seja, apresentam spread estacionário que reverte à média.

**4. Spread e z-score:** para cada par cointegrado, calcula-se o spread diário (Spread_t = log(A_t) − α − β·log(B_t)) e normaliza-se pelo histórico da janela de formação: Z_t = (Spread_t − μ) / σ. O z-score é o sinal operacional: |Z| ≥ 2 indica entrada, |Z| ≤ 0,5 indica saída, e |Z| ≥ 3 aciona o stop de divergência.

A documentação técnica completa, incluindo decisões metodológicas detalhadas, justificativas de cada escolha e pseudocódigo do backtesting, está disponível em [`Documentação modelos e dados.md`](Documentação%20modelos%20e%20dados.md).

## Resultados obtidos

O pipeline identificou **1.749 registros par-ano** e **1.559 pares únicos cointegrados** ao longo das 11 janelas anuais, de um total de 9.709 candidatos por correlação testados. A taxa de aprovação no teste ADF variou de 6% (2024) a 26% (2020), refletindo a maior ou menor coerência estrutural do mercado em cada período.

**Pares mais estáveis estruturalmente** (cointegrados em 3 ou mais janelas): PETR3×PETR4, ITSA4×ITUB4 e IGTA3×MULT3 figuram entre os mais persistentes, com p-values mínimos abaixo de 0,001. Esses pares possuem vínculos econômicos sólidos — mesma empresa em classes diferentes, ou relação holding-controlada — o que explica a estabilidade estatística ao longo do tempo.

**Efeitos de regime:** 2020 (COVID-19) registrou o maior número de pares cointegrados (313), resultado do choque sistêmico que comprimiu os ativos em movimentos conjuntos. Esse período requer atenção especial no backtesting, pois a cointegração observada pode refletir correlação de crise e não um vínculo estrutural de longo prazo. Em contraste, 2024 apresentou apenas 78 pares cointegrados, sinalizando um mercado mais disperso e com relações de equilíbrio mais difíceis de estabelecer.

**Snapshot operacional:** no último pregão disponível (dezembro de 2025), 148 pares apresentavam z-score calculado; destes, 18 exibiam sinal ativo (|Z| ≥ 2), com 12 SHORT e 6 LONG. Os pares com maior z-score naquele momento estavam acima do limiar de stop (|Z| > 3), indicando divergência e não uma oportunidade de entrada.

Os resultados detalhados por janela, ranking de pares e alertas operacionais estão documentados na seção 6 de [`Documentação modelos e dados.md`](Documentação%20modelos%20e%20dados.md).
