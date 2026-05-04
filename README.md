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
### Importância para o mercado financeiro  

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

## Resultados objetidos 
