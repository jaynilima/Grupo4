import requests
from pathlib import Path
from tqdm import tqdm


PASTA_PROJETO = Path(__file__).resolve().parents[1]
PASTA_DADOS = PASTA_PROJETO / "dados"

ANO_INICIAL = 2010
ANO_FINAL = 2025


def baixar_cotahist_anual(ano):
    PASTA_DADOS.mkdir(parents=True, exist_ok=True)

    nome_arquivo = f"COTAHIST_A{ano}.ZIP"
    caminho_destino = PASTA_DADOS / nome_arquivo

    if caminho_destino.exists():
        print(f"Arquivo já existe: {nome_arquivo}")
        return

    url = f"https://bvmf.bmfbovespa.com.br/InstDados/SerHist/{nome_arquivo}"

    resposta = requests.get(url, stream=True, timeout=60)

    if resposta.status_code != 200:
        print(f"Erro ao baixar {nome_arquivo}. Status: {resposta.status_code}")
        return

    tamanho_total = int(resposta.headers.get("content-length", 0))

    with open(caminho_destino, "wb") as arquivo:
        for bloco in tqdm(
            resposta.iter_content(chunk_size=8192),
            total=max(tamanho_total // 8192, 1),
            desc=f"Baixando {nome_arquivo}"
        ):
            if bloco:
                arquivo.write(bloco)

    print(f"Download concluído: {nome_arquivo}")


def main():
    for ano in range(ANO_INICIAL, ANO_FINAL + 1):
        baixar_cotahist_anual(ano)


if __name__ == "__main__":
    main()