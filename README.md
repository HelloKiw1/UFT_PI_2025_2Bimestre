# Processamento de Imagens — Trabalhos Práticos

Repositório com implementações para a disciplina "Processamento de Imagens" (2º Bimestre — 2025/2).

Conteúdo implementado:

- Trabalho 1 — Morfologia Matemática
  - Erosão e Dilatação (imagens em tons de cinza)
  - Transformada Top-Hat
- Trabalho 2 — Processamento de Imagens Coloridas
  - Segmentação por limiarização (método do vale)

Como rodar o menu interativo

1. Criar um ambiente virtual e instalar dependências:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; python -m pip install -r requirements.txt
```

2. Rodar o menu interativo:

```powershell
python src\main.py
```

Observação sobre imagens de entrada

As imagens que você deve usar para os trabalhos estão na pasta `Data/` do repositório. Ao executar o menu interativo você pode:

- Digitar o caminho completo para uma imagem dentro de `Data`, por exemplo:

```powershell
Data\imagens\minha_imagem.png
```

- Ou copiar o caminho relativo `Data/` e colar quando o menu pedir o caminho da imagem.

Se preferir, também é possível mover as imagens para outro diretório e fornecer o caminho correspondente ao menu.

O menu permitirá carregar uma imagem (caminho no disco) ou usar uma imagem de exemplo gerada automaticamente. Os resultados podem ser visualizados (se houver GUI) e salvos em disco.

Observações

- As implementações usam OpenCV e NumPy.
- O método do vale encontra um limiar a partir do histograma suavizado; se não encontrar dois picos, recorre ao método de Otsu.
- Testes básicos com pytest estão em `tests/test_basic.py`.
