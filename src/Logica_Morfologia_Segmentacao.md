# Lógica do Código: Morfologia e Segmentação

Este documento explica, de forma direta e ligada ao código, como a implementação do repositório realiza as operações de morfologia (erosão, dilatação, top-hat) e a segmentação por limiar (método do vale + Otsu). Todo o texto está em português e referencia as funções nos arquivos `src/morphology.py` e `src/segmentation.py`.

## Contrato (inputs / outputs)
- Morfologia (em `src/morphology.py`)
  - Funções: `erosion(image, kernel_size=3, iterations=1)`, `dilation(image, kernel_size=3, iterations=1)`, `top_hat(image, kernel_size=15)`.
  - Entrada: imagem em tons de cinza (NumPy 2D, dtype uint8).
  - Saída: imagem processada (mesma forma, uint8).

- Segmentação (em `src/segmentation.py`)
  - Funções principais: `valley_threshold_gray(gray, sigma=2.0)`, `otsu_threshold(gray)`, `segment_color_by_valley(img_color, sigma=2.0, return_mask=False)`.
  - Entrada: imagem colorida (NumPy 3D, uint8) para `segment_color_by_valley`; internamente converte para grayscale.
  - Saída: imagem colorida com fundo zerado (e opcionalmente máscara binária 0/255).

## Morfologia — como o código implementa (passo a passo)

1) Erosão (`erosion`):
   - Lógica: a função percorre cada pixel e calcula o mínimo dos valores dentro de uma janela quadrada centrada no pixel. Esse mínimo substitui o valor original.
   - No código: a entrada é acolhida em `img`; faz-se `np.pad(img, pad, mode='constant', constant_values=255)` para tratar bordas. Depois, para cada (i,j) a janela é obtida com `padded[i:i+kernel_size, j:j+kernel_size]` e `window.min()` é atribuído a `out[i,j]`. O laço pode repetir `iterations` vezes.
   - Efeito prático: objetos claros afinam e pequenos pontos brilhantes desaparecem.

2) Dilatação (`dilation`):
   - Lógica: análoga à erosão, mas usando máximo em vez de mínimo. Para cada pixel, toma-se o máximo na janela e atribui.
   - No código: usa `np.pad(..., constant_values=0)` e `window.max()` dentro dos loops aninhados; repete conforme `iterations`.
   - Efeito prático: regiões claras aumentam, pequenos buracos são preenchidos e componentes próximos se conectam.

3) Top-hat (`top_hat`):
   - Lógica: top-hat branco = imagem original - abertura(original). A abertura é erosão seguida de dilatação com o mesmo elemento estrutural.
   - No código: chama `opened = erosion(image, kernel_size)` e `opened = dilation(opened, kernel_size)`. Em seguida subtrai `opened` da `image` em `int16` para evitar underflow, aplica `np.clip(..., 0, 255)` e converte para `uint8`.
   - Efeito prático: mantém pequenas estruturas brilhantes que foram removidas pela abertura.

Observação sobre padding: o valor usado no pad influencia bordas; a implementação escolhe 255 para erosão e 0 para dilatação — comportamento que favorece preservar a lógica didática usada no trabalho.

## Segmentação — como o código implementa (passo a passo)

1) Conversão para intensidade:
   - A função `segment_color_by_valley` calcula intensidade luminosa por combinação linear: `gray = 0.114*B + 0.587*G + 0.299*R` (cada canal convertido para float antes).

2) Histograma e suavização:
   - `valley_threshold_gray` obtém histograma de 256 bins com `np.histogram(gray.ravel(), bins=256, range=(0,255))`.
   - Suaviza esse histograma com `_smooth_histogram` que constrói um kernel Gaussiano (tamanho proporcional a `sigma`) e aplica `np.convolve(hist, gauss, mode='same')`.

3) Detecção de picos e escolha do limiar (método do vale):
   - O código varre o histograma suavizado `sh` e coleta índices `i` onde `sh[i] > sh[i-1] and sh[i] > sh[i+1]` (picos locais).
   - Se há pelo menos dois picos, selecionam-se os dois de maior amplitude (ordenação por `sh[p]`) e busca-se o índice do valor mínimo entre esses dois picos; esse índice é o limiar retornado.
   - Se não há pelo menos dois picos (histograma unimodal ou ruidoso), o código cai no fallback `otsu_threshold`.

4) Fallback: Otsu (`otsu_threshold`):
   - Calcula probabilidades dos níveis de intensidade (hist/total), as somas cumulativas `omega` e as médias cumulativas `mu`.
   - Para cada limiar t, computa a variância entre-classes sigma_b^2 e escolhe o t que maximiza essa métrica (`np.nanargmax`).

5) Criação da máscara e aplicação:
   - Com o limiar `th` obtido, cria-se `mask = (gray > th).astype(np.uint8) * 255` (foreground = mais claros).
   - Heurística: se a máscara cobre >85% dos pixels, o código inverte para `mask = (gray <= th)` (preserva cenário onde o background é claro e o foreground é escuro).
   - Aplica-se a máscara sobre a imagem colorida com `segmented[mask == 0] = 0` (zera o fundo).