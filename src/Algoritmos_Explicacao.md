## Morfologia (arquivo: `src/morphology.py`)

- Erosão
  - O que faz: reduz regiões claras (foreground) e elimina pequenas protuberâncias; afina objetos.
  - Como faz: para cada pixel substitui seu valor pelo mínimo encontrado numa janela quadrada ao redor (min-filter). Bordas são tratadas com padding. Repetir a operação quantas vezes for especificado.

- Dilatação
  - O que faz: amplia regiões claras, conecta componentes próximos e preenche pequenos buracos.
  - Como faz: para cada pixel substitui pelo máximo na janela quadrada (max-filter). Também permite várias iterações.

- Top-hat (white top-hat)
  - O que faz: realça detalhes brilhantes menores que o elemento estrutural (pequenos pontos ou manchas claras).
  - Como faz: calcula a abertura (erosão seguida de dilatação) e subtrai essa abertura da imagem original; o resultado mantém apenas as estruturas que foram removidas pela abertura.

## Segmentação por limiar (arquivo: `src/segmentation.py`)

- Suavização do histograma
  - O que faz: reduz ruído no histograma de intensidades para detectar picos reais.
  - Como faz: convolui o histograma (256 bins) com um kernel Gaussiano definido pelo parâmetro `sigma`.

- Método do vale (valley)
  - O que faz: encontra um limiar entre duas classes de intensidade identificadas como dois picos no histograma.
  - Como faz: detecta picos locais no histograma suavizado; se houver ao menos dois, escolhe os dois maiores picos e retorna o índice do ponto mínimo (vale) entre eles como limiar.
  - Observação: se não houver dois picos bem definidos, usa-se Otsu como fallback.

- Otsu (fallback)
  - O que faz: escolhe o limiar que maximiza a separação entre classes (variância entre-classes).
  - Como faz: calcula o histograma, as probabilidades acumuladas e as médias cumulativas; avalia a variância entre-classes para cada limiar possível e retorna o melhor.

- Aplicação em imagens coloridas
  - O que faz: segmenta a imagem colorida em primeiro plano/fundo usando o limiar encontrado sobre a intensidade (luminosidade).
  - Como faz: converte a imagem para intensidade (combinação linear dos canais), obtém o limiar via método do vale (ou Otsu), cria máscara binária (pixels > limiar) e aplica essa máscara à imagem colorida (zera o fundo). Há uma heurística que inverte a máscara se quase toda a imagem ficar marcada como primeiro plano.



