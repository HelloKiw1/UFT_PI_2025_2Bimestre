import os
import sys
import cv2
import numpy as np
from pathlib import Path

from morphology import erosion, dilation, top_hat
from segmentation import segment_color_by_valley

# Repository layout helpers
# REPO_ROOT is the repository root (parent of src)
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / 'Data'
# Save outputs inside Data/saida so they stay with the dataset
SAIDA_DIR = DATA_DIR / 'saida'
SAIDA_DIR.mkdir(parents=True, exist_ok=True)


def read_image_grayscale(path: Path | str):
    p = Path(path)
    if not p.is_absolute():
        p = DATA_DIR / p
    if not p.exists():
        raise FileNotFoundError(f"Could not load image: {p}")
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image as grayscale: {p}")
    return img


def read_image_color(path: Path | str):
    p = Path(path)
    if not p.is_absolute():
        p = DATA_DIR / p
    if not p.exists():
        raise FileNotFoundError(f"Could not load image: {p}")
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image as color: {p}")
    return img


def sample_gray_image():
    # generate synthetic grayscale sample
    img = np.zeros((200, 300), dtype=np.uint8)
    cv2.rectangle(img, (30, 30), (120, 170), 200, -1)
    cv2.circle(img, (200, 100), 40, 150, -1)
    cv2.rectangle(img, (150, 30), (260, 90), 80, -1)
    return img


def sample_color_image():
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.rectangle(img, (10, 10), (150, 200), (0, 128, 255), -1)  # orange-ish
    cv2.circle(img, (240, 120), 60, (0, 255, 0), -1)  # green
    cv2.rectangle(img, (180, 20), (310, 80), (255, 0, 0), -1)  # blue
    return img


def show_and_maybe_save(img, title='result'):
    # try to show (works on desktop). Also ask to save.
    try:
        cv2.imshow(title, img)
        cv2.waitKey(1)
    except Exception:
        pass

    save = input('Salvar resultado em arquivo? (s/n): ').strip().lower()
    if save == 's':
        out = input('Nome de saída (ex: out.png) [enter para padrão]: ').strip()
        if not out:
            out = f"{title}.png"
        out_path = SAIDA_DIR / out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), img)
        print(f"Salvo: {out_path}")


def menu_morphology():
    print('\n-- Morfologia Matemática --')
    choice = input('Usar imagem de exemplo sintética? (s/n): ').strip().lower()
    if choice == 's':
        img = sample_gray_image()
    else:
        prompt = f"Caminho da imagem em tons de cinza (relativo a '{DATA_DIR.name}'; enter para padrão 'flor.png'): "
        path_in = input(prompt).strip()
        if not path_in:
            path = DATA_DIR / 'flor.png'
        else:
            p = Path(path_in)
            path = p if p.is_absolute() else DATA_DIR / p
        try:
            img = read_image_grayscale(path)
        except FileNotFoundError as e:
            print(e)
            print('Usando imagem sintetica de fallback.')
            img = sample_gray_image()

    print('Operações:\n1) Erosão\n2) Dilatação\n3) Top-Hat')
    op = input('Escolha (1/2/3): ').strip()
    k = int(input('Tamanho do kernel (impar, ex 3): ').strip() or 3)
    it = 1
    if op in ('1', '2'):
        it = int(input('Iterações (ex 1): ').strip() or 1)

    if op == '1':
        out = erosion(img, kernel_size=k, iterations=it)
        show_and_maybe_save(out, 'erosion')
    elif op == '2':
        out = dilation(img, kernel_size=k, iterations=it)
        show_and_maybe_save(out, 'dilation')
    elif op == '3':
        out = top_hat(img, kernel_size=k)
        show_and_maybe_save(out, 'top_hat')
    else:
        print('Operação inválida')


def menu_segmentation():
    print('\n-- Segmentação de Imagens Coloridas (método do vale) --')
    choice = input('Usar imagem de exemplo sintética? (s/n): ').strip().lower()
    if choice == 's':
        img = sample_color_image()
    else:
        prompt = f"Caminho da imagem colorida (relativo a '{DATA_DIR.name}'; enter para padrão 'flor.png'): "
        path_in = input(prompt).strip()
        if not path_in:
            path = DATA_DIR / 'flor.png'
        else:
            p = Path(path_in)
            path = p if p.is_absolute() else DATA_DIR / p
        try:
            img = read_image_color(path)
        except FileNotFoundError as e:
            print(e)
            print('Usando imagem sintetica de fallback.')
            img = sample_color_image()

    sigma = float(input('Sigma para suavização do histograma (ex 2.0): ').strip() or 2.0)
    segmented, mask = None, None
    try:
        segmented, mask = segment_color_by_valley(img, sigma=sigma, return_mask=True)
    except Exception as e:
        print('Erro na segmentação:', e)
        return

    # show results
    show_and_maybe_save(mask, 'mask')
    show_and_maybe_save(segmented, 'segmented')


def main_loop():
    while True:
        print('\nMenu principal:\n1) Trabalho 1 - Morfologia Matemática\n2) Trabalho 2 - Segmentação (Imagens Coloridas)\n0) Sair')
        c = input('Escolha: ').strip()
        if c == '1':
            menu_morphology()
        elif c == '2':
            menu_segmentation()
        elif c == '0':
            print('Tchau')
            break
        else:
            print('Opção inválida')


if __name__ == '__main__':
    # ensure src is in path so imports work when running from project root
    project_src = Path(__file__).resolve().parent
    repo_root = project_src.parent
    # run with cwd = repository root so Data/ and saida/ are relative to repo
    os.chdir(repo_root)
    sys.path.insert(0, str(project_src))
    main_loop()
