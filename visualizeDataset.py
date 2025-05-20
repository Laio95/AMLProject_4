import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image

def visualizza_maschera(num_img, base_path=r'..\dataset\train\urban', alpha=0.5):
    """
    Visualizza l'immagine con la maschera sovrapposta.

    Args:
        num_img (int): Numero dell'immagine (es. 1 per '1.png').
        base_path (str): Percorso base della cartella 'rural'.
        alpha (float): Trasparenza della maschera.
    """
    # Costruisci i percorsi
    img_path = os.path.join(base_path, 'images_png', f'{num_img}.png')
    mask_path = os.path.join(base_path, 'masks_png', f'{num_img}.png')
    
    # Carica immagine e maschera
    img = np.array(Image.open(img_path).convert('RGB')) / 255.0
    mask = np.array(Image.open(mask_path))
    
    # Colori per classi 0-5
    colors = [
        (0, 0, 0, 0),         # Classe 0 - trasparente  none
        (0.5, 0.5, 0.5, alpha),     # Classe 1 - ciano        background
        (1, 0, 0, alpha),     # Classe 2 - rosso        building
        (1, 1, 0, alpha),     # Classe 3 - giallo       road  
        (0, 0, 1, alpha),     # Classe 4 - blu          water 
        (1, 0, 1, alpha),     # Classe 5 - magenta      barren
        (0, 1, 0, alpha),     # Classe 6 - verde        forest
        (0.3, 0.5, 0.3, alpha),   # Classe 7 - arancione    agricolture
    ]
    cmap = ListedColormap(colors)

    # Visualizzazione affiancata
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))

    axs[0].imshow(img)
    axs[0].set_title(f'Immagine {num_img} originale')
    axs[0].axis('off')

    axs[1].imshow(img)
    axs[1].imshow(mask, cmap=cmap, vmin=0, vmax=7, interpolation='nearest')
    axs[1].set_title(f'Immagine {num_img} con maschera')
    axs[1].axis('off')

    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Visualizza un'immagine con maschera sovrapposta.")
    parser.add_argument("image_number", type=int, help="Numero dell'immagine da visualizzare (es. 1 per '1.png')")
    args = parser.parse_args()

    visualizza_maschera(args.image_number)

if __name__ == "__main__":
    main()