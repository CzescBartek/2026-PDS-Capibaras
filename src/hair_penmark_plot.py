import cv2
import matplotlib.pyplot as plt
import numpy as np
from hair_coverage import hair_coverage
from hair_remove import removeHair_auto
from penmark_remove import remove_penmarks

# --- 1. SET YOUR FILE PATHS ---
path_hair = '../data/imgs/PAT_279_429_142.png' 
path_pen = '../data/imgs/PAT_361_1567_487.png' 

def process_save_and_plot():
    # Load Images
    img_hair_bgr = cv2.imread(path_hair)
    img_pen_bgr = cv2.imread(path_pen)

    if img_hair_bgr is None or img_pen_bgr is None:
        print("Error: One of the images could not be loaded. Check your paths.")
        return

    # --- 2. PROCESS & SAVE HAIR REMOVAL ---
    gray_hair = cv2.cvtColor(img_hair_bgr, cv2.COLOR_BGR2GRAY)
    out_hair_bgr = removeHair_auto(img_hair_bgr, gray_hair)
    
    # Save files
    cv2.imwrite('../result/figures/original_hair.png', img_hair_bgr)
    cv2.imwrite('../result/figures/result_no_hair.png', out_hair_bgr)

    # --- 3. PROCESS & SAVE PENMARK REMOVAL ---
    out_pen_bgr = remove_penmarks(img_pen_bgr)
    
    # Save files
    cv2.imwrite('../result/figures/original_pen.png', img_pen_bgr)
    cv2.imwrite('../result/figures/result_no_pen.png', out_pen_bgr)

    # --- 4. OPTIONAL: DISPLAY IN NOTEBOOK ---
    # Convert to RGB for visualization only
    img_hair_rgb = cv2.cvtColor(img_hair_bgr, cv2.COLOR_BGR2RGB)
    out_hair_rgb = cv2.cvtColor(out_hair_bgr, cv2.COLOR_BGR2RGB)
    img_pen_rgb = cv2.cvtColor(img_pen_bgr, cv2.COLOR_BGR2RGB)
    out_pen_rgb = cv2.cvtColor(out_pen_bgr, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes[0, 0].imshow(img_hair_rgb); axes[0, 0].set_title("Original (Hair)"); axes[0, 0].axis('off')
    axes[0, 1].imshow(out_hair_rgb); axes[0, 1].set_title("Result (Hair Removed)"); axes[0, 1].axis('off')
    axes[1, 0].imshow(img_pen_rgb); axes[1, 0].set_title("Original (Penmarks)"); axes[1, 0].axis('off')
    axes[1, 1].imshow(out_pen_rgb); axes[1, 1].set_title("Result (Penmarks Removed)"); axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("Success: 4 images saved to your current directory.")

process_save_and_plot()