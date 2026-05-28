import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

# 1. Setup
df = pd.read_csv('../data/features.csv')
features_to_plot = ["FEATURE_A", "FEATURE_BORDER_COMPACTNESS", "FEATURE_BORDER_CONVEXITY", 'FEATURE_B_H']
output_folder = "../result/figures"


# 2. Data Cleaning
df['Cancerous'] = df['Cancerous'].astype(str)

for f in features_to_plot:
    lower, upper = df[f].quantile([0.01, 0.99])
    df = df[(df[f] >= lower) & (df[f] <= upper)]

scaler = StandardScaler()
df[features_to_plot] = scaler.fit_transform(df[features_to_plot])

# 3. Plotting Loop (Splits into 6 files)
for feature in features_to_plot:
    clean_name = feature.lower().replace(' ', '_')
    
    # --- FILE 1: KDE PLOT ---
    plt.figure(figsize=(10, 6))
    ax_kde = sns.kdeplot(
        data=df, x=feature, hue="Cancerous", fill=True, 
        common_norm=False, palette=["blue", "red"], 
        hue_order=["0", "1"], alpha=0.5, linewidth=2, bw_adjust=1.5
    )
    plt.title(f"Distribution Profile: {feature}", fontsize=14, fontweight='bold')
    plt.xlim(-3, 3)
    
    # Fix KDE Legend
    legend = ax_kde.get_legend()
    if legend:
        legend.set_title("Status")
        for t, l in zip(legend.get_texts(), ["Non-cancer", "Cancer"]):
            t.set_text(l)
            
    plt.savefig(f"{output_folder}/{clean_name}_kde.png", dpi=300)
    plt.close()
    print(f"Saved: {clean_name}_kde.png")

    # --- FILE 2: BOX PLOT ---
    plt.figure(figsize=(6, 6))
    # Fix for FutureWarning: Assign x to hue and set legend=False
    ax_box = sns.boxplot(
        data=df, x="Cancerous", y=feature, hue="Cancerous",
        order=["0", "1"], palette=["blue", "red"], 
        legend=False, width=0.5
    )
    
    # Fix for UserWarning: Set ticks before setting labels
    ax_box.set_xticks([0, 1])
    ax_box.set_xticklabels(['Non-cancer', 'Cancer'])
    
    plt.title(f"Statistical Ranges: {feature}", fontsize=12)
    plt.ylim(-3, 3)
    
    plt.savefig(f"{output_folder}/{clean_name}_boxplot.png", dpi=300)
    plt.close()
    print(f"Saved: {clean_name}_boxplot.png")

print("\nDone! 6 files generated in ../result/figures")