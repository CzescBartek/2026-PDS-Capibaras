import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, ConfusionMatrixDisplay
import joblib

def RandomForest_model_reduced(features_path):
    # --- 1. Data Loading and Preprocessing ---
    df = pd.read_csv(features_path).dropna(axis=0)
    df = df.drop(['img_id'], axis=1)

    # Specific feature subset
    selected_features = [
        'FEATURE_A', 'FEATURE_B_H', 'FEATURE_B_S', 'FEATURE_B_V', 
        'FEATURE_BORDER_COMPACTNESS', 'FEATURE_BORDER_CONVEXITY', 'Hair'
    ]
    X = df[selected_features]
    y = df['Cancerous']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1907, stratify=y
    )

    feature_names = X.columns.tolist()
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    kfold = KFold(n_splits=5, random_state=1907, shuffle=True)

    # --- 2. CV PLOT 1: Impact of Max Depth ---
    base_n = 100
    depth_range = range(1, 21)
    depth_means = []
    depth_stds = []

    print("Analyzing Max Depth (Subset Features)...")
    for d in depth_range:
        fold_aucs = []
        for train_idx, val_idx in kfold.split(X_train_scaled):
            rf = RandomForestClassifier(n_estimators=base_n, max_depth=d, random_state=1907, n_jobs=-1)
            rf.fit(X_train_scaled[train_idx], np.ravel(y_train.iloc[train_idx]))
            probs = rf.predict_proba(X_train_scaled[val_idx])[:, 1]
            fold_aucs.append(roc_auc_score(y_train.iloc[val_idx], probs))
        depth_means.append(np.mean(fold_aucs))
        depth_stds.append(np.std(fold_aucs))

    best_d_from_graph = depth_range[np.argmax(depth_means)]
    
    plt.figure(figsize=(10, 6))
    plt.errorbar(depth_range, depth_means, yerr=depth_stds, fmt='-o', color='#1f77b4', label='Mean AUC ±1 std')
    plt.title(f'Optimal Max Depth: {best_d_from_graph} (Subset)')
    plt.xlabel('max_depth')
    plt.ylabel('Mean ROC AUC')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('../result/figures/CV_Max_Depth_reduced.png', dpi=300)
    plt.show()

    # --- 3. CV PLOT 2: Impact of N Estimators ---
    n_test_range = [10, 50, 100, 200, 300, 400, 500, 600]
    n_means = []
    n_stds = []

    print(f"Analyzing N_Estimators with depth={best_d_from_graph} (Subset Features)...")
    for n in n_test_range:
        fold_aucs = []
        for train_idx, val_idx in kfold.split(X_train_scaled):
            rf = RandomForestClassifier(n_estimators=n, max_depth=best_d_from_graph, random_state=1907, n_jobs=-1)
            rf.fit(X_train_scaled[train_idx], np.ravel(y_train.iloc[train_idx]))
            probs = rf.predict_proba(X_train_scaled[val_idx])[:, 1]
            fold_aucs.append(roc_auc_score(y_train.iloc[val_idx], probs))
        n_means.append(np.mean(fold_aucs))
        n_stds.append(np.std(fold_aucs))

    best_n_from_graph = n_test_range[np.argmax(n_means)]

    plt.figure(figsize=(10, 6))
    plt.errorbar(n_test_range, n_means, yerr=n_stds, fmt='-s', color='#d62728', label='Mean AUC ±1 std')
    plt.title(f'Optimal N Estimators: {best_n_from_graph} (Subset)')
    plt.xlabel('n_estimators')
    plt.ylabel('Mean ROC AUC')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('../result/figures/CV_N_Estimators_reduced.png', dpi=300)
    plt.show()

    # --- 4. Final Model Training ---
    print(f"\nFinal training: n_estimators={best_n_from_graph}, max_depth={best_d_from_graph}")
    
    classifier = RandomForestClassifier(
        n_estimators=best_n_from_graph,
        max_depth=best_d_from_graph,
        random_state=1907,
        oob_score=True,
        n_jobs=-1
    )

    classifier.fit(X_train_scaled, y_train)
    print("Out-of-Bag Score:", classifier.oob_score_)

    # --- 5. Exporting ---
    joblib.dump(classifier, '../result/models/RandomForest_Model_reduced.pkl')
    joblib.dump(X_test_scaled, '../result/models/X_test_reduced.pkl') 
    joblib.dump(feature_names, '../result/models/feature_names_reduced.pkl') 
    joblib.dump(y_test, '../result/models/Y_test_reduced.pkl')

    # --- 6. Final Evaluation Plots ---
    y_pred = classifier.predict(X_test_scaled)
    testprobs = classifier.predict_proba(X_test_scaled)[:, 1]
    final_auc = roc_auc_score(y_test, testprobs)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax1, cmap='Blues')
    ax1.set_title('Confusion Matrix (Subset)')

    fpr, tpr, _ = roc_curve(y_test, testprobs)
    ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {final_auc:.4f})')
    ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curve (Subset)')
    ax2.legend(loc="lower right")
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('../result/figures/RandomForest_CM_ROC_reduced.png', dpi=300) 
    plt.show()

if __name__ == "__main__":
    RandomForest_model_reduced('../data/features.csv')