import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, roc_auc_score, roc_curve, 
                             ConfusionMatrixDisplay, accuracy_score, 
                             recall_score, precision_score, f1_score)

def RandomForest_model(features_path, prediction_results_path, model_path, feature_names_path, X_test_path, y_test_path):
    # 1. Data Loading and Preparation
    df = pd.read_csv(features_path).dropna(axis=0)
    if 'img_id' in df.columns:
        df = df.drop(['img_id'], axis=1)

    X = df.drop(['Cancerous'], axis=1)
    y = df['Cancerous']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1907, stratify=y
    )

    feature_names = X.columns.tolist()
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Prepare directories for results based on provided paths
    figures_dir = os.path.join(os.path.dirname(model_path), '../figures')
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(os.path.dirname(prediction_results_path), exist_ok=True)

    kfold = KFold(n_splits=5, random_state=1907, shuffle=True)

    # 2. Hyperparameter Tuning: Max Depth
    base_n = 100
    depth_range = range(1, 21)
    depth_means = []
    depth_stds = []

    print("Analyzing Max Depth...")
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
    plt.errorbar(depth_range, depth_means, yerr=depth_stds, fmt='-o', label='Mean AUC ±1 std')
    plt.title(f'Cross-Validation: Max Depth (Optimal: {best_d_from_graph})')
    plt.xlabel('max_depth')
    plt.ylabel('Mean ROC AUC')
    plt.grid(True)
    plt.savefig(os.path.join(figures_dir, 'CV_Max_Depth_Plot.png'))
    plt.show()

    # 3. Hyperparameter Tuning: N Estimators
    n_test_range = [10, 50, 100, 200, 300, 400, 500, 600]
    n_means = []
    n_stds = []

    print(f"Analyzing N_Estimators with depth={best_d_from_graph}...")
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
    plt.errorbar(n_test_range, n_means, yerr=n_stds, fmt='-s', color='red', label='Mean AUC ±1 std')
    plt.title(f'Cross-Validation: N Estimators (Optimal: {best_n_from_graph})')
    plt.xlabel('n_estimators')
    plt.ylabel('Mean ROC AUC')
    plt.grid(True)
    plt.savefig(os.path.join(figures_dir, 'CV_N_Estimators_Plot.png'))
    plt.show()

    # 4. Final Model Training
    print(f"Final training: n_estimators={best_n_from_graph}, max_depth={best_d_from_graph}")
    classifier = RandomForestClassifier(
        n_estimators=best_n_from_graph,
        max_depth=best_d_from_graph,
        random_state=0,
        oob_score=True,
        n_jobs=-1
    )
    classifier.fit(X_train_scaled, y_train)

    # 5. Saving Artifacts using provided variable paths
    joblib.dump(classifier, model_path)
    joblib.dump(X_test_scaled, X_test_path) 
    joblib.dump(feature_names, feature_names_path) 
    joblib.dump(y_test, y_test_path)

    # 6. Evaluation
    y_pred = classifier.predict(X_test_scaled)
    testprobs = classifier.predict_proba(X_test_scaled)[:, 1]
    
    # Save predictions to CSV
    pd.DataFrame({'Actual': y_test, 'Predicted': y_pred, 'Probability': testprobs}).to_csv(prediction_results_path, index=False)

    final_auc = roc_auc_score(y_test, testprobs)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax1, cmap='Blues')
    ax1.set_title('Confusion Matrix')

    fpr, tpr, _ = roc_curve(y_test, testprobs)
    ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {final_auc:.4f})')
    ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('Receiver Operating Characteristic')
    ax2.legend(loc="lower right")
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'RandomForest_CM_ROC.png'), dpi=300) 
    plt.show()
    
    # Final Metrics Output
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n" + "—"*30)
    print(f"• Accuracy:  {acc:.4f}")
    print(f"• Recall:    {rec:.4f}")
    print(f"• AUC:       {final_auc:.4f}")
    print(f"• Precision: {prec:.4f}")
    print(f"• F1 Score:  {f1:.4f}")
    print("—"*30)

if __name__ == "__main__":
    RandomForest_model(
        features_path = "../data/features.csv",
        prediction_results_path = "../result/predictions/predictions.csv",
        model_path = "../result/models/RandomForest_Model.pkl",
        feature_names_path = '../result/models/feature_names.pkl',
        X_test_path = '../result/models/X_test.pkl',
        y_test_path = '../result/models/Y_test.pkl',


    )