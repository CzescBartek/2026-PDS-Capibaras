import joblib
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import (confusion_matrix, roc_auc_score, roc_curve, 
                             ConfusionMatrixDisplay, accuracy_score, 
                             recall_score, precision_score, f1_score)

def evaluate_model(X_test_path, y_test_path, model_path):
    # 1. Load artifacts
    classifier = joblib.load(model_path)
    X_test_scaled = np.asarray(joblib.load(X_test_path))
    y_test = joblib.load(y_test_path)
    
    # 2. Generate Predictions
    y_pred = classifier.predict(X_test_scaled)
    testprobs = classifier.predict_proba(X_test_scaled)[:, 1]
    
    # 3. Calculate Metrics
    final_auc = roc_auc_score(y_test, testprobs)
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # 4. Print results to console (matches training script)
    print("\n" + "—"*30)
    print(f"• Accuracy:  {acc:.4f}")
    print(f"• Recall:    {rec:.4f}")
    print(f"• AUC:       {final_auc:.4f}")
    print(f"• Precision: {prec:.4f}")
    print(f"• F1 Score:  {f1:.4f}")
    print("—"*30)

    # 5. Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-cancer", "Cancer"])
    disp.plot(ax=ax1, cmap='Blues')
    ax1.set_title('Confusion Matrix')

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, testprobs)
    ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {final_auc:.4f})')
    ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('Receiver Operating Characteristic (ROC)')
    ax2.legend(loc="lower right")
    ax2.grid(True)

    # 6. Save and Show
    # Determine figure directory relative to model_path or use hardcoded
    figures_dir = '../result/figures'
    os.makedirs(figures_dir, exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'RandomForest_CM_ROC_Eval.png'), dpi=300, bbox_inches='tight') 
    plt.show()

if __name__ == "__main__":
    # Example usage matching your paths
    evaluate_model(
        X_test_path = '../result/models/X_test.pkl',
        y_test_path = '../result/models/Y_test.pkl',
        model_path  = '../result/models/RandomForest_Model.pkl'
    )