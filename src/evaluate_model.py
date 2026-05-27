def evaluate_model(model, X_test_scaled, y_test, features_names):

    testprobs = classifier.predict_proba(X_test_scaled)[:, 1]
    final_auc = roc_auc_score(y_test, testprobs)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax1, cmap='Blues')
    ax1.set_title('Confusion Matrix')

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

    plt.tight_layout()
    plt.savefig('../result/figures/RandomForest_CM_ROC.png', dpi=300, bbox_inches='tight') 
    plt.show()

