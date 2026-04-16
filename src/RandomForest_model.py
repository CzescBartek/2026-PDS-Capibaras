import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score,roc_curve, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
import joblib

df= pd.read_csv('../data/features.csv')
df=df.dropna(axis=0)
df=df.drop(['img_id'], axis=1)

X = df.iloc[:,:-2].values
y = df.iloc[:,-2].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=1907
)
scaler = StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
classifier=RandomForestClassifier(
    n_estimators=100,
    random_state=1907,
    oob_score=True
)


classifier.fit(X_train, y_train)
print("Out-of-Bag Score:", classifier.oob_score_)



kfold= KFold(n_splits=5, random_state=1907,shuffle=True)
cv_scores=cross_val_score(classifier, X_train,np.ravel(y_train),cv=5,scoring='roc_auc')
print(np.mean(cv_scores))


joblib.dump(classifier, '../result/models/RandomForest_Model.pkl')

y_pred = classifier.predict(X_test)
testprobs = classifier.predict_proba(X_test)[:,1]
final_auc = roc_auc_score(y_test, testprobs)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
cm = confusion_matrix(y_test,y_pred)
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

plt.savefig('../result/figures/RandomForest_CM_ROC', dpi=300, bbox_inches='tight') 
plt.show()
plt.close() 