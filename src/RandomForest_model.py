import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score,roc_curve
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

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

y_pred = classifier.predict(X_test)



print("Accuracy:", accuracy_score(y_test,y_pred))
print("\nConfusion Matrix:\n",confusion_matrix(y_test,y_pred))
print("\nClassification Report:\n",classification_report(y_test,y_pred))
sns.heatmap(confusion_matrix(y_test,y_pred), annot=True,fmt='d')
plt.title("Confusion Matrix")
plt.show()


kfold= KFold(n_splits=5, random_state=1907,shuffle=True)

cv_scores=cross_val_score(classifier, X_train,np.ravel(y_train),cv=5,scoring='roc_auc')

testprobs = classifier.predict_proba(X_test)[:,1]

print(roc_auc_score(y_test,testprobs))
print(np.mean(cv_scores))

fpr, tpr, _ = roc_curve(y_test, testprobs)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_score(y_test,testprobs):.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.legend(loc="lower right")

plt.tight_layout()
plt.show()
