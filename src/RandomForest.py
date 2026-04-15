import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
df= pd.read_csv('../data/features.csv')
encoders={}
for col in df.select_dtypes(include=['object']).columns:
    le= LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col]=le    

df.info()
X = df.iloc[:,:-2].values
y = df.iloc[:,-2].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=1907
)

classifier=RandomForestClassifier(
    n_estimators=100,
    random_state=1907,
    oob_score=True
)

classifier.fit(X_train, y_train)
print("Out-of-Bag Score:", classifier.oob_score_)

y_pred = classifier.predict(X_test)

print("\nClassification Report:\n",classification_report(y_test,y_pred))
sns.heatmap(confusion_matrix(y_test,y_pred), annot=True,fmt='d')
plt.title("Confusion Matrix")
plt.show()
