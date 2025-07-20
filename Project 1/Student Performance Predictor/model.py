

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
data = {
    'Hours_Studied': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Result':        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  # 0 = Fail, 1 = Pass
}

df = pd.DataFrame(data)
plt.scatter(df['Hours_Studied'], df['Result'], color='blue')
plt.title('Hours Studied vs Result')
plt.xlabel('Hours Studied')
plt.ylabel('Pass (1) / Fail (0)')
plt.grid(True)
plt.savefig("hours_vs_result.png") 
plt.show()
X = df[['Hours_Studied']]
y = df['Result']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

hours = 6.5
pred = model.predict([[hours]])
print(f"\nPrediction for {hours} hours studied: {'Pass' if pred[0] == 1 else 'Fail'}")
