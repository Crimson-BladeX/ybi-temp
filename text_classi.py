import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Data
df = pd.read_csv("3346b5c1-62e8-4271-b16b-fb5d1ee2c4c1.csv")

# 2. Preprocessing
df = df[['description', 'medical_specialty']].dropna()
df['description'] = df['description'].astype(str)
df['medical_specialty'] = df['medical_specialty'].astype(str)

# 3. Split Data
X_train, X_test, y_train, y_test = train_test_split(
    df['description'], df['medical_specialty'], test_size=0.2, random_state=42, stratify=df['medical_specialty']
)

# 4. Feature Extraction (TF-IDF)
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 5. Train Model (Logistic Regression)
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# 6. Evaluate Model
y_pred = model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 7. Confusion Matrix (Optional Visualization)
from sklearn.metrics import confusion_matrix
import numpy as np

plt.figure(figsize=(12, 8))
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
sns.heatmap(cm, annot=False, fmt='d', cmap='Blues',
            xticklabels=model.classes_,
            yticklabels=model.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()
