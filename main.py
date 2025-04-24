import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load dataset
df = pd.read_csv("data/penguins_size.csv")
df.dropna(inplace=True)

# Encode categorical variables
le = LabelEncoder()
df['species'] = le.fit_transform(df['species'])
df['sex'] = le.fit_transform(df['sex'])
df['island'] = le.fit_transform(df['island'])

# Features and target
X = df.drop("species", axis=1)
y = df["species"]

# Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(n_neighbors=3),
    "SVM": SVC(probability=True),
    "Random Forest": RandomForestClassifier()
}

# Training and evaluation
results = []
os.makedirs("images", exist_ok=True)
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average='weighted'),
        "Recall": recall_score(y_test, y_pred, average='weighted'),
        "F1 Score": f1_score(y_test, y_pred, average='weighted')
    })
    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.savefig(f'images/confusion_matrix_{name.replace(" ", "_")}.png')
    plt.close()

# Decision Tree feature importance
dt_importance = models['Decision Tree'].feature_importances_
plt.figure()
sns.barplot(x=dt_importance, y=X.columns)
plt.title("Feature Importance - Decision Tree")
plt.tight_layout()
plt.savefig("images/feature_importance_Decision_Tree.png")
plt.close()

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv("results.csv", index=False)
