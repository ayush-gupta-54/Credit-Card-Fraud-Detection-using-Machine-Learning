import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline


# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_csv("creditcard.csv")

print("Dataset Loaded Successfully")
print("Shape:", df.shape)
print("\nClass Distribution:")
print(df["Class"].value_counts())


# -----------------------------
# 2. Split Features and Target
# -----------------------------
X = df.drop("Class", axis=1)
y = df["Class"]


# -----------------------------
# 3. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -----------------------------
# 4. Fast Models
# -----------------------------
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=500,
            class_weight="balanced",
            random_state=42
        ))
    ]),

    "Random Forest": Pipeline([
        ("model", RandomForestClassifier(
            n_estimators=30,
            max_depth=10,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        ))
    ])
}


# -----------------------------
# 5. Train and Evaluate
# -----------------------------
best_model = None
best_score = 0
best_model_name = ""

for name, model in models.items():
    print("\n" + "=" * 50)
    print("Training:", name)
    print("=" * 50)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    print("Accuracy:", accuracy)
    print("ROC-AUC:", roc_auc)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    if roc_auc > best_score:
        best_score = roc_auc
        best_model = model
        best_model_name = name


# -----------------------------
# 6. Save Best Model
# -----------------------------
joblib.dump(best_model, "fraud_detection_model.pkl")

print("\nBest Model:", best_model_name)
print("Best ROC-AUC:", best_score)
print("Model saved as fraud_detection_model.pkl")