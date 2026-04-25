import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from xgboost import XGBClassifier
from preprocess import load_and_clean, split_features
import mlflow
import mlflow.sklearn
import pickle

# Load and prepare data
df = load_and_clean("data/churn.csv")
X, y = split_features(df)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model with MLflow tracking
with mlflow.start_run():
    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42
    )
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, proba)
    
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 4)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("auc_roc", auc)
    mlflow.sklearn.log_model(model, "churn_model")
    
    print(f"Accuracy : {acc:.4f}")
    print(f"AUC-ROC  : {auc:.4f}")
    print(classification_report(y_test, preds))

# Save model locally
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Model saved to model.pkl")
