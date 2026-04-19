import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sentence_transformers import SentenceTransformer
import joblib

import numpy as np
from scipy.sparse import hstack, csr_matrix
import re


import os

DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/processed/processed_data.csv"))


def load_data():
    df = pd.read_csv(DATA_PATH)
    if len(df) > 300000:
        df = df.sample(n=300000, random_state=42)
    print("Dataset shape:", df.shape)
    print("\nClass distribution:\n")
    print(df["difficulty"].value_counts())
    return df


def create_features(df, use_embeddings=False):
    if use_embeddings:
        print("Using Sentence Embeddings...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        X_embeddings = model.encode(df["question"].tolist(), show_progress_bar=True)
        scaler = StandardScaler()
        X_numeric = scaler.fit_transform(
            df[["score", "question_length", "tag_count"]]
        )
        X = hstack([csr_matrix(X_embeddings), csr_matrix(X_numeric)])
        return X, df["difficulty"], model

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), stop_words="english", max_features=50000, min_df=5
    )

    X_tfidf = vectorizer.fit_transform(df["question"])
    scaler = StandardScaler()

    X_numeric = scaler.fit_transform(
        df[["score", "question_length", "tag_count"]]
    )
    
    X = hstack([X_tfidf, X_numeric])
    y = df["difficulty"]

    print("\nFeature matrix shape:", X.shape)

    return X, y, vectorizer


if __name__ == "__main__":
    df = load_data()
    df = df.dropna(subset=["question"])
    df["question"] = df["question"].astype(str)
    df["question_length"] = df["question"].apply(len)
    df["tag_count"] = df["tags"].apply(lambda x: len(re.findall(r'<[^>]+>', str(x))) if pd.notna(x) else 0)

    X, y, vectorizer = create_features(df)

    # print("This is X:",X)
    # print("This is y:",y)

    print("This is vectorizer:", vectorizer)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=69, stratify=y
    )

    # 1. Logistic Regression
    print("\n--- Training Logistic Regression ---")
    lr_model = LogisticRegression(max_iter=5000, solver="saga", class_weight="balanced")
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)

    # 2. Decision Tree
    print("\n--- Training Decision Tree ---")
    dt_model = DecisionTreeClassifier(max_depth=20, class_weight="balanced", random_state=42)
    dt_model.fit(X_train, y_train)
    y_pred_dt = dt_model.predict(X_test)

    # Comparison Report
    print("\n" + "="*30)
    print("MODEL PERFORMANCE COMPARISON")
    print("="*30)

    print("\n[Logistic Regression]")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred_lr))

    print("\n[Decision Tree]")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_dt):.4f}")
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred_dt))

    print("\n--- Confusion Matrix (Decision Tree) ---")
    print(confusion_matrix(y_test, y_pred_dt))

    # Save Models
    MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models"))
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    print("\n--- Saving Models ---")
    if isinstance(vectorizer, TfidfVectorizer):
        joblib.dump(vectorizer, os.path.join(MODELS_DIR, "vectorizer.joblib"))
    
    joblib.dump(lr_model, os.path.join(MODELS_DIR, "lr_model.joblib"))
    joblib.dump(dt_model, os.path.join(MODELS_DIR, "dt_model.joblib"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))
    
    print(f"Models saved successfully to: {MODELS_DIR}")
