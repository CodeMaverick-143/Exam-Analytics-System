#!/usr/bin/env python3
"""
Generate visualizations for the Exam Analytics Pro paper.
Outputs:
  - class_distribution.png: Class distribution countplot
  - confusion_matrix_lr.png: LR confusion matrix heatmap
  - confusion_matrix_dt.png: DT confusion matrix heatmap
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from scipy.sparse import hstack

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['font.size'] = 11

# Load data
print("Loading data...")
DATA_PATH = 'data/processed/processed_data.csv'
df = pd.read_csv(DATA_PATH)
if len(df) > 300000:
    df = df.sample(n=300000, random_state=42)

df = df.dropna(subset=['question'])
df['question'] = df['question'].astype(str)
df['question_length'] = df['question'].apply(len)
df['tag_count'] = df['tags'].apply(lambda x: len(str(x).split(',')) if pd.notna(x) else 0)

# Load models
print("Loading models...")
MODELS_DIR = 'models'
vectorizer = joblib.load(os.path.join(MODELS_DIR, "vectorizer.joblib"))
lr_model = joblib.load(os.path.join(MODELS_DIR, "lr_model.joblib"))
dt_model = joblib.load(os.path.join(MODELS_DIR, "dt_model.joblib"))

# Create features
print("Creating features...")
X_tfidf = vectorizer.transform(df['question'])
scaler = StandardScaler()
X_numeric = scaler.fit_transform(df[['score', 'question_length', 'tag_count']])
X = hstack([X_tfidf, X_numeric])
y = df['difficulty']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=69, stratify=y
)

# Get predictions
y_pred_lr = lr_model.predict(X_test)
y_pred_dt = dt_model.predict(X_test)

# --- Figure 1: Class Distribution ---
print("Generating class distribution plot...")
fig, ax = plt.subplots(figsize=(8, 5))
class_counts = y_test.value_counts().sort_index()
colors = ['#2ecc71', '#f39c12', '#e74c3c']
class_counts.plot(kind='bar', ax=ax, color=colors, edgecolor='black', linewidth=1.5)
ax.set_title('Test Set Class Distribution', fontsize=14, fontweight='bold')
ax.set_xlabel('Difficulty Class', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
print("Saved: class_distribution.png")
plt.close()

# --- Figure 2: LR Confusion Matrix ---
print("Generating LR confusion matrix heatmap...")
cm_lr = confusion_matrix(y_test, y_pred_lr, labels=['Easy', 'Medium', 'Hard'])
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=['Easy', 'Medium', 'Hard'],
            yticklabels=['Easy', 'Medium', 'Hard'],
            ax=ax, linewidths=1.5, linecolor='black')
ax.set_title('Logistic Regression Confusion Matrix', fontsize=14, fontweight='bold')
ax.set_xlabel('Predicted', fontsize=12)
ax.set_ylabel('True', fontsize=12)
plt.tight_layout()
plt.savefig('confusion_matrix_lr.png', dpi=300, bbox_inches='tight')
print("Saved: confusion_matrix_lr.png")
plt.close()

# --- Figure 3: DT Confusion Matrix ---
print("Generating DT confusion matrix heatmap...")
cm_dt = confusion_matrix(y_test, y_pred_dt, labels=['Easy', 'Medium', 'Hard'])
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Oranges', cbar=True,
            xticklabels=['Easy', 'Medium', 'Hard'],
            yticklabels=['Easy', 'Medium', 'Hard'],
            ax=ax, linewidths=1.5, linecolor='black')
ax.set_title('Decision Tree Confusion Matrix', fontsize=14, fontweight='bold')
ax.set_xlabel('Predicted', fontsize=12)
ax.set_ylabel('True', fontsize=12)
plt.tight_layout()
plt.savefig('confusion_matrix_dt.png', dpi=300, bbox_inches='tight')
print("Saved: confusion_matrix_dt.png")
plt.close()

print("\nAll visualizations generated successfully!")
print("Files created:")
print("  - class_distribution.png")
print("  - confusion_matrix_lr.png")
print("  - confusion_matrix_dt.png")
