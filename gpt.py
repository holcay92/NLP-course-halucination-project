"""
Hallucination Detection & Correction in LLMs
Healthcare Domain
--------------------------------------------------
Covers:
- Dataset creation & labeling
- Uncertainty-based detection
- Classifier-based detection
- Retrieval-augmented verification
- Rule-based medical safety checks
- Ensemble detection
- Evaluation metrics
"""

import pandas as pd
import numpy as np
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)

# =========================================================
# 1. DATASET CREATION (SIMULATED HEALTHCARE LLM OUTPUTS)
# =========================================================

data = [
    {
        "prompt": "Is metformin safe during pregnancy?",
        "response": "Metformin is commonly prescribed during pregnancy and is considered safe.",
        "label": 0
    },
    {
        "prompt": "What is the maximum daily dose of paracetamol for adults?",
        "response": "Adults can safely take up to 6000 mg of paracetamol per day.",
        "label": 1
    },
    {
        "prompt": "Can antibiotics treat viral infections?",
        "response": "Antibiotics are effective treatments for viral infections like influenza.",
        "label": 1
    },
    {
        "prompt": "What is the first-line treatment for hypertension?",
        "response": "Lifestyle changes and ACE inhibitors are commonly used as first-line treatment.",
        "label": 0
    },
    {
        "prompt": "Is insulin used for type 1 diabetes?",
        "response": "Insulin is required for managing type 1 diabetes.",
        "label": 0
    },
    {
        "prompt": "Is ibuprofen safe in the third trimester of pregnancy?",
        "response": "Ibuprofen is completely safe throughout pregnancy with no risks.",
        "label": 1
    }
]

df = pd.DataFrame(data)

# =========================================================
# 2. UNCERTAINTY-BASED DETECTION
# =========================================================
# Proxy for log-probability / entropy using risky phrases

RISK_PHRASES = [
    "always", "never", "guaranteed", "completely safe",
    "no risk", "all patients", "100%", "no side effects"
]

def uncertainty_score(text):
    score = 0
    for phrase in RISK_PHRASES:
        if re.search(rf"\b{phrase}\b", text.lower()):
            score += 1
    return score

df["uncertainty_score"] = df["response"].apply(uncertainty_score)
df["uncertainty_flag"] = (df["uncertainty_score"] > 0).astype(int)

# =========================================================
# 3. RULE-BASED MEDICAL SAFETY CHECKS
# =========================================================

def medical_rule_check(prompt, response):
    p = prompt.lower()
    r = response.lower()

    # Paracetamol dosage rule
    if "paracetamol" in p and "6000" in r:
        return 1

    # Antibiotics for viruses rule
    if "antibiotics" in p and "viral" in r and "effective" in r:
        return 1

    # Pregnancy NSAID rule
    if "pregnancy" in p and "ibuprofen" in r and "safe" in r:
        return 1

    return 0

df["rule_based_flag"] = df.apply(
    lambda x: medical_rule_check(x["prompt"], x["response"]),
    axis=1
)

# =========================================================
# 4. RETRIEVAL-AUGMENTED VERIFICATION (SIMULATED)
# =========================================================
# Trusted medical knowledge (WHO / FDA-like)

TRUSTED_KB = {
    "paracetamol": "Maximum recommended dose is 4000 mg per day",
    "antibiotics": "Antibiotics do not treat viral infections",
    "ibuprofen_pregnancy": "NSAIDs should be avoided in the third trimester"
}

def retrieval_verification(prompt, response):
    p = prompt.lower()
    r = response.lower()

    if "paracetamol" in p:
        return "4000" in r

    if "antibiotics" in p:
        return "do not" in r or "not effective" in r

    if "ibuprofen" in p and "pregnancy" in p:
        return "avoid" in r or "risk" in r

    return True

df["retrieval_verified"] = df.apply(
    lambda x: retrieval_verification(x["prompt"], x["response"]),
    axis=1
)
df["retrieval_flag"] = (~df["retrieval_verified"]).astype(int)

# =========================================================
# 5. CLASSIFIER-BASED HALLUCINATION DETECTION
# =========================================================

X_text = df["prompt"] + " " + df["response"]
y = df["label"]

vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(X_text)

classifier = LogisticRegression()
classifier.fit(X, y)

df["classifier_pred"] = classifier.predict(X)

# =========================================================
# 6. ENSEMBLE HALLUCINATION DETECTOR
# =========================================================

df["detected_hallucination"] = (
    (df["uncertainty_flag"] == 1) |
    (df["rule_based_flag"] == 1) |
    (df["retrieval_flag"] == 1) |
    (df["classifier_pred"] == 1)
).astype(int)

# =========================================================
# 7. EVALUATION
# =========================================================

y_true = df["label"]
y_pred = df["detected_hallucination"]

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(
    y_true,
    y_pred,
    target_names=["Non-Hallucinated", "Hallucinated"]
))

print("\n=== CONFUSION MATRIX ===")
print(confusion_matrix(y_true, y_pred))

print("\n=== METRICS ===")
print("Accuracy :", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred))
print("Recall   :", recall_score(y_true, y_pred))

# =========================================================
# 8. OUTPUT FINAL DATASET
# =========================================================

print("\n=== FINAL DATASET ===")
print(df[[
    "prompt",
    "response",
    "label",
    "detected_hallucination",
    "uncertainty_flag",
    "rule_based_flag",
    "retrieval_flag",
    "classifier_pred"
]])
