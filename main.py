"""
Hallucination Detection and Correction in LLMs - Healthcare Domain
Author: NLP Course Project
Description: Comprehensive system for detecting and correcting hallucinations in medical LLM outputs
"""

import numpy as np
import json
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Import the medical dataset
from medical_dataset import get_dataset, get_dataset_statistics

print("=" * 80)
print("HALLUCINATION DETECTION & CORRECTION IN HEALTHCARE LLMs")
print("=" * 80)


# ============================================================================
# 1. LOAD MEDICAL DATASET
# ============================================================================
print("\n[1] Loading Healthcare Dataset...")

medical_dataset = get_dataset()
dataset_stats = get_dataset_statistics()

print(f"✓ Loaded {dataset_stats['total']} medical cases")
print(f"  - Non-hallucinated: {dataset_stats['factual']}")
print(f"  - Hallucinated: {dataset_stats['hallucinated']}")


# ============================================================================
# 2. DETECTION METHODS
# ============================================================================
print("\n[2] Initializing Detection Methods...")

# Method 1: Entailment-Based Detection (NLI)
print("  → Loading NLI model for entailment detection...")
nli_model = pipeline("text-classification", model="facebook/bart-large-mnli", device=-1)

# Method 2: Sentence Similarity (Semantic Coherence)
print("  → Loading sentence similarity model...")
from sentence_transformers import SentenceTransformer, util
similarity_model = SentenceTransformer('all-MiniLM-L6-v2')

# Method 3: Medical Domain Classifier
print("  → Loading domain-specific classifier...")
domain_classifier = pipeline("text-classification", model="cross-encoder/ms-marco-MiniLM-L-6-v2", device=-1)


class HallucinationDetector:
    """Multi-method hallucination detection system"""
    
    def __init__(self, nli_model, similarity_model, domain_classifier):
        self.nli_model = nli_model
        self.similarity_model = similarity_model
        self.domain_classifier = domain_classifier
        
    def detect_via_entailment(self, evidence, output):
        """Method 1: NLI-based detection - checks if evidence entails output"""
        # Proper NLI format: premise (evidence) entails hypothesis (output)
        input_text = f"{evidence}</s></s>{output}"
        result = self.nli_model(input_text)[0]
        
        # Check if the relationship is entailment or contradiction
        label = result['label'].lower()
        score = result['score']
        
        print(f"    NLI: {label} ({score:.3f})")
        
        # Entailment = factual, Contradiction = hallucination
        if 'entailment' in label and score > 0.5:
            return 0, score  # Not hallucinated
        elif 'contradiction' in label and score > 0.5:
            return 1, score  # Hallucinated
        else:  # neutral or low confidence
            return 1, score  # Flag as potential hallucination
    
    def detect_via_similarity(self, evidence, output):
        """Method 2: Semantic similarity - low similarity indicates hallucination"""
        emb1 = self.similarity_model.encode(evidence, convert_to_tensor=True)
        emb2 = self.similarity_model.encode(output, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(emb1, emb2).item()
        
        print(f"    Similarity: {similarity:.3f}")
        
        # Threshold: similarity < 0.5 suggests hallucination
        is_hallucination = 1 if similarity < 0.5 else 0
        return is_hallucination, similarity
    
    def detect_via_domain_classifier(self, evidence, output):
        """Method 3: Cross-encoder relevance scoring"""
        input_text = f"{evidence} [SEP] {output}"
        result = self.domain_classifier(input_text)[0]
        score = result['score']
        
        print(f"    Domain: {score:.3f}")
        
        # Low relevance score indicates hallucination
        is_hallucination = 1 if score < 0.3 else 0
        return is_hallucination, score
    
    def ensemble_detection(self, evidence, output, weights=[0.5, 0.3, 0.2]):
        """Ensemble method combining all three detectors"""
        print(f"  Detection scores:")
        pred1, score1 = self.detect_via_entailment(evidence, output)
        pred2, score2 = self.detect_via_similarity(evidence, output)
        pred3, score3 = self.detect_via_domain_classifier(evidence, output)
        
        # Weighted voting - sum weights of methods that predict hallucination
        hallucination_weight = 0
        hallucination_weight += weights[0] if pred1 == 1 else 0
        hallucination_weight += weights[1] if pred2 == 1 else 0
        hallucination_weight += weights[2] if pred3 == 1 else 0
        
        # Decision: if majority of weighted votes say hallucination
        final_pred = 1 if hallucination_weight >= 0.4 else 0
        confidence = hallucination_weight if final_pred == 1 else (1 - hallucination_weight)
        
        print(f"  → Final: {'HALLUCINATION' if final_pred == 1 else 'FACTUAL'} (confidence: {confidence:.3f})")
        
        return final_pred, confidence, {
            'entailment': (pred1, score1),
            'similarity': (pred2, score2),
            'domain': (pred3, score3)
        }

detector = HallucinationDetector(nli_model, similarity_model, domain_classifier)
print("✓ Detection methods initialized")


# ============================================================================
# 3. CORRECTION STRATEGIES
# ============================================================================
print("\n[3] Setting up Correction Strategies...")

class HallucinationCorrector:
    """Multiple correction strategies for detected hallucinations"""
    
    def __init__(self, dataset_evidence):
        self.evidence_db = dataset_evidence
        
    def rag_correction(self, query, evidence):
        """Strategy 1: Retrieval-Augmented Generation"""
        corrected = f"Based on medical evidence: {evidence}"
        return {
            'method': 'RAG',
            'corrected_output': corrected,
            'explanation': 'Response grounded in verified medical literature'
        }
    
    def rule_based_correction(self, llm_output, evidence):
        """Strategy 2: Domain-specific rule application"""
        dangerous_patterns = [
            ('cure', 'treatment options include'),
            ('definitely', 'may'),
            ('never', 'typically not recommended'),
            ('always', 'generally'),
            ('all', 'some')
        ]
        
        corrected = llm_output
        applied_rules = []
        
        for wrong, correct in dangerous_patterns:
            if wrong in corrected.lower():
                corrected = corrected.replace(wrong, correct)
                applied_rules.append(f"Replaced '{wrong}' with '{correct}'")
        
        return {
            'method': 'Rule-Based',
            'corrected_output': corrected,
            'applied_rules': applied_rules,
            'explanation': 'Applied medical safety rules to reduce absolute claims'
        }
    
    def explanation_feedback(self, query, llm_output, evidence):
        """Strategy 3: Explanatory correction with reasoning"""
        return {
            'method': 'Explanation Feedback',
            'issue': 'The output contradicts established medical evidence',
            'correct_information': evidence,
            'recommendation': f"Consult healthcare provider regarding: {query}",
            'warning': '⚠️ This response contains medical misinformation'
        }
    
    def human_in_loop_template(self, query, llm_output, evidence):
        """Strategy 4: Human-in-the-loop correction template"""
        return {
            'method': 'Human-in-Loop',
            'flagged_output': llm_output,
            'evidence_provided': evidence,
            'action_required': 'Medical expert review needed',
            'risk_level': 'HIGH' if any(word in llm_output.lower() 
                          for word in ['cure', 'never', 'always', 'definitely']) else 'MEDIUM'
        }

corrector = HallucinationCorrector([d['evidence'] for d in medical_dataset])
print("✓ Correction strategies ready")


# ============================================================================
# 4. EVALUATION PIPELINE
# ============================================================================
print("\n[4] Running Detection & Evaluation...")
print("-" * 80)

results = []
all_predictions = []
all_labels = []

for item in medical_dataset:
    print(f"\nCase {item['id']}: {item['query'][:60]}...")
    # Run ensemble detection
    prediction, confidence, method_scores = detector.ensemble_detection(
        item['evidence'], 
        item['llm_output']
    )
    
    all_predictions.append(prediction)
    all_labels.append(item['label'])
    
    # Determine correction if hallucination detected
    correction = None
    if prediction == 1:
        correction = {
            'rag': corrector.rag_correction(item['query'], item['evidence']),
            'rule': corrector.rule_based_correction(item['llm_output'], item['evidence']),
            'explanation': corrector.explanation_feedback(item['query'], item['llm_output'], item['evidence']),
            'human_loop': corrector.human_in_loop_template(item['query'], item['llm_output'], item['evidence'])
        }
    
    results.append({
        'id': item['id'],
        'query': item['query'],
        'prediction': prediction,
        'actual': item['label'],
        'confidence': confidence,
        'method_scores': method_scores,
        'correction': correction,
        'category': item['category']
    })

# Calculate metrics
accuracy = accuracy_score(all_labels, all_predictions)
precision = precision_score(all_labels, all_predictions)
recall = recall_score(all_labels, all_predictions)
f1 = f1_score(all_labels, all_predictions)
cm = confusion_matrix(all_labels, all_predictions)

print("\n" + "=" * 80)
print("EVALUATION RESULTS")
print("=" * 80)
print(f"\nOverall Metrics:")
print(f"  Accuracy:  {accuracy:.3f}")
print(f"  Precision: {precision:.3f}")
print(f"  Recall:    {recall:.3f}")
print(f"  F1-Score:  {f1:.3f}")

print(f"\nConfusion Matrix:")
print(f"                 Predicted")
print(f"               Non-H  Hall")
print(f"  Actual Non-H    {cm[0][0]:3d}   {cm[0][1]:3d}")
print(f"         Hall     {cm[1][0]:3d}   {cm[1][1]:3d}")

tn, fp, fn, tp = cm.ravel()
print(f"\nDetailed Breakdown:")
print(f"  True Positives (Correctly detected hallucinations):  {tp}")
print(f"  True Negatives (Correctly identified non-halluc.):   {tn}")
print(f"  False Positives (False alarms):                      {fp}")
print(f"  False Negatives (Missed hallucinations):             {fn}")


# ============================================================================
# 5. DETAILED CASE ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("SAMPLE CASE ANALYSIS")
print("=" * 80)

# Show 3 examples: correct detection, false positive, false negative
for i, res in enumerate(results[:3]):
    print(f"\n--- Case {res['id']}: {res['category']} ---")
    print(f"Query: {res['query'][:80]}...")
    print(f"Actual Label: {'HALLUCINATION' if res['actual'] == 1 else 'FACTUAL'}")
    print(f"Predicted: {'HALLUCINATION' if res['prediction'] == 1 else 'FACTUAL'}")
    print(f"Confidence: {res['confidence']:.3f}")
    print(f"Detection Verdict: {'✓ CORRECT' if res['prediction'] == res['actual'] else '✗ INCORRECT'}")
    
    if res['prediction'] == 1 and res['correction']:
        print(f"\nCorrection Applied (RAG Method):")
        print(f"  {res['correction']['rag']['corrected_output'][:100]}...")


# ============================================================================
# 6. SAVE RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

# Save detailed results
output_data = {
    'metrics': {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist()
    },
    'results': results
}

with open('detection_results.json', 'w') as f:
    json.dump(output_data, f, indent=2, default=str)

print("✓ Results saved to detection_results.json")

# Save evaluation report
with open('evaluation_report.txt', 'w') as f:
    f.write("HALLUCINATION DETECTION & CORRECTION - EVALUATION REPORT\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Dataset Size: {len(medical_dataset)}\n")
    f.write(f"Accuracy: {accuracy:.3f}\n")
    f.write(f"Precision: {precision:.3f}\n")
    f.write(f"Recall: {recall:.3f}\n")
    f.write(f"F1-Score: {f1:.3f}\n\n")
    f.write(f"Confusion Matrix:\n{cm}\n\n")
    f.write(f"True Positives: {tp}\n")
    f.write(f"True Negatives: {tn}\n")
    f.write(f"False Positives: {fp}\n")
    f.write(f"False Negatives: {fn}\n")

print("✓ Report saved to evaluation_report.txt")

print("\n" + "=" * 80)
print("EXECUTION COMPLETE")
print("=" * 80)
print("\nNext Steps:")
print("  1. Review detection_results.json for detailed analysis")
print("  2. Read evaluation_report.txt for summary metrics")
print("  3. Check final_report.md for comprehensive documentation")
print("=" * 80)