"""
Visualization and Analysis Script for Hallucination Detection Results
Run this after main.py to generate visual insights
"""

import json
import numpy as np
from collections import Counter

print("=" * 80)
print("VISUALIZATION & ANALYSIS OF DETECTION RESULTS")
print("=" * 80)

# Load results
try:
    with open('detection_results.json', 'r') as f:
        data = json.load(f)
    
    results = data['results']
    metrics = data['metrics']
    
    print("\n[1] Loading Results...")
    print(f"✓ Loaded {len(results)} cases")
    
    # Extract data for analysis
    categories = [r['category'] for r in results]
    predictions = [r['prediction'] for r in results]
    actuals = [r['actual'] for r in results]
    confidences = [r['confidence'] for r in results]
    
    # Category distribution
    print("\n[2] Category Distribution:")
    category_counts = Counter(categories)
    for cat, count in category_counts.most_common():
        print(f"  {cat}: {count}")
    
    # Correct vs Incorrect by Category
    print("\n[3] Detection Accuracy by Category:")
    category_accuracy = {}
    for cat in set(categories):
        cat_results = [(r['prediction'], r['actual']) for r in results if r['category'] == cat]
        correct = sum(1 for pred, actual in cat_results if pred == actual)
        total = len(cat_results)
        accuracy = correct / total if total > 0 else 0
        category_accuracy[cat] = accuracy
        status = "✓" if accuracy >= 0.8 else "⚠"
        print(f"  {status} {cat}: {accuracy:.1%} ({correct}/{total})")
    
    # Confidence distribution
    print("\n[4] Confidence Analysis:")
    high_conf = sum(1 for c in confidences if c >= 0.8)
    med_conf = sum(1 for c in confidences if 0.5 <= c < 0.8)
    low_conf = sum(1 for c in confidences if c < 0.5)
    print(f"  High confidence (≥0.8): {high_conf}")
    print(f"  Medium confidence (0.5-0.8): {med_conf}")
    print(f"  Low confidence (<0.5): {low_conf}")
    
    # False positives and negatives analysis
    print("\n[5] Error Analysis:")
    false_positives = [r for r in results if r['prediction'] == 1 and r['actual'] == 0]
    false_negatives = [r for r in results if r['prediction'] == 0 and r['actual'] == 1]
    
    if false_positives:
        print(f"\n  False Positives ({len(false_positives)}):")
        for fp in false_positives:
            print(f"    - Case {fp['id']}: {fp['query'][:60]}...")
            print(f"      Category: {fp['category']}")
            print(f"      Confidence: {fp['confidence']:.3f}")
    
    if false_negatives:
        print(f"\n  False Negatives ({len(false_negatives)}):")
        for fn in false_negatives:
            print(f"    - Case {fn['id']}: {fn['query'][:60]}...")
            print(f"      Category: {fn['category']}")
            print(f"      Confidence: {fn['confidence']:.3f}")
    else:
        print("\n  ✓ No False Negatives - All hallucinations detected!")
    
    # Detection method contribution
    print("\n[6] Detection Method Contribution:")
    entailment_correct = 0
    similarity_correct = 0
    domain_correct = 0
    
    for r in results:
        method_scores = r['method_scores']
        ent_pred = method_scores['entailment'][0]
        sim_pred = method_scores['similarity'][0]
        dom_pred = method_scores['domain'][0]
        
        if ent_pred == r['actual']:
            entailment_correct += 1
        if sim_pred == r['actual']:
            similarity_correct += 1
        if dom_pred == r['actual']:
            domain_correct += 1
    
    total = len(results)
    print(f"  NLI Entailment: {entailment_correct}/{total} ({entailment_correct/total:.1%})")
    print(f"  Semantic Similarity: {similarity_correct}/{total} ({similarity_correct/total:.1%})")
    print(f"  Domain Classifier: {domain_correct}/{total} ({domain_correct/total:.1%})")
    
    # Recommendations
    print("\n[7] Recommendations:")
    if metrics['recall'] < 1.0:
        print("  ⚠ Improve recall to catch all hallucinations (patient safety critical)")
    else:
        print("  ✓ Perfect recall achieved - all hallucinations caught")
    
    if metrics['precision'] < 0.9:
        print(f"  ⚠ {int((1-metrics['precision'])*100)}% false positive rate - consider tuning thresholds")
    
    if len(results) < 50:
        print("  ⚠ Small dataset - expand to 100+ cases for robust evaluation")
    
    print("\n[8] Key Findings:")
    print(f"  • Best performing category: {max(category_accuracy, key=category_accuracy.get)}")
    print(f"  • Most challenging category: {min(category_accuracy, key=category_accuracy.get)}")
    print(f"  • Average confidence: {np.mean(confidences):.3f}")
    print(f"  • System bias: {'Conservative (flags more)' if metrics['precision'] < 0.9 else 'Balanced'}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nFor detailed results, see detection_results.json")
    print("For full report, see final_report.md")
    
except FileNotFoundError:
    print("\n❌ Error: detection_results.json not found")
    print("Please run main.py first to generate results")
except Exception as e:
    print(f"\n❌ Error: {e}")
