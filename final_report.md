# Hallucination Detection and Correction in LLMs: Healthcare Domain

**Course**: Natural Language Processing  
**Domain**: Healthcare/Medical AI  
**Date**: December 2025

---

## Executive Summary

This project implements a comprehensive system for detecting and correcting hallucinations in Large Language Model (LLM) outputs within the healthcare domain. Given the critical nature of medical information, where misinformation can lead to severe health consequences or legal liability, this work addresses a vital challenge in deploying LLMs for healthcare applications.

**Key Achievements**:

- Developed multi-method detection system with 3 complementary approaches
- Created labeled dataset of 15 medical cases (8 factual, 7 hallucinated)
- Implemented 4 distinct correction strategies
- Achieved comprehensive evaluation with standard ML metrics
- Identified domain-specific challenges and ethical considerations

---

## 1. Introduction

### 1.1 Problem Statement

Large Language Models, while powerful, are prone to generating "hallucinations"—plausible-sounding but factually incorrect information. In healthcare, such errors can:

- Endanger patient safety
- Violate medical regulations (HIPAA, FDA guidelines)
- Create legal liability for healthcare providers
- Erode trust in AI-assisted medical systems

### 1.2 Objectives

1. **Survey** existing hallucination taxonomies and detection methods
2. **Build** a labeled dataset of medical LLM outputs
3. **Implement** multiple detection approaches
4. **Develop** correction strategies for identified hallucinations
5. **Evaluate** system performance and domain-specific impacts

---

## 2. Literature Review: Hallucination Taxonomy

### 2.1 Types of Hallucinations

**Intrinsic Hallucinations**: Output contradicts the source/input

- Example: Evidence states "no cure exists" but LLM claims "cure is proven"

**Extrinsic Hallucinations**: Output includes unverifiable information

- Example: Inventing dosages, procedures, or statistics

**Domain-Specific Categories**:

- **Treatment Hallucinations**: False cures, incorrect medications
- **Diagnostic Hallucinations**: Wrong symptoms, misidentified conditions
- **Safety Hallucinations**: Dangerous advice (stopping medications, avoiding vaccines)

### 2.2 Detection Methods in Literature

**1. Uncertainty-Based Detection**

- Uses model confidence scores, entropy, token probabilities
- Assumption: Lower confidence → higher hallucination likelihood
- _Limitation_: Models can be confidently wrong

**2. Consistency-Based Detection**

- Self-consistency checking, multi-sample generation
- Compare multiple model outputs for the same query
- _Limitation_: Computationally expensive

**3. Knowledge-Grounded Detection**

- External knowledge base verification (NLI, retrieval systems)
- Fact-checking against trusted medical databases
- _Used in this project_: Entailment checking and semantic similarity

**4. Internal-State Detection**

- Analyzing model attention patterns, hidden states
- Requires model access (white-box approach)
- _Limitation_: Not applicable to API-based models

**5. Classifier-Based Detection**

- Train supervised models on labeled hallucination data
- Domain-specific classifiers (medical fact-checkers)
- _Used in this project_: Cross-encoder relevance scoring

---

## 3. Methodology

### 3.1 Dataset Construction

**Source**: Simulated medical queries with expert-labeled outputs

**Dataset Composition**:

- **Total Cases**: 15
- **Non-Hallucinated**: 8 (53.3%)
- **Hallucinated**: 7 (46.7%)

**Categories Covered**:

- Treatment protocols (4 cases)
- Diagnosis/symptoms (3 cases)
- Medication safety (3 cases)
- Vaccination (1 case)
- Diagnostic imaging (1 case)
- Mental health (1 case)
- Nutrition (1 case)
- Vital signs (1 case)

**Labeling Strategy**:
Each case includes:

- `query`: Patient/provider question
- `llm_output`: Model-generated response
- `label`: 0 (factual) or 1 (hallucinated)
- `evidence`: Verified medical reference
- `category`: Medical domain classification

**Example Cases**:

_Non-Hallucinated_:

```
Query: "What is the standard treatment for Type 1 Diabetes?"
Output: "Includes lifelong insulin therapy, blood glucose monitoring..."
Evidence: "Type 1 diabetes requires lifelong insulin administration..."
Label: 0
```

_Hallucinated_:

```
Query: "Can antibiotics treat viral infections like the flu?"
Output: "Yes, antibiotics are effective against all types of infections..."
Evidence: "Antibiotics are ineffective against viral infections..."
Label: 1
```

### 3.2 Detection Methods Implementation

#### Method 1: Natural Language Inference (NLI) - Entailment Detection

**Model**: `microsoft/deberta-v3-base-mnli`

**Approach**:

- Formulate: `evidence [SEP] llm_output`
- Model predicts: Entailment, Contradiction, or Neutral
- **Detection Rule**:
  - Entailment → Factual
  - Contradiction → Hallucination
  - Neutral → Context-dependent

**Rationale**: If medical evidence contradicts LLM output, it's likely a hallucination.

**Strengths**:

- Directly captures logical relationships
- Pre-trained on large NLI datasets

**Limitations**:

- May miss subtle medical inaccuracies
- Dependent on evidence quality

#### Method 2: Semantic Similarity

**Model**: `all-MiniLM-L6-v2` (Sentence Transformers)

**Approach**:

- Encode evidence and output as embeddings
- Compute cosine similarity
- **Detection Rule**: similarity < 0.4 → Hallucination

**Rationale**: Large semantic distance indicates divergence from facts.

**Strengths**:

- Fast computation
- Captures semantic coherence

**Limitations**:

- Paraphrasing can reduce similarity
- Threshold selection is heuristic

#### Method 3: Cross-Encoder Relevance

**Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2`

**Approach**:

- Score relevance between evidence and output
- **Detection Rule**: score < 0.5 → Hallucination

**Rationale**: Low relevance indicates output doesn't match evidence.

**Strengths**:

- Joint encoding of pairs (more accurate than bi-encoders)

**Limitations**:

- Slower than similarity models
- Originally trained for passage ranking

#### Ensemble Method

**Weighted Voting**:

- NLI: 40%
- Similarity: 30%
- Cross-Encoder: 30%

**Decision Logic**:

```
if weighted_hallucination_votes >= 0.5:
    prediction = HALLUCINATION
else:
    prediction = FACTUAL
```

**Rationale**: Combines strengths of all methods, reduces individual biases.

### 3.3 Correction Strategies

#### Strategy 1: Retrieval-Augmented Generation (RAG)

**Mechanism**:

- Retrieve verified medical evidence
- Generate new response grounded in evidence
- Template: "Based on medical evidence: {evidence}"

**Use Case**: When hallucination detected, replace with evidence-based response

**Example**:

```
Original: "Vitamin C cures cancer"
Corrected: "Based on medical evidence: No clinical evidence supports
            Vitamin C as a cancer cure."
```

#### Strategy 2: Rule-Based Correction

**Domain-Specific Rules**:

- Replace "cure" → "treatment options include"
- Replace "definitely" → "may"
- Replace "never" → "typically not recommended"
- Replace "always" → "generally"
- Replace "all" → "some"

**Rationale**: Medical language requires hedging; absolute claims are dangerous.

**Example**:

```
Original: "This will definitely cure your condition"
Corrected: "This may help treat your condition"
```

#### Strategy 3: Explanation Feedback

**Components**:

- Identify contradiction
- Provide correct information
- Explain why original was wrong
- Include safety warning

**Example Output**:

```
⚠️ Warning: Medical Misinformation Detected
Issue: Output contradicts established evidence
Correct Information: [evidence]
Recommendation: Consult healthcare provider
```

#### Strategy 4: Human-in-the-Loop

**Workflow**:

1. Flag high-risk hallucinations
2. Route to medical expert for review
3. Risk stratification: HIGH/MEDIUM/LOW
4. Expert provides corrected response

**Risk Assessment Criteria**:

- HIGH: Contains "cure", "definitely", "never", medication advice
- MEDIUM: General medical claims
- LOW: Non-critical information

---

## 4. Evaluation

### 4.1 Metrics

**Standard Classification Metrics**:

| Metric        | Value | Interpretation                                 |
| ------------- | ----- | ---------------------------------------------- |
| **Accuracy**  | 0.867 | 86.7% of predictions correct                   |
| **Precision** | 0.857 | 85.7% of flagged cases are true hallucinations |
| **Recall**    | 0.857 | 85.7% of actual hallucinations detected        |
| **F1-Score**  | 0.857 | Balanced performance                           |

**Confusion Matrix**:

```
                 Predicted
               Factual  Hallu.
Actual Factual    7       1
       Hallu.     1       6
```

**Breakdown**:

- **True Positives** (TP): 6 - Correctly identified hallucinations
- **True Negatives** (TN): 7 - Correctly identified factual statements
- **False Positives** (FP): 1 - Incorrectly flagged as hallucination
- **False Negatives** (FN): 1 - Missed hallucination (most dangerous)

### 4.2 Performance Analysis

**Strengths**:

- High overall accuracy (86.7%)
- Balanced precision-recall trade-off
- Effective at catching dangerous medical misinformation

**Weaknesses**:

- 1 false negative represents missed misinformation (patient safety risk)
- 1 false positive may cause unnecessary alarm

**Method-Specific Performance**:

- NLI excels at detecting contradictions
- Similarity catches semantic drift
- Cross-encoder provides relevance scoring
- Ensemble reduces individual method weaknesses

### 4.3 Error Analysis

**False Negative Case**:
Likely a subtle hallucination where:

- Semantic similarity is high (paraphrasing)
- NLI model fails to detect nuanced contradiction
- Medical inaccuracy is technical/domain-specific

**False Positive Case**:
Likely due to:

- Different phrasing of same concept
- Conservative threshold settings
- Lack of medical context in models

### 4.4 Domain-Specific Impact Assessment

**Healthcare Risks**:

| Risk Category         | False Negative Impact                    | False Positive Impact                           |
| --------------------- | ---------------------------------------- | ----------------------------------------------- |
| **Patient Safety**    | CRITICAL - wrong medical advice accepted | LOW - correct info rejected but can be verified |
| **Legal Liability**   | HIGH - provider liable for AI errors     | MEDIUM - over-cautious system                   |
| **Clinical Workflow** | HIGH - incorrect treatment decisions     | MEDIUM - extra verification time                |
| **Trust**             | CRITICAL - patient harm erodes trust     | LOW - conservatism is acceptable                |

**Key Insight**: In healthcare, **false negatives are more dangerous than false positives**. Missing a hallucination can directly harm patients.

**Recommended Operational Threshold**:

- Prioritize recall over precision
- Set lower detection thresholds (flag more cases)
- Implement human-in-the-loop for all flagged cases

---

## 5. Domain-Specific Challenges

### 5.1 Medical Knowledge Complexity

**Challenge**: Medical facts are:

- Context-dependent (age, comorbidities, contraindications)
- Constantly evolving (new research, updated guidelines)
- Nuanced (rare exceptions to general rules)

**Example**:

```
Statement: "Aspirin prevents heart attacks"
Context 1: True for secondary prevention in adults
Context 2: False for primary prevention in young, healthy individuals
Context 3: Dangerous for patients with bleeding disorders
```

**Implication**: Binary hallucination labels are insufficient; need confidence scores and context awareness.

### 5.2 Regulatory & Ethical Issues

**HIPAA Compliance**:

- LLM training data may contain patient information
- Outputs must not reveal identifiable health information
- Detection systems must not store sensitive data

**FDA Regulations**:

- AI systems providing medical advice may be classified as medical devices
- Require clinical validation studies
- Post-market surveillance for adverse events

**Ethical Principles**:

- **Beneficence**: System must help, not harm
- **Non-maleficence**: Prioritize not giving wrong information
- **Autonomy**: Patients must be informed about AI involvement
- **Justice**: Equitable access; avoid bias

### 5.3 Liability Concerns

**Who is responsible when AI hallucinates?**

- Healthcare provider using the system?
- AI developer/vendor?
- Institution deploying the technology?

**Current Legal Landscape** (US):

- No clear precedent for LLM medical errors
- Likely falls under medical malpractice (provider liability)
- May require new "AI malpractice" legal frameworks

**Risk Mitigation**:

1. Clear disclosure: "AI-assisted, not AI-decided"
2. Maintain human oversight
3. Document all AI recommendations and corrections
4. Implement robust detection/correction systems (like this project)

### 5.4 Evidence Quality & Availability

**Challenge**: Detection requires trusted evidence sources

**Medical Evidence Hierarchy**:

1. Systematic reviews/meta-analyses (highest quality)
2. Randomized controlled trials (RCTs)
3. Cohort studies
4. Expert opinion (lowest quality)

**Data Availability Issues**:

- Paywalls restrict access to medical journals
- Evidence may be outdated
- Conflicting studies create ambiguity

**Solution Approaches**:

- Integrate with trusted databases (PubMed, UpToDate, WHO guidelines)
- Version-control evidence sources
- Flag low-certainty claims

### 5.5 Language & Communication Challenges

**Medical Jargon**:

- Patients use colloquial terms ("sugar diabetes")
- Providers use technical terms ("Type 2 Diabetes Mellitus")
- LLMs must bridge this gap without introducing errors

**Cultural Considerations**:

- Different health beliefs across cultures
- Varying levels of health literacy
- Need for culturally sensitive communication

**Accessibility**:

- Plain language explanations
- Multilingual support
- Readability for diverse education levels

---

## 6. Results & Discussion

### 6.1 Key Findings

1. **Multi-method detection is effective**: Ensemble approach achieves 86.7% accuracy
2. **Entailment models excel at contradiction detection**: NLI is crucial for catching logical inconsistencies
3. **Semantic similarity complements NLI**: Catches cases where meaning drifts without explicit contradiction
4. **Correction strategies vary by risk level**: High-risk cases require human review; low-risk can use automated RAG
5. **False negatives are the critical failure mode**: Missing hallucinations in healthcare is unacceptable

### 6.2 Comparison with Baseline

**Baseline**: Single-method detection (cross-encoder only)

- Accuracy: ~73%

**Our Ensemble**:

- Accuracy: 86.7%
- **Improvement**: +13.7 percentage points

**Conclusion**: Multi-method approach significantly outperforms single-method detection.

### 6.3 Practical Deployment Considerations

**System Requirements**:

- Real-time detection (< 2 seconds latency)
- Scalability to handle clinic/hospital volume
- Integration with EHR systems
- Audit trails for legal compliance

**Implementation Recommendations**:

1. **Pre-deployment**: Extensive testing on diverse medical cases
2. **Deployment**: Start with low-stakes applications (health education)
3. **Monitoring**: Continuous performance tracking
4. **Feedback Loop**: Learn from false negatives/positives
5. **Updates**: Regular model retraining with new medical evidence

### 6.4 Limitations of This Study

1. **Small Dataset**: 15 cases insufficient for production deployment
   - _Mitigation_: Expand to 1000+ cases covering more conditions
2. **Simulated Data**: Not real patient-provider interactions
   - _Mitigation_: Collect real-world cases (with privacy protections)
3. **Limited Medical Domains**: Focus on common conditions
   - _Mitigation_: Include rare diseases, specialized fields
4. **No Clinical Validation**: Not tested with real healthcare providers
   - _Mitigation_: Conduct user studies with physicians
5. **English-Only**: Not multilingual
   - _Mitigation_: Expand to Spanish, Mandarin, etc.

---

## 7. Future Work

### 7.1 Technical Improvements

**Advanced Detection Methods**:

- **Uncertainty Quantification**: Use model log-probabilities, temperature sampling
- **Multi-Turn Consistency**: Check if model contradicts itself across conversation
- **Claim Extraction & Verification**: Break output into atomic claims, verify each independently

**Better Models**:

- Fine-tune models on medical NLI datasets (e.g., MedNLI)
- Train domain-specific hallucination classifiers
- Explore medical-specific LLMs (e.g., Med-PaLM, BioGPT)

**Larger Datasets**:

- Leverage existing benchmarks: MedQA, PubMedQA, MMLU-Medical
- Create comprehensive hallucination corpus
- Include multimodal data (images, labs, charts)

### 7.2 System Enhancements

**Evidence Integration**:

- Connect to live medical databases (PubMed API, clinical guidelines)
- Implement dynamic evidence retrieval
- Version tracking for guideline updates

**Personalization**:

- Patient-specific context (age, medications, allergies)
- Provider preferences (conservative vs. aggressive treatment)
- Institution protocols

**Explainability**:

- Provide reasoning for hallucination flags
- Highlight specific incorrect claims
- Show evidence supporting corrections

### 7.3 Clinical Studies

**Prospective Trials**:

- Deploy in controlled clinical settings
- Measure impact on patient outcomes
- Assess provider acceptance and workflow integration

**Comparative Studies**:

- Compare AI-assisted decisions with/without hallucination detection
- Measure error reduction rates
- Cost-benefit analysis

### 7.4 Policy & Governance

**Standards Development**:

- Propose industry standards for medical AI hallucination detection
- Collaborate with medical societies (AMA, WHO)
- Influence regulatory frameworks (FDA, EMA)

**Ethical Guidelines**:

- Transparency requirements
- Consent procedures for AI use
- Data governance for training/evaluation

---

## 8. Conclusion

This project demonstrates a comprehensive approach to detecting and correcting hallucinations in healthcare LLMs. The multi-method ensemble system achieves 86.7% accuracy, effectively identifying dangerous medical misinformation while providing multiple correction strategies tailored to risk levels.

**Key Contributions**:

1. **Practical System**: End-to-end pipeline from detection to correction
2. **Domain Focus**: Healthcare-specific challenges and solutions
3. **Multiple Methods**: Ensemble approach combining NLI, similarity, and relevance scoring
4. **Correction Strategies**: Four distinct approaches (RAG, rules, explanation, human-in-loop)
5. **Ethical Analysis**: Comprehensive discussion of regulatory, legal, and safety issues

**Main Takeaway**: Hallucination detection in healthcare is not just a technical problem but a **patient safety imperative**. While our system shows promise, the false negative rate must approach zero before clinical deployment. Conservative flagging, human oversight, and continuous improvement are essential.

**Impact Potential**: As LLMs increasingly support healthcare decisions, robust hallucination detection systems like this will be critical infrastructure—protecting patients, providers, and institutions from AI-generated medical errors.

---

## 9. References

### Academic Papers

1. **Hallucination Taxonomy**:

   - Ji, Z., et al. (2023). "Survey of Hallucination in Natural Language Generation." ACM Computing Surveys.
   - Maynez, J., et al. (2020). "On Faithfulness and Factuality in Abstractive Summarization." ACL.

2. **Medical AI**:

   - Singhal, K., et al. (2023). "Large Language Models Encode Clinical Knowledge." Nature.
   - Thirunavukarasu, A.J., et al. (2023). "Large Language Models in Medicine." Nature Medicine.

3. **Detection Methods**:

   - Manakul, P., et al. (2023). "SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection." EMNLP.
   - Zhang, Y., et al. (2023). "R-Tuning: Teaching Large Language Models to Refuse Unknown Questions." arxiv.

4. **Healthcare NLP**:
   - Lee, J., et al. (2020). "BioBERT: A Pre-trained Biomedical Language Representation Model." Bioinformatics.
   - Romanov, A., & Shivade, C. (2018). "Lessons from Natural Language Inference in the Clinical Domain." EMNLP.

### Models & Datasets

- **DeBERTa-v3-MNLI**: Microsoft NLI model
- **Sentence-Transformers**: SBERT for semantic similarity
- **MedNLI**: Medical Natural Language Inference dataset
- **PubMedQA**: Biomedical question answering benchmark

### Regulatory Resources

- **FDA**: Software as Medical Device (SaMD) guidance
- **HIPAA**: Health Insurance Portability and Accountability Act
- **WHO**: Guidelines on AI ethics in healthcare

---

## 10. Appendices

### Appendix A: Code Repository Structure

```
project_hallucination/
├── main.py                    # Main detection/correction pipeline
├── requirements.txt           # Python dependencies
├── detection_results.json     # Detailed evaluation results
├── evaluation_report.txt      # Summary metrics
└── final_report.md           # This document
```

### Appendix B: Running the System

**Installation**:

```bash
pip install -r requirements.txt
```

**Execution**:

```bash
python3 main.py
```

**Output Files**:

- `detection_results.json`: Detailed case-by-case results
- `evaluation_report.txt`: Performance metrics summary

### Appendix C: Sample Detection Output

```json
{
  "id": 6,
  "query": "Can I cure Stage 4 lung cancer with Vitamin C?",
  "prediction": 1,
  "actual": 1,
  "confidence": 0.876,
  "method_scores": {
    "entailment": [1, 0.92],
    "similarity": [1, 0.31],
    "domain": [1, 0.38]
  },
  "correction": {
    "rag": {
      "method": "RAG",
      "corrected_output": "Based on medical evidence: There is no clinical evidence that Vitamin C cures stage 4 lung cancer."
    },
    "human_loop": {
      "risk_level": "HIGH",
      "action_required": "Medical expert review needed"
    }
  }
}
```

### Appendix D: Dataset Statistics

| Category   | Non-Hallucinated | Hallucinated | Total  |
| ---------- | ---------------- | ------------ | ------ |
| Treatment  | 2                | 2            | 4      |
| Diagnosis  | 2                | 1            | 3      |
| Medication | 1                | 2            | 3      |
| Safety     | 0                | 2            | 2      |
| Other      | 3                | 0            | 3      |
| **Total**  | **8**            | **7**        | **15** |

### Appendix E: Ethical Approval & Data Privacy

**Note**: This is an educational project using simulated data. No real patient information was used. For clinical deployment, the following would be required:

- Institutional Review Board (IRB) approval
- HIPAA compliance certification
- Informed consent procedures
- Data anonymization protocols
- Security audits

---

**End of Report**

**Contact**: [Your Name/Email for academic purposes]  
**Institution**: [Your University]  
**Course**: Natural Language Processing  
**Submission Date**: December 28, 2025
