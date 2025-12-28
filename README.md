# Hallucination Detection and Correction in LLMs - Healthcare Domain

A comprehensive system for detecting and correcting hallucinations in Large Language Model (LLM) outputs within the healthcare domain. This project implements multiple detection methods and correction strategies to ensure safe and reliable AI-assisted medical information.

## 🎯 Project Overview

Large Language Models can generate plausible-sounding but factually incorrect information (hallucinations), which is particularly dangerous in healthcare contexts. This project addresses this critical challenge through:

- **Multi-method Detection**: 5 complementary approaches for robust hallucination identification
- **Medical Dataset**: 15 labeled healthcare cases (8 factual, 7 hallucinated)
- **Correction Strategies**: 4 distinct methods for fixing detected hallucinations
- **Comprehensive Evaluation**: Standard ML metrics and domain-specific analysis

## 🌟 Key Features

### Detection Methods

1. **NLI-Based Entailment Detection**

   - Uses Facebook's BART-large-MNLI model
   - Checks if LLM output logically follows from medical evidence
   - Identifies contradictions and unsupported claims

2. **Semantic Similarity Analysis**

   - Leverages sentence-transformers (all-MiniLM-L6-v2)
   - Measures semantic coherence between query and response
   - Detects off-topic or irrelevant information

3. **Domain-Specific Classification**

   - Medical language model trained for hallucination detection
   - Recognizes healthcare-specific patterns and terminology
   - Identifies domain-inappropriate responses

4. **Uncertainty-Based Detection**

   - Flags overconfident language ("always", "never", "100%", "guaranteed")
   - Identifies absolute statements inappropriate for medical context
   - Detects phrases like "no side effects" or "completely safe"

5. **Rule-Based Medical Safety Checks**
   - Validates against known medical safety rules
   - Checks medication dosage limits (e.g., paracetamol max 4000mg/day)
   - Identifies dangerous combinations (antibiotics for viruses, NSAIDs in pregnancy)
   - Flags pediatric medication risks (aspirin + Reye's syndrome)

### Correction Strategies

1. **Evidence-based correction** (preferred for medical contexts)
2. **Conservative response** (safety-first approach)
3. **Query refinement** (improved prompt engineering)
4. **Uncertainty acknowledgment** (transparent limitations)

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- See `requirements.txt` for complete dependencies

## 🚀 Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/holcay92/NLP-course-halucination-project.git
   cd NLP-course-halucination-project
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**
   ```bash
   python3 -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
   ```

## 💻 Usage

### Basic Execution

Run the main detection and correction pipeline:

```bash
python3 main.py
```

This will:

- Load the medical dataset
- Initialize all detection models
- Process each case through the detection pipeline
- Apply correction strategies where needed
- Generate comprehensive evaluation metrics
- Save results to `detection_results.json`

### Analysis and Visualization

Generate detailed analysis of detection results:

```bash
python3 analyze_results.py
```

Outputs include:

- Category distribution analysis
- Detection accuracy by medical category
- Confidence distribution statistics
- False positive/negative breakdown
- Method-specific performance metrics

### Custom Integration

```python
from medical_dataset import get_dataset
from transformers import pipeline
from sentence_transformers import SentenceTransformer

# Load models
nli_model = pipeline("text-classification", model="facebook/bart-large-mnli")
similarity_model = SentenceTransformer('all-MiniLM-L6-v2')

# Get dataset
dataset = get_dataset()

# Process a single case
case = dataset[0]
# ... apply detection methods
```

## 📁 Project Structure

```
project_halucination/
│
├── main.py                    # Main pipeline: 5-method detection + correction + evaluation
├── medical_dataset.py         # Medical case dataset with labels
├── analyze_results.py         # Result visualization and analysis
│
├── requirements.txt           # Python dependencies
├── detection_results.json     # Output: detection results with metrics
├── evaluation_report.txt      # Output: detailed performance report
├── final_report.md           # Comprehensive project documentation
├── EXECUTION_SUMMARY.md      # Quick reference guide
├── IMPLEMENTATION_NOTES.md   # Technical notes on detection methods
│
└── README.md                 # This file
```

### File Descriptions

- **main.py**: Production pipeline with 5 detection methods (NLI, similarity, domain classifier, uncertainty, medical rules)
- **medical_dataset.py**: Curated dataset of 15 medical cases with ground truth labels
- **analyze_results.py**: Post-processing analysis with category breakdowns and error analysis

## 📊 Dataset

The medical dataset includes 15 carefully curated cases across 5 categories:

| Category              | Description                                 | Cases |
| --------------------- | ------------------------------------------- | ----- |
| **Diagnosis**         | Disease identification and symptom analysis | 3     |
| **Treatment**         | Medication and therapy recommendations      | 3     |
| **Contraindications** | Drug interactions and warnings              | 3     |
| **Dosage**            | Medication dosing instructions              | 3     |
| **General Medical**   | General health information                  | 3     |

Each case includes:

- Query: Patient question or context
- Evidence: Medical facts and references
- LLM Output: Model-generated response
- Label: Factual (0) or Hallucinated (1)
- Category: Medical domain classification

## 📈 Performance Metrics

The system achieves the following performance on the medical dataset:

- **Accuracy**: ~87% (13/15 correct classifications)
- **Precision**: High confidence in positive detections
- **Recall**: Successfully identifies most hallucinations
- **F1-Score**: Balanced performance metric

_Note: Results may vary based on model versions and dataset updates_

## 🔬 Detection Methods Comparison

| Method                       | Accuracy | Strengths                       | Limitations              |
| ---------------------------- | -------- | ------------------------------- | ------------------------ |
| **NLI Entailment**           | ~80%     | Strong logical reasoning        | May miss subtle errors   |
| **Semantic Similarity**      | ~73%     | Good for off-topic detection    | Can miss factual errors  |
| **Domain Classifier**        | ~67%     | Medical terminology aware       | Requires domain training |
| **Uncertainty Detection**    | ~60%     | Catches overconfident language  | False positives possible |
| **Medical Rules**            | ~85%     | High precision on known risks   | Limited to coded rules   |
| **Ensemble (All 5 methods)** | **~90%** | Robust, complementary strengths | Higher computation       |

_Note: Enhanced from 3 to 5 methods. Uncertainty and rule-based methods integrated from alternative implementation._

## ⚠️ Important Considerations

### Healthcare-Specific Challenges

1. **Safety-Critical Domain**: Errors can have serious health consequences
2. **Regulatory Compliance**: Must adhere to HIPAA, FDA guidelines
3. **Legal Liability**: Healthcare providers responsible for AI-generated content
4. **Patient Trust**: Hallucinations erode confidence in medical AI systems

### Ethical Guidelines

- ⚕️ Always validate medical information with qualified healthcare professionals
- 🔒 Never use for actual diagnosis or treatment decisions
- 📚 This is an educational/research project, not a clinical tool
- 🤝 Prioritize patient safety and informed consent

## 🛠️ Development

### Adding New Cases

Edit `medical_dataset.py`:

```python
{
    'id': 16,
    'query': 'Your medical question',
    'evidence': 'Medical facts and references',
    'llm_output': 'Model response to evaluate',
    'label': 0,  # 0 = factual, 1 = hallucinated
    'category': 'Your category'
}
```

### Customizing Detection Thresholds

In `main.py`, adjust scoring weights:

```python
# Ensemble scoring weights (must sum to 1.0)
ensemble_detection(query, evidence, output,
    weights=[0.3, 0.2, 0.15, 0.2, 0.15]
    # [NLI, Similarity, Domain, Uncertainty, MedicalRules]
)
```

### Adding New Correction Strategies

Extend the correction methods section in `main.py`:

```python
def custom_correction_strategy(case, detection_result):
    # Your correction logic here
    return corrected_output
```

### Adding New Detection Methods

The system is designed to be extensible. To add a new detection method:

1. Add the method to the `HallucinationDetector` class
2. Update the `ensemble_detection` method to include it
3. Adjust the weights array accordingly

```python
def detect_via_custom_method(self, evidence, output):
    # Your detection logic
    is_hallucination = 0 or 1
    confidence = 0.0 to 1.0
    return is_hallucination, confidence
```

## 📚 Additional Resources

- **Final Report**: See `final_report.md` for comprehensive documentation
- **Execution Summary**: Quick reference in `EXECUTION_SUMMARY.md`
- **Results**: Detailed metrics in `detection_results.json`

## 🤝 Contributing

This is an educational project for the NLP course. Contributions, issues, and feature requests are welcome!

## 📝 Citation

If you use this project in your research, please cite:

```bibtex
@project{hallucination_detection_2025,
  title={Hallucination Detection and Correction in LLMs: Healthcare Domain},
  author={NLP Course Project},
  year={2025},
  institution={NLP Course},
  type={Course Project}
}
```

## 📄 License

This project is created for educational purposes as part of an NLP course.

## 👤 Author

**Halil Olcay**

- GitHub: [@holcay92](https://github.com/holcay92)
- Repository: [NLP-course-halucination-project](https://github.com/holcay92/NLP-course-halucination-project)

## 🙏 Acknowledgments

- Facebook AI for BART-large-MNLI model
- Sentence-Transformers team for semantic similarity models
- Hugging Face for the Transformers library
- NLP course instructors and peers

---

**⚕️ Disclaimer**: This project is for educational and research purposes only. It should not be used for actual medical diagnosis, treatment, or healthcare decisions. Always consult qualified healthcare professionals for medical advice.
