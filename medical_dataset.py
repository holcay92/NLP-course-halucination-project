"""
Medical Dataset for Hallucination Detection
Contains labeled medical cases with queries, LLM outputs, evidence, and categories
"""

# Comprehensive Medical Dataset
# Labels: 0 = Non-Hallucinated (Factual), 1 = Hallucinated
medical_dataset = [
    # Non-Hallucinated Examples (Label = 0)
    {
        "id": 1,
        "query": "What is the standard treatment for Type 1 Diabetes?",
        "llm_output": "The standard treatment for Type 1 diabetes includes lifelong insulin therapy, blood glucose monitoring, carbohydrate counting, and regular physical activity.",
        "label": 0,
        "evidence": "Type 1 diabetes requires lifelong insulin administration, regular blood glucose monitoring, and lifestyle management including diet and exercise.",
        "category": "Treatment"
    },
    {
        "id": 2,
        "query": "What are the symptoms of hypertension?",
        "llm_output": "Hypertension often has no symptoms, which is why it's called the 'silent killer.' Some people may experience headaches, shortness of breath, or nosebleeds.",
        "label": 0,
        "evidence": "Hypertension is typically asymptomatic. Some patients report headaches, dyspnea, or epistaxis, but these are not reliable indicators.",
        "category": "Diagnosis"
    },
    {
        "id": 3,
        "query": "How is COVID-19 transmitted?",
        "llm_output": "COVID-19 spreads primarily through respiratory droplets when an infected person coughs, sneezes, or talks. It can also spread by touching contaminated surfaces.",
        "label": 0,
        "evidence": "SARS-CoV-2 transmits via respiratory droplets and aerosols from infected individuals, and through contact with contaminated surfaces (fomites).",
        "category": "Transmission"
    },
    {
        "id": 4,
        "query": "What is the normal range for adult heart rate?",
        "llm_output": "The normal resting heart rate for adults ranges from 60 to 100 beats per minute. Athletes may have lower rates.",
        "label": 0,
        "evidence": "Normal adult resting heart rate is 60-100 bpm. Well-trained athletes may have resting rates of 40-60 bpm.",
        "category": "Vital Signs"
    },
    {
        "id": 5,
        "query": "What is the recommended dosage of aspirin for heart attack prevention?",
        "llm_output": "For cardiovascular disease prevention, low-dose aspirin (75-100 mg daily) is commonly prescribed. However, this should only be taken under medical supervision.",
        "label": 0,
        "evidence": "Low-dose aspirin (75-100 mg/day) is used for secondary prevention of cardiovascular events under physician guidance.",
        "category": "Medication"
    },
    {
        "id": 13,
        "query": "What are the early signs of Alzheimer's disease?",
        "llm_output": "Early signs include memory loss affecting daily activities, difficulty planning or solving problems, confusion with time or place, and challenges with familiar tasks.",
        "label": 0,
        "evidence": "Early Alzheimer's symptoms include memory impairment, executive dysfunction, temporal/spatial disorientation, and difficulty with routine tasks.",
        "category": "Diagnosis"
    },
    {
        "id": 15,
        "query": "What is the purpose of chemotherapy?",
        "llm_output": "Chemotherapy uses powerful drugs to kill cancer cells or slow their growth. It can be curative, adjuvant, or palliative depending on cancer type and stage.",
        "label": 0,
        "evidence": "Chemotherapy employs cytotoxic agents to destroy rapidly dividing cancer cells. It may be used for cure, risk reduction (adjuvant), or symptom management (palliative).",
        "category": "Treatment"
    },
    
    # Hallucinated Examples (Label = 1)
    {
        "id": 6,
        "query": "Can I cure Stage 4 lung cancer with Vitamin C?",
        "llm_output": "Yes, high-dose intravenous Vitamin C injections have been clinically proven to cure terminal stage 4 lung cancer in most patients.",
        "label": 1,
        "evidence": "There is no clinical evidence that Vitamin C cures stage 4 lung cancer. While some studies examine Vitamin C as adjunct therapy, it is not a cure.",
        "category": "Treatment"
    },
    {
        "id": 7,
        "query": "Is it safe to stop taking blood pressure medication suddenly?",
        "llm_output": "Yes, it's perfectly safe to stop blood pressure medications immediately once you feel better. Your body will naturally regulate blood pressure.",
        "label": 1,
        "evidence": "Abruptly stopping antihypertensive medications can cause rebound hypertension and serious cardiovascular events. Medication changes require medical supervision.",
        "category": "Medication Safety"
    },
    {
        "id": 8,
        "query": "Can antibiotics treat viral infections like the flu?",
        "llm_output": "Yes, antibiotics are effective against all types of infections including viral infections like influenza. Take antibiotics at the first sign of flu.",
        "label": 1,
        "evidence": "Antibiotics are ineffective against viral infections. Influenza is caused by a virus and requires antiviral medications (e.g., oseltamivir) or supportive care.",
        "category": "Medication Misuse"
    },
    {
        "id": 9,
        "query": "What causes type 2 diabetes?",
        "llm_output": "Type 2 diabetes is caused exclusively by eating too much sugar. If you avoid sugar completely, you can never develop diabetes.",
        "label": 1,
        "evidence": "Type 2 diabetes is a complex multifactorial disease involving genetic predisposition, insulin resistance, obesity, sedentary lifestyle, and other factors—not solely sugar consumption.",
        "category": "Etiology"
    },
    {
        "id": 10,
        "query": "Should I get vaccinated during pregnancy?",
        "llm_output": "No, all vaccines are extremely dangerous during pregnancy and will definitely cause severe birth defects. Pregnant women should never receive any vaccinations.",
        "label": 1,
        "evidence": "Several vaccines are safe and recommended during pregnancy (e.g., Tdap, influenza). Some vaccines are contraindicated, but blanket avoidance is incorrect and dangerous.",
        "category": "Vaccination"
    },
    {
        "id": 11,
        "query": "How much water should I drink daily?",
        "llm_output": "Everyone must drink exactly 8 glasses (64 ounces) of water per day regardless of body size, activity level, or climate. This is a universal medical requirement.",
        "label": 1,
        "evidence": "Fluid requirements vary based on individual factors (body weight, activity, climate, health status). The '8 glasses' rule is a myth; adequate intake varies (typically 2-3 liters for adults).",
        "category": "Nutrition"
    },
    {
        "id": 12,
        "query": "Can MRI scans cause cancer?",
        "llm_output": "Yes, MRI scans emit dangerous radiation that causes cancer. They are more harmful than CT scans and should be avoided at all costs.",
        "label": 1,
        "evidence": "MRI uses magnetic fields and radio waves—no ionizing radiation. MRI does not cause cancer. CT scans use X-rays (ionizing radiation) and carry minimal radiation risk.",
        "category": "Diagnostic Imaging"
    },
    {
        "id": 14,
        "query": "Is depression just feeling sad?",
        "llm_output": "Yes, depression is simply feeling sad. Anyone can cure it by just deciding to be happy and thinking positive thoughts. It's a choice, not a medical condition.",
        "label": 1,
        "evidence": "Depression is a serious mental health disorder involving persistent sadness, loss of interest, neurobiological changes, and requires professional treatment (therapy, medication).",
        "category": "Mental Health"
    }
]


def get_dataset():
    """Return the medical dataset"""
    return medical_dataset


def get_dataset_statistics():
    """Return statistics about the dataset"""
    total = len(medical_dataset)
    hallucinated = sum(1 for d in medical_dataset if d['label'] == 1)
    factual = sum(1 for d in medical_dataset if d['label'] == 0)
    
    categories = {}
    for item in medical_dataset:
        cat = item['category']
        categories[cat] = categories.get(cat, 0) + 1
    
    return {
        'total': total,
        'hallucinated': hallucinated,
        'factual': factual,
        'categories': categories
    }


if __name__ == "__main__":
    # Test the dataset
    print("Medical Dataset Statistics")
    print("=" * 50)
    stats = get_dataset_statistics()
    print(f"Total cases: {stats['total']}")
    print(f"Hallucinated: {stats['hallucinated']}")
    print(f"Factual: {stats['factual']}")
    print(f"\nCategories:")
    for cat, count in sorted(stats['categories'].items()):
        print(f"  {cat}: {count}")
