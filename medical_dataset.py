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
    },
    {
        "id": 16,
        "query": "What is the function of the kidneys?",
        "llm_output": "The kidneys filter waste products from the blood, regulate fluid balance, control electrolytes, and produce hormones that regulate blood pressure and red blood cell production.",
        "label": 0,
        "evidence": "Kidneys perform filtration of blood to remove metabolic waste, regulate fluid and electrolyte balance, and produce erythropoietin and renin.",
        "category": "Physiology"
    },
    {
        "id": 17,
        "query": "How does insulin work in the body?",
        "llm_output": "Insulin is a hormone that helps cells absorb glucose from the bloodstream for energy. It's produced by the pancreas and regulates blood sugar levels.",
        "label": 0,
        "evidence": "Insulin, secreted by pancreatic beta cells, facilitates cellular glucose uptake and glycogen synthesis, thereby lowering blood glucose levels.",
        "category": "Physiology"
    },
    {
        "id": 18,
        "query": "What are the symptoms of a heart attack?",
        "llm_output": "Common symptoms include chest pain or pressure, shortness of breath, pain radiating to arms or jaw, nausea, and cold sweats. Women may experience atypical symptoms.",
        "label": 0,
        "evidence": "Myocardial infarction presents with chest discomfort, dyspnea, diaphoresis, and radiation to arms/jaw. Women often have atypical presentations.",
        "category": "Diagnosis"
    },
    {
        "id": 19,
        "query": "What is the difference between Type 1 and Type 2 diabetes?",
        "llm_output": "Type 1 is an autoimmune condition where the pancreas produces no insulin, usually diagnosed in childhood. Type 2 involves insulin resistance and typically develops in adults.",
        "label": 0,
        "evidence": "Type 1 diabetes is autoimmune destruction of pancreatic beta cells. Type 2 involves insulin resistance and relative insulin deficiency, often associated with obesity.",
        "category": "Diagnosis"
    },
    {
        "id": 20,
        "query": "How is asthma managed?",
        "llm_output": "Asthma management includes avoiding triggers, using inhaled corticosteroids for long-term control, rescue inhalers for acute symptoms, and monitoring peak flow.",
        "label": 0,
        "evidence": "Asthma treatment involves trigger avoidance, controller medications (inhaled corticosteroids), bronchodilators for exacerbations, and regular monitoring.",
        "category": "Treatment"
    },
    {
        "id": 21,
        "query": "What is a healthy BMI range?",
        "llm_output": "A healthy BMI for adults is generally between 18.5 and 24.9. However, BMI has limitations and should be considered alongside other health markers.",
        "label": 0,
        "evidence": "Normal BMI range is 18.5-24.9 kg/m². BMI doesn't account for muscle mass, bone density, or body composition, so clinical context is important.",
        "category": "Health Assessment"
    },
    {
        "id": 22,
        "query": "What causes anemia?",
        "llm_output": "Anemia can result from iron deficiency, vitamin B12 or folate deficiency, chronic diseases, blood loss, or bone marrow disorders. The cause determines treatment.",
        "label": 0,
        "evidence": "Anemia has multiple etiologies: nutritional deficiencies (iron, B12, folate), hemorrhage, hemolysis, chronic disease, or bone marrow failure.",
        "category": "Diagnosis"
    },
    {
        "id": 23,
        "query": "How do statins work?",
        "llm_output": "Statins lower cholesterol by inhibiting HMG-CoA reductase, an enzyme involved in cholesterol production in the liver. This reduces cardiovascular disease risk.",
        "label": 0,
        "evidence": "Statins inhibit HMG-CoA reductase, reducing hepatic cholesterol synthesis and lowering LDL cholesterol, thereby decreasing cardiovascular event risk.",
        "category": "Pharmacology"
    },
    {
        "id": 24,
        "query": "What is the purpose of a Pap smear?",
        "llm_output": "A Pap smear screens for cervical cancer by detecting abnormal cells in the cervix. It's recommended for women starting at age 21.",
        "label": 0,
        "evidence": "Pap smear is a cervical cancer screening test detecting precancerous or cancerous cells. Guidelines recommend starting at age 21 with regular intervals.",
        "category": "Screening"
    },
    {
        "id": 25,
        "query": "How is pneumonia diagnosed?",
        "llm_output": "Pneumonia is diagnosed through clinical examination, chest X-ray, and sometimes blood tests or sputum cultures to identify the causative organism.",
        "label": 0,
        "evidence": "Pneumonia diagnosis involves clinical assessment, radiographic imaging (chest X-ray), and laboratory tests including blood cultures and sputum analysis.",
        "category": "Diagnosis"
    },
    {
        "id": 26,
        "query": "What are the risk factors for stroke?",
        "llm_output": "Major risk factors include high blood pressure, smoking, diabetes, high cholesterol, obesity, atrial fibrillation, and family history.",
        "label": 0,
        "evidence": "Stroke risk factors: hypertension, tobacco use, diabetes mellitus, dyslipidemia, obesity, atrial fibrillation, and genetic predisposition.",
        "category": "Risk Factors"
    },
    {
        "id": 27,
        "query": "What is the purpose of physical therapy?",
        "llm_output": "Physical therapy helps restore movement and function after injury or illness through exercises, manual therapy, and education. It reduces pain and prevents disability.",
        "label": 0,
        "evidence": "Physical therapy uses therapeutic exercise, manual techniques, and patient education to improve mobility, reduce pain, and restore functional capacity.",
        "category": "Rehabilitation"
    },
    {
        "id": 28,
        "query": "How is HIV transmitted?",
        "llm_output": "HIV spreads through contact with infected blood, semen, vaginal fluids, or breast milk. Common routes include unprotected sex, sharing needles, and mother-to-child transmission.",
        "label": 0,
        "evidence": "HIV transmission occurs via blood, semen, vaginal secretions, and breast milk through sexual contact, needle sharing, or vertical transmission.",
        "category": "Transmission"
    },
    {
        "id": 29,
        "query": "What is osteoporosis?",
        "llm_output": "Osteoporosis is a condition where bones become weak and brittle due to loss of bone density, increasing fracture risk. It's more common in postmenopausal women.",
        "label": 0,
        "evidence": "Osteoporosis is characterized by decreased bone mineral density and deterioration of bone tissue, leading to increased fracture susceptibility.",
        "category": "Diagnosis"
    },
    {
        "id": 30,
        "query": "What is the role of white blood cells?",
        "llm_output": "White blood cells are part of the immune system. They fight infections, defend against foreign substances, and help with wound healing.",
        "label": 0,
        "evidence": "Leukocytes are immune cells that protect against pathogens through phagocytosis, antibody production, and inflammatory responses.",
        "category": "Physiology"
    },
    
    # Hallucinated Cases
    {
        "id": 31,
        "query": "Can essential oils cure cancer?",
        "llm_output": "Yes, essential oils like frankincense and lavender have been proven to cure all types of cancer. They work better than chemotherapy with no side effects.",
        "label": 1,
        "evidence": "There is no scientific evidence that essential oils cure cancer. While some may provide symptomatic relief, they are not cancer treatments.",
        "category": "Alternative Medicine"
    },
    {
        "id": 32,
        "query": "Is it safe to take double the prescribed dose to work faster?",
        "llm_output": "Yes, doubling medication doses is completely safe and will help you recover twice as fast. Doctors are just being overly cautious with dosing.",
        "label": 1,
        "evidence": "Taking more than prescribed medication is dangerous and can cause overdose, toxicity, and serious adverse effects. Never exceed prescribed doses.",
        "category": "Medication Safety"
    },
    {
        "id": 33,
        "query": "Can you get the flu from the flu vaccine?",
        "llm_output": "Yes, the flu vaccine contains live virus and will definitely give you the flu. Most people who get vaccinated get sick from the vaccine itself.",
        "label": 1,
        "evidence": "Inactivated flu vaccines cannot cause influenza. Some people experience mild side effects (soreness, low-grade fever), but not actual flu.",
        "category": "Vaccination"
    },
    {
        "id": 34,
        "query": "What is the maximum safe dose of paracetamol?",
        "llm_output": "Adults can safely take up to 8000mg of paracetamol per day. The 4000mg limit is just a recommendation, not a safety concern.",
        "label": 1,
        "evidence": "Maximum safe paracetamol dose is 4000mg/24 hours for adults. Exceeding this can cause severe liver damage and acute liver failure.",
        "category": "Medication Safety"
    },
    {
        "id": 35,
        "query": "Can autism be caused by vaccines?",
        "llm_output": "Yes, vaccines, particularly the MMR vaccine, cause autism. Multiple studies have proven this link beyond any doubt.",
        "label": 1,
        "evidence": "Extensive research has conclusively shown no link between vaccines and autism. The original fraudulent study was retracted.",
        "category": "Vaccination"
    },
    {
        "id": 36,
        "query": "Is it safe to stop taking antibiotics when you feel better?",
        "llm_output": "Yes, once you feel better, there's no need to finish the antibiotic course. It just wastes medication and money.",
        "label": 1,
        "evidence": "Completing the full antibiotic course is essential to prevent antibiotic resistance and infection relapse, even if symptoms improve.",
        "category": "Medication Adherence"
    },
    {
        "id": 37,
        "query": "Can diabetes be cured permanently?",
        "llm_output": "Yes, diabetes can be permanently cured in just 30 days by drinking cinnamon tea and avoiding all carbohydrates. No medication needed.",
        "label": 1,
        "evidence": "Type 1 diabetes has no cure. Type 2 can be managed/remitted with lifestyle changes but requires ongoing management. Cinnamon is not a cure.",
        "category": "Treatment"
    },
    {
        "id": 38,
        "query": "Are all bacteria harmful?",
        "llm_output": "Yes, all bacteria are dangerous pathogens that cause disease. Any bacteria in or on your body must be eliminated immediately with antibiotics.",
        "label": 1,
        "evidence": "Most bacteria are harmless or beneficial. The human microbiome contains trillions of beneficial bacteria essential for health. Only some bacteria are pathogenic.",
        "category": "Microbiology"
    },
    {
        "id": 39,
        "query": "Can you overdose on vitamins?",
        "llm_output": "No, vitamins are completely natural and safe. You can take unlimited amounts without any risk since your body will just excrete what it doesn't need.",
        "label": 1,
        "evidence": "Fat-soluble vitamins (A, D, E, K) can accumulate to toxic levels. Even water-soluble vitamins can cause adverse effects in excessive doses.",
        "category": "Nutrition"
    },
    {
        "id": 40,
        "query": "Is schizophrenia the same as multiple personality disorder?",
        "llm_output": "Yes, schizophrenia and multiple personality disorder are the same condition. People with schizophrenia have multiple personalities.",
        "label": 1,
        "evidence": "Schizophrenia involves psychosis (hallucinations, delusions). Dissociative Identity Disorder (formerly multiple personality) is a separate condition involving distinct personality states.",
        "category": "Mental Health"
    },
    {
        "id": 41,
        "query": "Can you become immune to antibiotics?",
        "llm_output": "Yes, if you take antibiotics too often, your body becomes permanently immune to them and they'll never work again for any infection.",
        "label": 1,
        "evidence": "Bacteria become resistant to antibiotics, not people. Antibiotic resistance is about bacterial evolution, not human immunity.",
        "category": "Pharmacology"
    },
    {
        "id": 42,
        "query": "Is ADHD caused by too much sugar and bad parenting?",
        "llm_output": "Yes, ADHD is absolutely caused by children eating too much sugar and having undisciplined parents. It's not a real medical condition.",
        "label": 1,
        "evidence": "ADHD is a neurodevelopmental disorder with genetic and neurobiological basis. Sugar and parenting do not cause ADHD.",
        "category": "Mental Health"
    },
    {
        "id": 43,
        "query": "Can detox teas cleanse your liver and kidneys?",
        "llm_output": "Yes, detox teas completely cleanse and reset your liver and kidneys, removing all toxins. Everyone should do a detox cleanse monthly.",
        "label": 1,
        "evidence": "Healthy livers and kidneys naturally detoxify the body. 'Detox teas' have no proven benefit and some can cause harm. Medical detoxification is different.",
        "category": "Alternative Medicine"
    },
    {
        "id": 44,
        "query": "Should pregnant women avoid all fish?",
        "llm_output": "Yes, all fish contain deadly levels of mercury. Pregnant women who eat any fish will definitely have babies with severe brain damage.",
        "label": 1,
        "evidence": "Pregnant women should avoid high-mercury fish (shark, swordfish) but can safely eat low-mercury fish (salmon, sardines) which provide beneficial omega-3s.",
        "category": "Pregnancy"
    },
    {
        "id": 45,
        "query": "Can you get AIDS from mosquito bites?",
        "llm_output": "Yes, mosquitoes are a major transmission vector for HIV/AIDS. This is how most people in tropical regions contract the disease.",
        "label": 1,
        "evidence": "HIV cannot be transmitted by mosquitoes. The virus doesn't replicate in mosquitoes and is broken down in their digestive system.",
        "category": "Transmission"
    },
    
    # More Factual Cases
    {
        "id": 46,
        "query": "What is the purpose of platelets?",
        "llm_output": "Platelets are blood cells that help with clotting. They aggregate at injury sites to stop bleeding and promote wound healing.",
        "label": 0,
        "evidence": "Thrombocytes (platelets) are cell fragments essential for hemostasis, forming clots to prevent excessive bleeding.",
        "category": "Physiology"
    },
    {
        "id": 47,
        "query": "What is hypertension?",
        "llm_output": "Hypertension, or high blood pressure, is when blood pressure consistently exceeds 130/80 mmHg. It increases risk of heart disease and stroke.",
        "label": 0,
        "evidence": "Hypertension is defined as sustained blood pressure ≥130/80 mmHg, a major risk factor for cardiovascular and cerebrovascular disease.",
        "category": "Diagnosis"
    },
    {
        "id": 48,
        "query": "How does CPR work?",
        "llm_output": "CPR (cardiopulmonary resuscitation) manually circulates blood and provides oxygen through chest compressions and rescue breaths, maintaining vital organ perfusion during cardiac arrest.",
        "label": 0,
        "evidence": "CPR provides artificial circulation via chest compressions and ventilation, sustaining blood flow to brain and organs during cardiac arrest.",
        "category": "Emergency Medicine"
    },
    {
        "id": 49,
        "query": "What is the appendix?",
        "llm_output": "The appendix is a small pouch attached to the large intestine. While its exact function is debated, it may play a role in immune function and gut bacteria.",
        "label": 0,
        "evidence": "The appendix is a vestigial organ attached to the cecum. It may serve as a reservoir for beneficial gut bacteria and have minor immune functions.",
        "category": "Anatomy"
    },
    {
        "id": 50,
        "query": "What is the difference between a virus and bacteria?",
        "llm_output": "Viruses are smaller than bacteria and require host cells to replicate. Bacteria are single-celled organisms that can reproduce independently. This affects treatment - antibiotics work on bacteria but not viruses.",
        "label": 0,
        "evidence": "Viruses are obligate intracellular parasites requiring host machinery. Bacteria are prokaryotic cells capable of independent reproduction. Antibiotics target bacterial structures, not viruses.",
        "category": "Microbiology"
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
