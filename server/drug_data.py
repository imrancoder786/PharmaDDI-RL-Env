"""
Pharmaceutical Drug Interaction Knowledge Base.

Contains a curated database of ~30 common drugs and ~60 known drug-drug
interactions (DDIs) across major therapeutic classes, along with patient
scenario generators for 3 difficulty-graded tasks.
"""

import random
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Drug:
    name: str
    therapeutic_class: str
    common_dose: str
    frequency: str


@dataclass
class Interaction:
    drug_a: str
    drug_b: str
    severity: str          # minor | moderate | major | contraindicated
    clinical_effect: str
    recommendation: str    # monitor | adjust_dose | substitute | discontinue


@dataclass
class PatientScenario:
    patient_id: str
    age: int
    conditions: List[str]
    medications: List[Drug]
    ground_truth_interactions: List[Interaction]
    task_name: str
    task_difficulty: str


# ---------------------------------------------------------------------------
# Drug database (30 common drugs)
# ---------------------------------------------------------------------------

DRUGS: Dict[str, Drug] = {d.name: d for d in [
    # Cardiovascular
    Drug("warfarin",       "anticoagulant",    "5mg",    "once daily"),
    Drug("aspirin",        "antiplatelet",     "81mg",   "once daily"),
    Drug("clopidogrel",    "antiplatelet",     "75mg",   "once daily"),
    Drug("heparin",        "anticoagulant",    "5000U",  "twice daily"),
    Drug("lisinopril",     "ACE_inhibitor",    "10mg",   "once daily"),
    Drug("losartan",       "ARB",             "50mg",   "once daily"),
    Drug("amlodipine",     "calcium_blocker",  "5mg",    "once daily"),
    Drug("metoprolol",     "beta_blocker",     "50mg",   "twice daily"),
    Drug("atorvastatin",   "statin",           "20mg",   "once daily"),
    Drug("simvastatin",    "statin",           "40mg",   "once daily"),

    # CNS / Psych
    Drug("fluoxetine",     "SSRI",             "20mg",   "once daily"),
    Drug("sertraline",     "SSRI",             "50mg",   "once daily"),
    Drug("diazepam",       "benzodiazepine",   "5mg",    "twice daily"),
    Drug("tramadol",       "opioid_analgesic", "50mg",   "every 6 hours"),
    Drug("lithium",        "mood_stabilizer",  "300mg",  "twice daily"),
    Drug("carbamazepine",  "anticonvulsant",   "200mg",  "twice daily"),
    Drug("phenytoin",      "anticonvulsant",   "100mg",  "three times daily"),

    # Antibiotics / Anti-infectives
    Drug("ciprofloxacin",  "fluoroquinolone",  "500mg",  "twice daily"),
    Drug("metronidazole",  "antibiotic",       "500mg",  "three times daily"),
    Drug("fluconazole",    "antifungal",       "150mg",  "once daily"),
    Drug("rifampin",       "antibiotic",       "600mg",  "once daily"),
    Drug("erythromycin",   "macrolide",        "500mg",  "four times daily"),

    # Analgesics / Anti-inflammatory
    Drug("ibuprofen",      "NSAID",            "400mg",  "every 6 hours"),
    Drug("naproxen",       "NSAID",            "250mg",  "twice daily"),
    Drug("acetaminophen",  "analgesic",        "500mg",  "every 6 hours"),

    # Diabetes
    Drug("metformin",      "biguanide",        "500mg",  "twice daily"),
    Drug("glipizide",      "sulfonylurea",     "5mg",    "once daily"),

    # Other
    Drug("omeprazole",     "PPI",              "20mg",   "once daily"),
    Drug("levothyroxine",  "thyroid_hormone",  "50mcg",  "once daily"),
    Drug("spironolactone", "K_sparing_diuretic","25mg",  "once daily"),
]}


# ---------------------------------------------------------------------------
# Interaction database (~60 DDI pairs)
# ---------------------------------------------------------------------------

INTERACTIONS: List[Interaction] = [
    # --- Anticoagulant / Antiplatelet bleeding risks ---
    Interaction("warfarin", "aspirin",        "major",           "Increased bleeding risk due to combined anticoagulant and antiplatelet effects",                     "monitor"),
    Interaction("warfarin", "ibuprofen",      "major",           "NSAIDs increase bleeding risk and may displace warfarin from protein binding",                       "substitute"),
    Interaction("warfarin", "naproxen",       "major",           "NSAIDs increase bleeding risk with warfarin",                                                        "substitute"),
    Interaction("warfarin", "clopidogrel",    "major",           "Dual antithrombotic therapy significantly increases hemorrhage risk",                                 "monitor"),
    Interaction("warfarin", "fluconazole",    "major",           "Fluconazole inhibits CYP2C9, dramatically increasing warfarin levels and bleeding risk",              "adjust_dose"),
    Interaction("warfarin", "metronidazole",  "major",           "Metronidazole inhibits warfarin metabolism, increasing INR and bleeding risk",                        "adjust_dose"),
    Interaction("warfarin", "rifampin",       "major",           "Rifampin induces CYP enzymes, dramatically reducing warfarin effectiveness",                          "adjust_dose"),
    Interaction("warfarin", "erythromycin",   "moderate",        "Erythromycin may inhibit warfarin metabolism, increasing INR",                                        "monitor"),
    Interaction("warfarin", "omeprazole",     "minor",           "Omeprazole may slightly increase warfarin levels via CYP2C19 inhibition",                            "monitor"),
    Interaction("warfarin", "simvastatin",    "moderate",        "Simvastatin may enhance anticoagulant effect of warfarin",                                           "monitor"),
    Interaction("heparin",  "aspirin",        "major",           "Combined use increases risk of serious bleeding",                                                     "monitor"),
    Interaction("aspirin",  "ibuprofen",      "moderate",        "Ibuprofen may reduce cardioprotective effect of aspirin",                                            "substitute"),
    Interaction("aspirin",  "naproxen",       "moderate",        "NSAIDs may interfere with aspirin's antiplatelet effect",                                            "substitute"),
    Interaction("clopidogrel", "omeprazole",  "moderate",        "Omeprazole reduces clopidogrel activation via CYP2C19 inhibition",                                  "substitute"),

    # --- ACE inhibitor / ARB / potassium ---
    Interaction("lisinopril", "losartan",     "contraindicated", "Dual RAAS blockade increases risk of hyperkalemia, hypotension, and renal failure",                  "discontinue"),
    Interaction("lisinopril", "spironolactone","major",          "Combined use significantly increases hyperkalemia risk",                                              "monitor"),
    Interaction("losartan",   "spironolactone","major",          "Combined use significantly increases hyperkalemia risk",                                              "monitor"),
    Interaction("lisinopril", "ibuprofen",    "moderate",        "NSAIDs reduce antihypertensive effect and increase renal risk with ACE inhibitors",                  "substitute"),
    Interaction("lisinopril", "naproxen",     "moderate",        "NSAIDs reduce antihypertensive effect and worsen renal function",                                    "substitute"),
    Interaction("losartan",   "ibuprofen",    "moderate",        "NSAIDs reduce ARB effectiveness and increase renal risk",                                            "substitute"),

    # --- Statin interactions ---
    Interaction("simvastatin", "erythromycin", "contraindicated","Erythromycin inhibits CYP3A4, causing dangerous simvastatin accumulation and rhabdomyolysis risk",    "discontinue"),
    Interaction("simvastatin", "fluconazole",  "major",          "Fluconazole inhibits CYP3A4, increasing simvastatin levels and myopathy risk",                       "substitute"),
    Interaction("simvastatin", "amlodipine",   "moderate",       "Amlodipine increases simvastatin levels; limit simvastatin to 20mg/day",                             "adjust_dose"),
    Interaction("atorvastatin", "erythromycin","major",          "Erythromycin increases atorvastatin levels via CYP3A4 inhibition",                                    "adjust_dose"),
    Interaction("atorvastatin", "fluconazole", "major",          "Fluconazole inhibits CYP3A4, increasing statin levels and myopathy risk",                            "adjust_dose"),
    Interaction("simvastatin", "carbamazepine","moderate",       "Carbamazepine induces CYP3A4, reducing statin effectiveness",                                        "adjust_dose"),
    Interaction("atorvastatin", "rifampin",    "moderate",       "Rifampin induces CYP3A4, reducing atorvastatin levels",                                              "adjust_dose"),

    # --- SSRI / Serotonin interactions ---
    Interaction("fluoxetine", "tramadol",     "contraindicated", "High risk of serotonin syndrome due to combined serotonergic activity",                               "discontinue"),
    Interaction("sertraline", "tramadol",     "major",           "Risk of serotonin syndrome with combined serotonergic drugs",                                         "discontinue"),
    Interaction("fluoxetine", "lithium",      "moderate",        "SSRIs may increase lithium levels and serotonin syndrome risk",                                       "monitor"),
    Interaction("sertraline", "lithium",      "moderate",        "SSRIs may increase lithium levels and serotonin syndrome risk",                                       "monitor"),
    Interaction("fluoxetine", "warfarin",     "moderate",        "Fluoxetine inhibits CYP2C9, potentially increasing warfarin levels",                                  "monitor"),
    Interaction("fluoxetine", "carbamazepine","moderate",        "Fluoxetine inhibits carbamazepine metabolism, increasing toxicity risk",                               "monitor"),
    Interaction("fluoxetine", "phenytoin",    "moderate",        "Fluoxetine inhibits CYP2C9, increasing phenytoin levels",                                            "monitor"),
    Interaction("fluoxetine", "diazepam",     "moderate",        "Fluoxetine inhibits CYP2C19/3A4, increasing diazepam levels and sedation",                           "adjust_dose"),
    Interaction("sertraline", "diazepam",     "minor",           "Sertraline may slightly increase diazepam levels",                                                   "monitor"),

    # --- Anticonvulsant interactions ---
    Interaction("carbamazepine", "phenytoin",    "moderate",     "Mutual enzyme induction alters levels of both drugs unpredictably",                                   "monitor"),
    Interaction("carbamazepine", "warfarin",     "major",        "Carbamazepine induces CYP enzymes, reducing warfarin effectiveness",                                 "adjust_dose"),
    Interaction("carbamazepine", "erythromycin", "major",        "Erythromycin inhibits carbamazepine metabolism, causing toxicity",                                    "substitute"),
    Interaction("phenytoin",     "warfarin",     "major",        "Complex bidirectional interaction affecting levels of both drugs",                                    "monitor"),
    Interaction("phenytoin",     "fluconazole",  "major",        "Fluconazole inhibits CYP2C9, increasing phenytoin to toxic levels",                                  "adjust_dose"),
    Interaction("carbamazepine", "fluconazole",  "major",        "Fluconazole inhibits carbamazepine metabolism, increasing toxicity risk",                             "adjust_dose"),
    Interaction("phenytoin",     "omeprazole",   "moderate",     "Omeprazole may increase phenytoin levels via CYP2C19 inhibition",                                    "monitor"),

    # --- Quinolone interactions ---
    Interaction("ciprofloxacin", "warfarin",     "major",        "Ciprofloxacin inhibits warfarin metabolism, increasing bleeding risk",                                "monitor"),
    Interaction("ciprofloxacin", "theophylline", "major",        "Ciprofloxacin inhibits theophylline metabolism, risk of toxicity",                                    "adjust_dose"),
    Interaction("ciprofloxacin", "metformin",    "moderate",     "Ciprofloxacin may alter blood glucose levels with metformin",                                        "monitor"),
    Interaction("ciprofloxacin", "phenytoin",    "moderate",     "Ciprofloxacin may alter phenytoin levels",                                                           "monitor"),
    Interaction("ciprofloxacin", "diazepam",     "minor",        "Ciprofloxacin may slightly increase diazepam levels",                                                "monitor"),

    # --- Metformin interactions ---
    Interaction("metformin", "ibuprofen",     "moderate",        "NSAIDs may impair renal function, increasing metformin accumulation and lactic acidosis risk",        "monitor"),
    Interaction("metformin", "fluconazole",   "minor",           "Fluconazole may slightly increase metformin levels",                                                  "monitor"),

    # --- Beta-blocker interactions ---
    Interaction("metoprolol", "fluoxetine",   "moderate",        "Fluoxetine inhibits CYP2D6, increasing metoprolol levels and bradycardia risk",                      "adjust_dose"),
    Interaction("metoprolol", "verapamil",    "major",           "Combined use may cause severe bradycardia and heart block",                                           "discontinue"),
    Interaction("metoprolol", "diazepam",     "minor",           "Minor additive CNS depression",                                                                       "monitor"),

    # --- Thyroid interactions ---
    Interaction("levothyroxine", "omeprazole",   "moderate",     "PPIs reduce gastric acid, impairing levothyroxine absorption",                                       "adjust_dose"),
    Interaction("levothyroxine", "carbamazepine","moderate",     "Carbamazepine increases levothyroxine metabolism",                                                    "adjust_dose"),
    Interaction("levothyroxine", "warfarin",     "moderate",     "Levothyroxine may enhance warfarin's anticoagulant effect",                                          "monitor"),

    # --- Lithium interactions ---
    Interaction("lithium", "ibuprofen",      "major",            "NSAIDs reduce lithium excretion, causing toxicity",                                                   "substitute"),
    Interaction("lithium", "naproxen",       "major",            "NSAIDs reduce lithium excretion, causing toxicity",                                                   "substitute"),
    Interaction("lithium", "lisinopril",     "major",            "ACE inhibitors reduce lithium excretion, increasing toxicity risk",                                   "monitor"),
    Interaction("lithium", "losartan",       "major",            "ARBs may reduce lithium excretion, increasing toxicity risk",                                         "monitor"),
    Interaction("lithium", "metformin",      "minor",            "Minor potential for altered lithium levels",                                                           "monitor"),

    # --- Sulfonylurea interactions ---
    Interaction("glipizide", "fluconazole",  "major",            "Fluconazole inhibits CYP2C9, increasing glipizide levels and hypoglycemia risk",                     "adjust_dose"),
    Interaction("glipizide", "ciprofloxacin","moderate",         "Ciprofloxacin may potentiate hypoglycemic effect of sulfonylureas",                                   "monitor"),
    Interaction("glipizide", "rifampin",     "moderate",         "Rifampin induces CYP2C9, reducing glipizide effectiveness",                                          "adjust_dose"),
]

# Build a fast lookup: frozenset({drug_a, drug_b}) -> Interaction
_INTERACTION_INDEX: Dict[frozenset, Interaction] = {}
for _ix in INTERACTIONS:
    _key = frozenset({_ix.drug_a.lower(), _ix.drug_b.lower()})
    _INTERACTION_INDEX[_key] = _ix


def lookup_interaction(drug_a: str, drug_b: str) -> Optional[Interaction]:
    """Check if two drugs interact. Returns Interaction or None."""
    return _INTERACTION_INDEX.get(frozenset({drug_a.lower(), drug_b.lower()}))


def get_all_interactions_for_drugs(drug_names: List[str]) -> List[Interaction]:
    """Find all pairwise interactions among a list of drugs."""
    results = []
    names = [d.lower() for d in drug_names]
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            ix = lookup_interaction(names[i], names[j])
            if ix is not None:
                results.append(ix)
    return results


# ---------------------------------------------------------------------------
# Condition-to-drug mapping for realistic scenarios
# ---------------------------------------------------------------------------

CONDITION_DRUGS: Dict[str, List[str]] = {
    "hypertension":       ["lisinopril", "losartan", "amlodipine", "metoprolol"],
    "atrial_fibrillation": ["warfarin", "metoprolol"],
    "DVT":                ["warfarin", "heparin"],
    "coronary_artery_disease": ["aspirin", "clopidogrel", "atorvastatin", "metoprolol"],
    "hyperlipidemia":     ["atorvastatin", "simvastatin"],
    "type2_diabetes":     ["metformin", "glipizide"],
    "depression":         ["fluoxetine", "sertraline"],
    "anxiety":            ["diazepam", "sertraline"],
    "chronic_pain":       ["tramadol", "ibuprofen", "naproxen", "acetaminophen"],
    "epilepsy":           ["carbamazepine", "phenytoin"],
    "bipolar_disorder":   ["lithium", "carbamazepine"],
    "hypothyroidism":     ["levothyroxine"],
    "GERD":               ["omeprazole"],
    "heart_failure":      ["lisinopril", "metoprolol", "spironolactone"],
    "infection":          ["ciprofloxacin", "metronidazole", "erythromycin"],
    "fungal_infection":   ["fluconazole"],
    "tuberculosis":       ["rifampin"],
    "osteoarthritis":     ["ibuprofen", "naproxen", "acetaminophen"],
    "rheumatoid_arthritis": ["ibuprofen", "naproxen"],
    "edema":              ["spironolactone"],
}

# ---------------------------------------------------------------------------
# Scenario generators
# ---------------------------------------------------------------------------

SEVERITY_ORDER = {"minor": 1, "moderate": 2, "major": 3, "contraindicated": 4}


def _pick_conditions_and_drugs(
    rng: random.Random,
    target_drug_count: int,
    min_interactions: int,
) -> Tuple[List[str], List[Drug], List[Interaction]]:
    """Build a clinically plausible medication list that guarantees interactions."""
    all_conditions = list(CONDITION_DRUGS.keys())

    for _attempt in range(200):
        rng.shuffle(all_conditions)
        chosen_conditions = []
        chosen_drug_names = set()

        for cond in all_conditions:
            candidates = CONDITION_DRUGS[cond]
            new_drugs = [d for d in candidates if d not in chosen_drug_names]
            if not new_drugs:
                continue
            pick = rng.choice(new_drugs)
            chosen_drug_names.add(pick)
            chosen_conditions.append(cond)
            if len(chosen_drug_names) >= target_drug_count:
                break

        # Pad if needed
        remaining = [d for d in DRUGS if d not in chosen_drug_names]
        while len(chosen_drug_names) < target_drug_count and remaining:
            extra = rng.choice(remaining)
            remaining.remove(extra)
            chosen_drug_names.add(extra)

        drug_list = [DRUGS[n] for n in chosen_drug_names if n in DRUGS]
        interactions = get_all_interactions_for_drugs(list(chosen_drug_names))

        if len(interactions) >= min_interactions:
            return chosen_conditions[:5], drug_list, interactions

    # Fallback: force known interacting set
    fallback_drugs = ["warfarin", "aspirin", "ibuprofen", "fluoxetine", "tramadol",
                      "lisinopril", "spironolactone", "simvastatin"][:target_drug_count]
    drug_list = [DRUGS[n] for n in fallback_drugs]
    interactions = get_all_interactions_for_drugs(fallback_drugs)
    return ["coronary_artery_disease", "chronic_pain", "depression"], drug_list, interactions


def generate_easy_scenario(seed: Optional[int] = None) -> PatientScenario:
    """Task 1 - Easy: 2-drug pair interaction check."""
    rng = random.Random(seed)

    # Pick a known interacting pair
    ix = rng.choice(INTERACTIONS)
    drug_a = DRUGS[ix.drug_a]
    drug_b = DRUGS[ix.drug_b]

    # Pick relevant conditions
    conditions = set()
    for cond, cond_drugs in CONDITION_DRUGS.items():
        if ix.drug_a in cond_drugs or ix.drug_b in cond_drugs:
            conditions.add(cond)
    conditions = list(conditions)[:2] or ["general_checkup"]

    age = rng.randint(40, 80)

    return PatientScenario(
        patient_id=f"PT-EASY-{seed or rng.randint(1000,9999)}",
        age=age,
        conditions=conditions,
        medications=[drug_a, drug_b],
        ground_truth_interactions=[ix],
        task_name="easy_pair_check",
        task_difficulty="easy",
    )


def generate_medium_scenario(seed: Optional[int] = None) -> PatientScenario:
    """Task 2 - Medium: 5-drug patient, find all interacting pairs + severity."""
    rng = random.Random(seed)
    conditions, drugs, interactions = _pick_conditions_and_drugs(rng, 5, 2)
    age = rng.randint(45, 75)

    return PatientScenario(
        patient_id=f"PT-MED-{seed or rng.randint(1000,9999)}",
        age=age,
        conditions=conditions,
        medications=drugs,
        ground_truth_interactions=interactions,
        task_name="medium_multi_drug",
        task_difficulty="medium",
    )


def generate_hard_scenario(seed: Optional[int] = None) -> PatientScenario:
    """Task 3 - Hard: 8-drug polypharmacy, find interactions + severity + recommendations."""
    rng = random.Random(seed)
    conditions, drugs, interactions = _pick_conditions_and_drugs(rng, 8, 4)
    age = rng.randint(60, 85)

    return PatientScenario(
        patient_id=f"PT-HARD-{seed or rng.randint(1000,9999)}",
        age=age,
        conditions=conditions,
        medications=drugs,
        ground_truth_interactions=interactions,
        task_name="hard_polypharmacy",
        task_difficulty="hard",
    )


# ---------------------------------------------------------------------------
# Task definitions for openenv.yaml
# ---------------------------------------------------------------------------

TASK_DEFINITIONS = {
    "easy_pair_check": {
        "description": "Given 2 medications, identify if a drug interaction exists and classify its severity.",
        "difficulty": "easy",
        "generator": generate_easy_scenario,
    },
    "medium_multi_drug": {
        "description": "Given 5 medications, find ALL interacting drug pairs and classify each severity level.",
        "difficulty": "medium",
        "generator": generate_medium_scenario,
    },
    "hard_polypharmacy": {
        "description": "Given 8 medications (polypharmacy), find all interactions, classify severity, and recommend clinical actions.",
        "difficulty": "hard",
        "generator": generate_hard_scenario,
    },
}


def drug_to_dict(drug: Drug) -> dict:
    return asdict(drug)


def interaction_to_dict(ix: Interaction) -> dict:
    return asdict(ix)
