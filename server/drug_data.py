"""
PharmaDDI Drug Knowledge Base — Enhanced with DDI-Bench inspired data.
Uses a larger drug database with distribution-change-aware scenario generation
and adaptive curriculum support.
"""

import random
import json
import os
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional


@dataclass
class Drug:
    name: str
    therapeutic_class: str
    common_dose: str
    frequency: str
    cyp_inhibitor: List[str] = field(default_factory=list)   # e.g. ["CYP2C9", "CYP3A4"]
    cyp_substrate: List[str] = field(default_factory=list)   # e.g. ["CYP2D6"]
    narrow_therapeutic_index: bool = False


@dataclass
class Interaction:
    drug_a: str
    drug_b: str
    severity: str          # minor | moderate | major | contraindicated
    clinical_effect: str
    recommendation: str    # monitor | adjust_dose | substitute | discontinue
    mechanism: str = ""    # pharmacokinetic | pharmacodynamic | unknown
    evidence_level: str = "established"  # established | probable | suspected


@dataclass
class PatientScenario:
    patient_id: str
    age: int
    conditions: List[str]
    medications: List[Drug]
    ground_truth_interactions: List[Interaction]
    task_name: str
    task_difficulty: str
    curriculum_focus: str = "general"   # which drug class this scenario targets


SEVERITY_ORDER = {"minor": 1, "moderate": 2, "major": 3, "contraindicated": 4}


# ---------------------------------------------------------------------------
# Expanded Drug Database (60+ drugs across all major classes)
# Inspired by DrugBank coverage in DDI-Bench
# ---------------------------------------------------------------------------

DRUGS: Dict[str, Drug] = {d.name: d for d in [
    # Anticoagulants / Antiplatelets
    Drug("warfarin",       "anticoagulant",       "5mg",    "once daily",      cyp_inhibitor=[], cyp_substrate=["CYP2C9","CYP3A4"], narrow_therapeutic_index=True),
    Drug("aspirin",        "antiplatelet",        "81mg",   "once daily"),
    Drug("clopidogrel",    "antiplatelet",        "75mg",   "once daily",      cyp_substrate=["CYP2C19"]),
    Drug("heparin",        "anticoagulant",       "5000U",  "twice daily"),
    Drug("apixaban",       "anticoagulant",       "5mg",    "twice daily",     cyp_substrate=["CYP3A4"]),
    Drug("rivaroxaban",    "anticoagulant",       "20mg",   "once daily",      cyp_substrate=["CYP3A4"]),
    Drug("dabigatran",     "anticoagulant",       "150mg",  "twice daily"),
    Drug("ticagrelor",     "antiplatelet",        "90mg",   "twice daily",     cyp_substrate=["CYP3A4"]),

    # Cardiovascular
    Drug("lisinopril",     "ACE_inhibitor",       "10mg",   "once daily"),
    Drug("enalapril",      "ACE_inhibitor",       "10mg",   "twice daily"),
    Drug("losartan",       "ARB",                 "50mg",   "once daily",      cyp_substrate=["CYP2C9"]),
    Drug("valsartan",      "ARB",                 "80mg",   "once daily"),
    Drug("amlodipine",     "calcium_blocker",     "5mg",    "once daily",      cyp_inhibitor=["CYP3A4"], cyp_substrate=["CYP3A4"]),
    Drug("diltiazem",      "calcium_blocker",     "120mg",  "twice daily",     cyp_inhibitor=["CYP3A4"]),
    Drug("verapamil",      "calcium_blocker",     "120mg",  "three times daily",cyp_inhibitor=["CYP3A4"]),
    Drug("metoprolol",     "beta_blocker",        "50mg",   "twice daily",     cyp_substrate=["CYP2D6"]),
    Drug("atenolol",       "beta_blocker",        "50mg",   "once daily"),
    Drug("carvedilol",     "beta_blocker",        "12.5mg", "twice daily",     cyp_substrate=["CYP2D6"]),
    Drug("atorvastatin",   "statin",              "20mg",   "once daily",      cyp_substrate=["CYP3A4"]),
    Drug("simvastatin",    "statin",              "40mg",   "once daily",      cyp_substrate=["CYP3A4"]),
    Drug("rosuvastatin",   "statin",              "10mg",   "once daily"),
    Drug("digoxin",        "cardiac_glycoside",   "0.125mg","once daily",      narrow_therapeutic_index=True),
    Drug("amiodarone",     "antiarrhythmic",      "200mg",  "once daily",      cyp_inhibitor=["CYP2C9","CYP3A4","CYP2D6"], narrow_therapeutic_index=True),

    # CNS / Psychiatry
    Drug("fluoxetine",     "SSRI",                "20mg",   "once daily",      cyp_inhibitor=["CYP2D6","CYP2C9"]),
    Drug("sertraline",     "SSRI",                "50mg",   "once daily",      cyp_inhibitor=["CYP2D6"]),
    Drug("paroxetine",     "SSRI",                "20mg",   "once daily",      cyp_inhibitor=["CYP2D6"]),
    Drug("escitalopram",   "SSRI",                "10mg",   "once daily"),
    Drug("venlafaxine",    "SNRI",                "75mg",   "once daily",      cyp_substrate=["CYP2D6"]),
    Drug("duloxetine",     "SNRI",                "60mg",   "once daily",      cyp_inhibitor=["CYP2D6"]),
    Drug("diazepam",       "benzodiazepine",      "5mg",    "twice daily",     cyp_substrate=["CYP2C19","CYP3A4"]),
    Drug("alprazolam",     "benzodiazepine",      "0.5mg",  "three times daily",cyp_substrate=["CYP3A4"]),
    Drug("tramadol",       "opioid_analgesic",    "50mg",   "every 6 hours",   cyp_substrate=["CYP2D6","CYP3A4"]),
    Drug("oxycodone",      "opioid_analgesic",    "10mg",   "every 4-6 hours", cyp_substrate=["CYP3A4","CYP2D6"]),
    Drug("lithium",        "mood_stabilizer",     "300mg",  "twice daily",     narrow_therapeutic_index=True),
    Drug("valproate",      "anticonvulsant",      "500mg",  "twice daily",     cyp_inhibitor=["CYP2C9"], narrow_therapeutic_index=True),
    Drug("carbamazepine",  "anticonvulsant",      "200mg",  "twice daily",     cyp_inhibitor=["CYP3A4"], cyp_substrate=["CYP3A4"]),
    Drug("phenytoin",      "anticonvulsant",      "100mg",  "three times daily",cyp_inhibitor=["CYP2C9"], cyp_substrate=["CYP2C9"], narrow_therapeutic_index=True),
    Drug("lamotrigine",    "anticonvulsant",      "100mg",  "twice daily"),
    Drug("quetiapine",     "antipsychotic",       "200mg",  "twice daily",     cyp_substrate=["CYP3A4"]),
    Drug("haloperidol",    "antipsychotic",       "5mg",    "twice daily",     cyp_substrate=["CYP2D6","CYP3A4"]),

    # Anti-infectives
    Drug("ciprofloxacin",  "fluoroquinolone",     "500mg",  "twice daily",     cyp_inhibitor=["CYP1A2"]),
    Drug("levofloxacin",   "fluoroquinolone",     "500mg",  "once daily"),
    Drug("metronidazole",  "antibiotic",          "500mg",  "three times daily",cyp_inhibitor=["CYP2C9"]),
    Drug("fluconazole",    "antifungal",          "150mg",  "once daily",      cyp_inhibitor=["CYP2C9","CYP3A4"]),
    Drug("itraconazole",   "antifungal",          "200mg",  "once daily",      cyp_inhibitor=["CYP3A4"]),
    Drug("rifampin",       "antibiotic",          "600mg",  "once daily",      cyp_inhibitor=["CYP3A4","CYP2C9"]),  # strong inducer
    Drug("erythromycin",   "macrolide",           "500mg",  "four times daily", cyp_inhibitor=["CYP3A4"]),
    Drug("clarithromycin", "macrolide",           "500mg",  "twice daily",     cyp_inhibitor=["CYP3A4"]),

    # Analgesics / Anti-inflammatory
    Drug("ibuprofen",      "NSAID",               "400mg",  "every 6 hours"),
    Drug("naproxen",       "NSAID",               "250mg",  "twice daily"),
    Drug("celecoxib",      "COX2_inhibitor",      "200mg",  "once daily",      cyp_substrate=["CYP2C9"]),
    Drug("acetaminophen",  "analgesic",           "500mg",  "every 6 hours"),

    # Diabetes
    Drug("metformin",      "biguanide",           "500mg",  "twice daily"),
    Drug("glipizide",      "sulfonylurea",        "5mg",    "once daily",      cyp_substrate=["CYP2C9"]),
    Drug("sitagliptin",    "DPP4_inhibitor",      "100mg",  "once daily"),
    Drug("insulin_glargine","insulin",            "10U",    "once daily at bedtime"),

    # Other
    Drug("omeprazole",     "PPI",                 "20mg",   "once daily",      cyp_inhibitor=["CYP2C19"]),
    Drug("pantoprazole",   "PPI",                 "40mg",   "once daily"),
    Drug("levothyroxine",  "thyroid_hormone",     "50mcg",  "once daily"),
    Drug("spironolactone", "K_sparing_diuretic",  "25mg",   "once daily"),
    Drug("furosemide",     "loop_diuretic",       "40mg",   "once daily"),
    Drug("allopurinol",    "xanthine_oxidase_inh","300mg",  "once daily"),
    Drug("colchicine",     "antigout",            "0.6mg",  "twice daily",     cyp_substrate=["CYP3A4"]),
    Drug("theophylline",   "bronchodilator",      "300mg",  "twice daily",     narrow_therapeutic_index=True),
    Drug("tacrolimus",     "immunosuppressant",   "1mg",    "twice daily",     cyp_substrate=["CYP3A4"], narrow_therapeutic_index=True),
    Drug("cyclosporine",   "immunosuppressant",   "100mg",  "twice daily",     cyp_inhibitor=["CYP3A4"], cyp_substrate=["CYP3A4"], narrow_therapeutic_index=True),
]}


# ---------------------------------------------------------------------------
# Expanded Interaction Database (120+ pairs)
# ---------------------------------------------------------------------------

INTERACTIONS: List[Interaction] = [
    # === ANTICOAGULANT INTERACTIONS ===
    Interaction("warfarin","aspirin",        "major",           "Increased bleeding risk via antiplatelet + anticoagulant combination",             "monitor",     "pharmacodynamic"),
    Interaction("warfarin","ibuprofen",      "major",           "NSAIDs inhibit platelet function and increase GI bleeding with warfarin",          "substitute",  "pharmacodynamic"),
    Interaction("warfarin","naproxen",       "major",           "NSAIDs increase bleeding risk with anticoagulants",                               "substitute",  "pharmacodynamic"),
    Interaction("warfarin","clopidogrel",    "major",           "Dual antithrombotic therapy significantly increases hemorrhage risk",              "monitor",     "pharmacodynamic"),
    Interaction("warfarin","fluconazole",    "major",           "Fluconazole inhibits CYP2C9, dramatically raising warfarin levels",               "adjust_dose", "pharmacokinetic"),
    Interaction("warfarin","metronidazole",  "major",           "Metronidazole inhibits CYP2C9, increasing INR and bleeding risk",                 "adjust_dose", "pharmacokinetic"),
    Interaction("warfarin","rifampin",       "major",           "Rifampin strongly induces CYP2C9/3A4, drastically reducing warfarin effect",      "adjust_dose", "pharmacokinetic"),
    Interaction("warfarin","amiodarone",     "major",           "Amiodarone inhibits CYP2C9 and CYP3A4, markedly potentiating warfarin",           "adjust_dose", "pharmacokinetic"),
    Interaction("warfarin","erythromycin",   "moderate",        "Erythromycin inhibits CYP3A4, moderately increasing warfarin exposure",           "monitor",     "pharmacokinetic"),
    Interaction("warfarin","clarithromycin", "major",           "Clarithromycin inhibits CYP3A4, significantly raising warfarin levels",           "adjust_dose", "pharmacokinetic"),
    Interaction("warfarin","omeprazole",     "minor",           "Omeprazole may slightly increase warfarin via CYP2C19 inhibition",                "monitor",     "pharmacokinetic"),
    Interaction("warfarin","simvastatin",    "moderate",        "Simvastatin may enhance anticoagulant effect of warfarin",                        "monitor",     "pharmacokinetic"),
    Interaction("warfarin","valproate",      "moderate",        "Valproate inhibits CYP2C9, potentially increasing warfarin levels",               "monitor",     "pharmacokinetic"),
    Interaction("warfarin","phenytoin",      "major",           "Complex bidirectional interaction — phenytoin both inhibits and induces warfarin metabolism", "monitor", "pharmacokinetic"),
    Interaction("warfarin","ciprofloxacin",  "major",           "Ciprofloxacin inhibits CYP1A2 and gut flora, raising INR",                       "monitor",     "pharmacokinetic"),
    Interaction("warfarin","levothyroxine",  "moderate",        "Levothyroxine increases catabolism of clotting factors, enhancing warfarin effect","monitor",    "pharmacodynamic"),
    Interaction("heparin", "aspirin",        "major",           "Combined anticoagulant + antiplatelet therapy increases serious bleeding risk",    "monitor",     "pharmacodynamic"),
    Interaction("apixaban","ibuprofen",      "major",           "NSAIDs combined with DOACs markedly raise GI bleeding risk",                      "substitute",  "pharmacodynamic"),
    Interaction("rivaroxaban","aspirin",     "major",           "Combined anticoagulant and antiplatelet increases hemorrhage risk",                "monitor",     "pharmacodynamic"),
    Interaction("aspirin", "ibuprofen",      "moderate",        "Ibuprofen competes with aspirin for COX-1 binding, reducing cardioprotection",    "substitute",  "pharmacodynamic"),
    Interaction("aspirin", "naproxen",       "moderate",        "NSAIDs may interfere with aspirin's antiplatelet effect",                         "substitute",  "pharmacodynamic"),
    Interaction("clopidogrel","omeprazole",  "moderate",        "Omeprazole inhibits CYP2C19, reducing clopidogrel activation to active metabolite","substitute", "pharmacokinetic"),
    Interaction("ticagrelor","clarithromycin","major",          "Clarithromycin inhibits CYP3A4, raising ticagrelor to potentially toxic levels",  "substitute",  "pharmacokinetic"),

    # === ACE INHIBITOR / ARB / POTASSIUM ===
    Interaction("lisinopril","losartan",     "contraindicated", "Dual RAAS blockade: hyperkalemia, hypotension, and acute kidney injury risk",     "discontinue", "pharmacodynamic"),
    Interaction("lisinopril","spironolactone","major",          "Combined potassium retention causes dangerous hyperkalemia",                       "monitor",     "pharmacodynamic"),
    Interaction("losartan",  "spironolactone","major",          "ARB + potassium-sparing diuretic: severe hyperkalemia risk",                      "monitor",     "pharmacodynamic"),
    Interaction("lisinopril","ibuprofen",    "moderate",        "NSAIDs reduce renal prostaglandins, blunting ACE inhibitor effect and causing AKI","substitute",  "pharmacodynamic"),
    Interaction("lisinopril","naproxen",     "moderate",        "NSAIDs antagonize ACE inhibitor antihypertensive effect",                         "substitute",  "pharmacodynamic"),
    Interaction("losartan",  "ibuprofen",    "moderate",        "NSAIDs reduce ARB effectiveness and increase renal failure risk",                 "substitute",  "pharmacodynamic"),
    Interaction("lisinopril","lithium",      "major",           "ACE inhibitors reduce lithium renal excretion, causing lithium toxicity",         "monitor",     "pharmacokinetic"),
    Interaction("losartan",  "lithium",      "major",           "ARBs reduce lithium clearance, increasing toxicity risk",                         "monitor",     "pharmacokinetic"),

    # === STATIN INTERACTIONS ===
    Interaction("simvastatin","erythromycin","contraindicated", "Erythromycin inhibits CYP3A4, causing simvastatin accumulation and rhabdomyolysis","discontinue", "pharmacokinetic"),
    Interaction("simvastatin","clarithromycin","contraindicated","Clarithromycin inhibits CYP3A4, dramatically raising simvastatin to toxic levels","discontinue", "pharmacokinetic"),
    Interaction("simvastatin","fluconazole",  "major",          "Fluconazole inhibits CYP3A4, increasing simvastatin myopathy risk",               "substitute",  "pharmacokinetic"),
    Interaction("simvastatin","amlodipine",   "moderate",       "Amlodipine raises simvastatin AUC — limit simvastatin to 20mg/day",               "adjust_dose", "pharmacokinetic"),
    Interaction("simvastatin","itraconazole", "contraindicated","Itraconazole inhibits CYP3A4, causing life-threatening statin toxicity",          "discontinue", "pharmacokinetic"),
    Interaction("simvastatin","amiodarone",   "major",          "Amiodarone inhibits CYP3A4 and CYP2C9, raising simvastatin to toxic levels",      "adjust_dose", "pharmacokinetic"),
    Interaction("simvastatin","diltiazem",    "moderate",       "Diltiazem inhibits CYP3A4, raising simvastatin levels",                           "adjust_dose", "pharmacokinetic"),
    Interaction("simvastatin","verapamil",    "moderate",       "Verapamil inhibits CYP3A4, increasing simvastatin exposure",                      "adjust_dose", "pharmacokinetic"),
    Interaction("simvastatin","carbamazepine","moderate",       "Carbamazepine induces CYP3A4, reducing statin effectiveness",                     "adjust_dose", "pharmacokinetic"),
    Interaction("atorvastatin","erythromycin","major",          "Erythromycin raises atorvastatin levels via CYP3A4 inhibition",                   "adjust_dose", "pharmacokinetic"),
    Interaction("atorvastatin","clarithromycin","major",        "Clarithromycin inhibits CYP3A4, significantly raising atorvastatin",              "adjust_dose", "pharmacokinetic"),
    Interaction("atorvastatin","fluconazole", "major",          "Fluconazole inhibits CYP3A4, increasing atorvastatin myopathy risk",              "adjust_dose", "pharmacokinetic"),
    Interaction("atorvastatin","rifampin",    "moderate",       "Rifampin induces CYP3A4, dramatically reducing atorvastatin levels",              "adjust_dose", "pharmacokinetic"),
    Interaction("atorvastatin","cyclosporine","contraindicated","Cyclosporine inhibits drug transporters, causing dangerous statin accumulation",   "discontinue", "pharmacokinetic"),
    Interaction("tacrolimus","fluconazole",   "major",          "Fluconazole inhibits CYP3A4, raising tacrolimus to nephrotoxic levels",           "adjust_dose", "pharmacokinetic"),
    Interaction("tacrolimus","clarithromycin","major",          "Clarithromycin inhibits CYP3A4, markedly raising tacrolimus concentrations",      "adjust_dose", "pharmacokinetic"),
    Interaction("cyclosporine","simvastatin", "contraindicated","Cyclosporine raises simvastatin to rhabdomyolysis-inducing concentrations",       "discontinue", "pharmacokinetic"),

    # === SEROTONIN / CNS ===
    Interaction("fluoxetine","tramadol",     "contraindicated", "High serotonin syndrome risk from dual serotonergic activity",                    "discontinue", "pharmacodynamic"),
    Interaction("sertraline","tramadol",     "major",           "Combined serotonergic drugs cause serotonin syndrome",                            "discontinue", "pharmacodynamic"),
    Interaction("paroxetine","tramadol",     "major",           "Paroxetine inhibits CYP2D6 AND has serotonergic activity — dual risk",            "discontinue", "pharmacodynamic"),
    Interaction("venlafaxine","tramadol",    "major",           "SNRI + opioid analgesic: significant serotonin syndrome risk",                    "substitute",  "pharmacodynamic"),
    Interaction("fluoxetine","lithium",      "moderate",        "SSRIs may increase lithium levels and serotonin syndrome risk",                   "monitor",     "pharmacodynamic"),
    Interaction("sertraline","lithium",      "moderate",        "SSRIs + lithium increases serotonin syndrome risk",                               "monitor",     "pharmacodynamic"),
    Interaction("fluoxetine","carbamazepine","moderate",        "Fluoxetine inhibits CYP2D6, raising carbamazepine toxicity risk",                 "monitor",     "pharmacokinetic"),
    Interaction("fluoxetine","phenytoin",    "moderate",        "Fluoxetine inhibits CYP2C9, increasing phenytoin to toxic levels",                "monitor",     "pharmacokinetic"),
    Interaction("fluoxetine","diazepam",     "moderate",        "Fluoxetine inhibits CYP2C19/3A4, raising diazepam levels and sedation",           "adjust_dose", "pharmacokinetic"),
    Interaction("fluoxetine","metoprolol",   "moderate",        "Fluoxetine inhibits CYP2D6, raising metoprolol levels and bradycardia risk",      "adjust_dose", "pharmacokinetic"),
    Interaction("paroxetine","metoprolol",   "major",           "Paroxetine inhibits CYP2D6, causing metoprolol accumulation and bradycardia",     "adjust_dose", "pharmacokinetic"),
    Interaction("duloxetine","tramadol",     "major",           "SNRI + tramadol: serotonin syndrome and seizure risk",                            "discontinue", "pharmacodynamic"),
    Interaction("sertraline","diazepam",     "minor",           "Sertraline may mildly increase diazepam levels",                                  "monitor",     "pharmacokinetic"),
    Interaction("quetiapine","diazepam",     "moderate",        "Combined CNS depressants cause additive sedation and respiratory depression",      "monitor",     "pharmacodynamic"),
    Interaction("haloperidol","carbamazepine","moderate",       "Carbamazepine induces CYP3A4, reducing haloperidol plasma levels",                "adjust_dose", "pharmacokinetic"),

    # === ANTICONVULSANTS ===
    Interaction("carbamazepine","phenytoin",  "moderate",       "Mutual enzyme induction alters both drug levels unpredictably",                   "monitor",     "pharmacokinetic"),
    Interaction("carbamazepine","warfarin",   "major",          "Carbamazepine induces CYP enzymes, reducing warfarin effectiveness",              "adjust_dose", "pharmacokinetic"),
    Interaction("carbamazepine","erythromycin","major",         "Erythromycin inhibits carbamazepine metabolism, causing carbamazepine toxicity",  "substitute",  "pharmacokinetic"),
    Interaction("carbamazepine","clarithromycin","major",       "Clarithromycin inhibits CYP3A4, causing carbamazepine toxicity",                  "substitute",  "pharmacokinetic"),
    Interaction("phenytoin",   "fluconazole", "major",          "Fluconazole inhibits CYP2C9, raising phenytoin to toxic levels",                  "adjust_dose", "pharmacokinetic"),
    Interaction("phenytoin",   "omeprazole",  "moderate",       "Omeprazole inhibits CYP2C19, raising phenytoin levels",                           "monitor",     "pharmacokinetic"),
    Interaction("carbamazepine","fluconazole","major",          "Fluconazole inhibits CYP3A4, increasing carbamazepine toxicity risk",             "adjust_dose", "pharmacokinetic"),
    Interaction("valproate",   "lamotrigine", "moderate",       "Valproate inhibits lamotrigine glucuronidation, doubling lamotrigine levels",     "adjust_dose", "pharmacokinetic"),
    Interaction("valproate",   "phenytoin",   "moderate",       "Valproate displaces phenytoin from protein binding, altering free drug levels",  "monitor",     "pharmacokinetic"),

    # === FLUOROQUINOLONES ===
    Interaction("ciprofloxacin","theophylline","major",         "Ciprofloxacin inhibits CYP1A2, causing theophylline toxicity",                    "adjust_dose", "pharmacokinetic"),
    Interaction("ciprofloxacin","metformin",   "moderate",      "Ciprofloxacin alters blood glucose in diabetics on metformin",                    "monitor",     "pharmacodynamic"),
    Interaction("ciprofloxacin","phenytoin",   "moderate",      "Ciprofloxacin may alter phenytoin levels unpredictably",                          "monitor",     "pharmacokinetic"),
    Interaction("ciprofloxacin","diazepam",    "minor",         "Ciprofloxacin may mildly increase diazepam exposure",                            "monitor",     "pharmacokinetic"),
    Interaction("ciprofloxacin","tizanidine",  "contraindicated","Ciprofloxacin markedly raises tizanidine levels causing dangerous hypotension",  "discontinue", "pharmacokinetic"),

    # === LITHIUM ===
    Interaction("lithium",  "ibuprofen",     "major",           "NSAIDs reduce lithium renal excretion, causing lithium toxicity",                 "substitute",  "pharmacokinetic"),
    Interaction("lithium",  "naproxen",      "major",           "NSAIDs reduce lithium excretion — toxicity risk",                                 "substitute",  "pharmacokinetic"),
    Interaction("lithium",  "furosemide",    "major",           "Loop diuretics increase lithium reabsorption, raising lithium to toxic levels",   "monitor",     "pharmacokinetic"),
    Interaction("lithium",  "metformin",     "minor",           "Minor potential for altered lithium levels",                                      "monitor",     "unknown"),
    Interaction("lithium",  "theophylline",  "moderate",        "Theophylline increases renal lithium excretion, reducing lithium effectiveness",  "monitor",     "pharmacokinetic"),

    # === SULFONYLUREAS ===
    Interaction("glipizide","fluconazole",   "major",           "Fluconazole inhibits CYP2C9, raising glipizide — severe hypoglycemia risk",       "adjust_dose", "pharmacokinetic"),
    Interaction("glipizide","ciprofloxacin", "moderate",        "Ciprofloxacin potentiates hypoglycemic effect of sulfonylureas",                  "monitor",     "pharmacodynamic"),
    Interaction("glipizide","rifampin",      "moderate",        "Rifampin induces CYP2C9, reducing glipizide efficacy",                            "adjust_dose", "pharmacokinetic"),

    # === CARDIAC / DIGOXIN ===
    Interaction("digoxin",  "amiodarone",    "major",           "Amiodarone inhibits P-gp and renal clearance, raising digoxin to toxic levels",  "adjust_dose", "pharmacokinetic"),
    Interaction("digoxin",  "clarithromycin","major",           "Clarithromycin inhibits P-gp, raising digoxin to potentially toxic levels",      "adjust_dose", "pharmacokinetic"),
    Interaction("digoxin",  "spironolactone","moderate",        "Spironolactone may raise digoxin levels and cause false digoxin assay readings",  "monitor",     "pharmacokinetic"),
    Interaction("digoxin",  "furosemide",    "moderate",        "Furosemide-induced hypokalemia increases digoxin toxicity risk",                  "monitor",     "pharmacodynamic"),
    Interaction("amiodarone","metoprolol",   "major",           "Both slow conduction — combined use risks severe bradycardia and heart block",    "monitor",     "pharmacodynamic"),
    Interaction("verapamil","metoprolol",    "major",           "Combined calcium blocker + beta blocker causes severe bradycardia and heart block","discontinue", "pharmacodynamic"),
    Interaction("diltiazem","metoprolol",    "moderate",        "Additive negative chronotropy and dromotropy — bradycardia and AV block risk",    "monitor",     "pharmacodynamic"),

    # === THYROID ===
    Interaction("levothyroxine","omeprazole","moderate",        "PPIs raise gastric pH, reducing levothyroxine absorption",                        "adjust_dose", "pharmacokinetic"),
    Interaction("levothyroxine","carbamazepine","moderate",     "Carbamazepine accelerates levothyroxine metabolism via CYP induction",            "adjust_dose", "pharmacokinetic"),
    Interaction("levothyroxine","warfarin",   "moderate",       "Levothyroxine enhances warfarin anticoagulant effect",                            "monitor",     "pharmacodynamic"),

    # === METFORMIN ===
    Interaction("metformin","ibuprofen",     "moderate",        "NSAIDs impair renal function, increasing metformin accumulation and lactic acidosis risk","monitor","pharmacodynamic"),
    Interaction("metformin","fluconazole",   "minor",           "Fluconazole may slightly increase metformin levels",                              "monitor",     "pharmacokinetic"),

    # === COLCHICINE / GOUT ===
    Interaction("colchicine","clarithromycin","contraindicated","Clarithromycin inhibits CYP3A4 and P-gp, causing life-threatening colchicine toxicity","discontinue","pharmacokinetic"),
    Interaction("colchicine","itraconazole", "major",           "Itraconazole inhibits CYP3A4 raising colchicine to potentially fatal levels",    "discontinue", "pharmacokinetic"),
    Interaction("allopurinol","warfarin",    "moderate",        "Allopurinol inhibits warfarin metabolism, raising INR",                          "monitor",     "pharmacokinetic"),
    Interaction("allopurinol","azathioprine","contraindicated", "Allopurinol blocks azathioprine metabolism, causing severe bone marrow suppression","discontinue","pharmacokinetic"),
]

# Build lookup index
_INTERACTION_INDEX: Dict[frozenset, Interaction] = {}
for _ix in INTERACTIONS:
    _key = frozenset({_ix.drug_a.lower(), _ix.drug_b.lower()})
    _INTERACTION_INDEX[_key] = _ix


def lookup_interaction(drug_a: str, drug_b: str) -> Optional[Interaction]:
    return _INTERACTION_INDEX.get(frozenset({drug_a.lower(), drug_b.lower()}))


def get_all_interactions_for_drugs(drug_names: List[str]) -> List[Interaction]:
    results = []
    names = [d.lower() for d in drug_names]
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            ix = lookup_interaction(names[i], names[j])
            if ix is not None:
                results.append(ix)
    return results


# ---------------------------------------------------------------------------
# Drug classes grouped by mechanism — used for curriculum learning
# ---------------------------------------------------------------------------

CURRICULUM_DRUG_GROUPS = {
    "cyp3a4_interactions": ["simvastatin", "atorvastatin", "clarithromycin", "erythromycin",
                             "fluconazole", "itraconazole", "rifampin", "carbamazepine",
                             "amlodipine", "diltiazem", "verapamil", "tacrolimus", "colchicine"],
    "cyp2c9_interactions": ["warfarin", "phenytoin", "glipizide", "fluconazole",
                             "metronidazole", "amiodarone", "valproate", "celecoxib"],
    "cyp2d6_interactions": ["fluoxetine", "paroxetine", "metoprolol", "carvedilol",
                             "tramadol", "oxycodone", "haloperidol"],
    "serotonin_syndrome":  ["fluoxetine", "sertraline", "paroxetine", "venlafaxine",
                             "duloxetine", "tramadol", "lithium"],
    "bleeding_risk":       ["warfarin", "aspirin", "clopidogrel", "heparin", "apixaban",
                             "rivaroxaban", "ibuprofen", "naproxen"],
    "narrow_index_drugs":  ["warfarin", "lithium", "digoxin", "phenytoin", "theophylline",
                             "valproate", "tacrolimus", "cyclosporine"],
    "renal_toxicity":      ["lisinopril", "losartan", "ibuprofen", "naproxen",
                             "metformin", "lithium", "spironolactone"],
}

CONDITION_DRUGS: Dict[str, List[str]] = {
    "hypertension":           ["lisinopril", "losartan", "amlodipine", "metoprolol", "atenolol", "valsartan"],
    "atrial_fibrillation":    ["warfarin", "apixaban", "metoprolol", "digoxin", "amiodarone"],
    "DVT":                    ["warfarin", "heparin", "rivaroxaban", "apixaban"],
    "coronary_artery_disease":["aspirin", "clopidogrel", "atorvastatin", "metoprolol"],
    "heart_failure":          ["lisinopril", "carvedilol", "spironolactone", "furosemide", "digoxin"],
    "hyperlipidemia":         ["atorvastatin", "simvastatin", "rosuvastatin"],
    "type2_diabetes":         ["metformin", "glipizide", "sitagliptin", "insulin_glargine"],
    "depression":             ["fluoxetine", "sertraline", "escitalopram", "venlafaxine", "duloxetine"],
    "anxiety":                ["diazepam", "alprazolam", "sertraline", "escitalopram"],
    "chronic_pain":           ["tramadol", "oxycodone", "ibuprofen", "naproxen", "acetaminophen"],
    "epilepsy":               ["carbamazepine", "phenytoin", "valproate", "lamotrigine"],
    "bipolar_disorder":       ["lithium", "valproate", "quetiapine", "lamotrigine"],
    "hypothyroidism":         ["levothyroxine"],
    "GERD":                   ["omeprazole", "pantoprazole"],
    "infection":              ["ciprofloxacin", "metronidazole", "erythromycin", "clarithromycin", "levofloxacin"],
    "fungal_infection":       ["fluconazole", "itraconazole"],
    "tuberculosis":           ["rifampin"],
    "osteoarthritis":         ["ibuprofen", "naproxen", "celecoxib", "acetaminophen"],
    "rheumatoid_arthritis":   ["ibuprofen", "naproxen", "celecoxib"],
    "edema":                  ["spironolactone", "furosemide"],
    "gout":                   ["allopurinol", "colchicine", "ibuprofen"],
    "arrhythmia":             ["amiodarone", "metoprolol", "digoxin", "verapamil", "diltiazem"],
    "transplant":             ["tacrolimus", "cyclosporine"],
    "psychosis":              ["haloperidol", "quetiapine"],
    "COPD":                   ["theophylline"],
}


def _pick_conditions_and_drugs(
    rng: random.Random,
    target_drug_count: int,
    min_interactions: int,
    curriculum_focus: str = "general",
) -> Tuple[List[str], List[Drug], List[Interaction]]:
    """
    Build a clinically plausible medication list that guarantees interactions.
    When curriculum_focus is set, seeds the list with drugs from that focus group
    to create scenarios targeting a specific weakness.
    """
    all_conditions = list(CONDITION_DRUGS.keys())

    for _attempt in range(300):
        rng_local = random.Random(rng.randint(0, 999999))
        rng_local.shuffle(all_conditions)
        chosen_drug_names = set()

        # If curriculum mode: pre-seed with drugs from the focus group
        if curriculum_focus != "general" and curriculum_focus in CURRICULUM_DRUG_GROUPS:
            focus_drugs = CURRICULUM_DRUG_GROUPS[curriculum_focus]
            available_focus = [d for d in focus_drugs if d in DRUGS]
            # Pick 2–3 drugs from the focus area to ensure the scenario tests that weakness
            seed_count = min(3, len(available_focus), target_drug_count - 1)
            chosen_drug_names.update(rng_local.sample(available_focus, seed_count))

        chosen_conditions = []
        for cond in all_conditions:
            candidates = CONDITION_DRUGS[cond]
            new_drugs = [d for d in candidates if d not in chosen_drug_names and d in DRUGS]
            if not new_drugs:
                continue
            pick = rng_local.choice(new_drugs)
            chosen_drug_names.add(pick)
            chosen_conditions.append(cond)
            if len(chosen_drug_names) >= target_drug_count:
                break

        # Pad with any remaining drugs if needed
        remaining = [d for d in DRUGS if d not in chosen_drug_names]
        rng_local.shuffle(remaining)
        while len(chosen_drug_names) < target_drug_count and remaining:
            chosen_drug_names.add(remaining.pop())

        drug_list = [DRUGS[n] for n in chosen_drug_names if n in DRUGS]
        interactions = get_all_interactions_for_drugs(list(chosen_drug_names))

        if len(interactions) >= min_interactions:
            return chosen_conditions[:5], drug_list, interactions

    # Fallback with known interacting drugs
    fallback = ["warfarin", "aspirin", "simvastatin", "clarithromycin",
                "fluoxetine", "tramadol", "lisinopril", "spironolactone"][:target_drug_count]
    return ["coronary_artery_disease", "depression"], [DRUGS[n] for n in fallback], get_all_interactions_for_drugs(fallback)


def generate_easy_scenario(seed: Optional[int] = None, curriculum_focus: str = "general") -> PatientScenario:
    rng = random.Random(seed)

    # Filter interactions based on curriculum focus
    candidate_interactions = INTERACTIONS
    if curriculum_focus != "general" and curriculum_focus in CURRICULUM_DRUG_GROUPS:
        focus_drugs = set(CURRICULUM_DRUG_GROUPS[curriculum_focus])
        candidate_interactions = [
            ix for ix in INTERACTIONS
            if ix.drug_a in focus_drugs or ix.drug_b in focus_drugs
        ]
    if not candidate_interactions:
        candidate_interactions = INTERACTIONS

    ix = rng.choice(candidate_interactions)
    # Make sure both drugs exist in our database
    if ix.drug_a not in DRUGS or ix.drug_b not in DRUGS:
        ix = rng.choice([i for i in INTERACTIONS if i.drug_a in DRUGS and i.drug_b in DRUGS])

    drug_a = DRUGS[ix.drug_a]
    drug_b = DRUGS[ix.drug_b]
    conditions = []
    for cond, cond_drugs in CONDITION_DRUGS.items():
        if ix.drug_a in cond_drugs or ix.drug_b in cond_drugs:
            conditions.append(cond)
    conditions = conditions[:2] or ["general_checkup"]
    age = rng.randint(40, 80)

    return PatientScenario(
        patient_id=f"PT-EASY-{seed or rng.randint(1000,9999)}",
        age=age, conditions=conditions,
        medications=[drug_a, drug_b],
        ground_truth_interactions=[ix],
        task_name="easy_pair_check",
        task_difficulty="easy",
        curriculum_focus=curriculum_focus,
    )


def generate_medium_scenario(seed: Optional[int] = None, curriculum_focus: str = "general") -> PatientScenario:
    rng = random.Random(seed)
    conditions, drugs, interactions = _pick_conditions_and_drugs(rng, 5, 2, curriculum_focus)
    return PatientScenario(
        patient_id=f"PT-MED-{seed or rng.randint(1000,9999)}",
        age=rng.randint(45, 75), conditions=conditions,
        medications=drugs, ground_truth_interactions=interactions,
        task_name="medium_multi_drug", task_difficulty="medium",
        curriculum_focus=curriculum_focus,
    )


def generate_hard_scenario(seed: Optional[int] = None, curriculum_focus: str = "general") -> PatientScenario:
    rng = random.Random(seed)
    conditions, drugs, interactions = _pick_conditions_and_drugs(rng, 8, 4, curriculum_focus)
    return PatientScenario(
        patient_id=f"PT-HARD-{seed or rng.randint(1000,9999)}",
        age=rng.randint(60, 85), conditions=conditions,
        medications=drugs, ground_truth_interactions=interactions,
        task_name="hard_polypharmacy", task_difficulty="hard",
        curriculum_focus=curriculum_focus,
    )


TASK_DEFINITIONS = {
    "easy_pair_check":   {"description": "2-drug interaction identification", "difficulty": "easy",   "generator": generate_easy_scenario},
    "medium_multi_drug": {"description": "5-drug polypharmacy review",        "difficulty": "medium", "generator": generate_medium_scenario},
    "hard_polypharmacy": {"description": "8-drug complex polypharmacy",       "difficulty": "hard",   "generator": generate_hard_scenario},
}

def drug_to_dict(drug: Drug) -> dict:   return asdict(drug)
def interaction_to_dict(ix: Interaction) -> dict: return asdict(ix)