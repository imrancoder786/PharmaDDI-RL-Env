# PharmaDDI: Clinical Drug-Drug Interaction RL Environment
### *Empowering AI Agents to Prevent Adverse Drug Events in Complex Polypharmacy*

---

##  Motivation & Real-World Utility
**Drug-Drug Interactions (DDIs)** are a leading cause of preventable medical errors, hospitalizations, and mortality worldwide. As patients age and develop multiple chronic conditions, the complexity of their medication regimens (polypharmacy) increases exponentially. 

In a typical clinical workflow, a pharmacist must:
1. Identify all interacting pairs within a list of 5-10+ drugs.
2. Assess the clinical significance (Severity).
3. Determine the best course of action (Recommendation) to mitigate risk while maintaining therapeutic efficacy.

**PharmaDDI** provides a robust, clinically-grounded RL environment to benchmark AI agents on these life-critical decision-making tasks.

---

##  Environment Overview
The **PharmaDDI** environment presents the agent with a patient "Patient Profile" containing:
- **Demographics**: Age and medical conditions.
- **Medication List**: Generic names, therapeutic classes, doses, and frequencies.

The agent's goal is to detect and report all clinically significant interactions within the list, simulating the expertise of a board-certified clinical pharmacist.

---

##  Action & Observation Spaces

### Observation Space
The agent receives a `PharmaDDIObservation` object containing the clinical context:

```python
class MedicationInfo(BaseModel):
    name: str # e.g., "warfarin"
    therapeutic_class: str # e.g., "anticoagulant"
    common_dose: str # e.g., "5mg"
    frequency: str # e.g., "once daily"

class PharmaDDIObservation(Observation):
    patient_id: str
    patient_age: int
    patient_conditions: List[str]
    medications: List[MedicationInfo]
    num_medications: int
    instructions: str
```

### Action Space
The agent responds with a `PharmaDDIAction` containing a list of interaction reports:

```python
class InteractionReport(BaseModel):
    drug_a: str
    drug_b: str
    severity: str # minor | moderate | major | contraindicated
    clinical_effect: str # Clinical consequence
    recommendation: str # monitor | adjust_dose | substitute | discontinue

class PharmaDDIAction(Action):
    interactions_found: List[InteractionReport]
```

---

##  Tasks & Grader Logic
PharmaDDI features three tasks of increasing clinical complexity:

| Task ID | Medications | Objective | Grader Difficulty |
| :--- | :--- | :--- | :--- |
| **`easy_pair_check`** | 2 | Determine if a specific pair interacts. | Focused on binary detection and severity classification. |
| **`medium_multi_drug`** | 5 | Find ALL interaction pairs in the list. | Requires pairwise exhaustive checking (10 possible pairs). |
| **`hard_polypharmacy`** | 8 | Manage complex elderly patient regimens. | High penalty for missing critical interactions (Major/Contraindicated). |

---

##  Reward Function
The environment implements a **weighted partial-credit system** (0.0 - 1.0) to encourage precise clinical reasoning:

- **Identification (30-60%)**: Points for finding the correct interacting drug pairs.
- **Severity Classification (25-40%)**: Points for matching the ground-truth severity.
- **Recommendations (10-25%)**: Points for suggesting the correct clinical intervention.
- **Penalties**: 
    - **False Positives**: Deductions for flagging interactions that do not exist (prevents "alert fatigue").
    - **Critical Misses**: Large penalties for failing to identify *Major* or *Contraindicated* interactions.

---


##  Setup & Usage

### Local Development
1. **Clone & Install**:
   ```bash
   pip install -e .
   ```
2. **Run Server**:
   ```bash
   uvicorn server.app:app --host 0.0.0.0 --port 8000
   ```
3. **Verify with Validator**:
   ```bash
   python vall.py
   ```

### Docker Deployment
```bash
docker build -t pharmaddi-env:latest -f Dockerfile .
docker run -p 8000:8000 pharmaddi-env:latest
```

### Hugging Face Space
The environment is optimized for HF Spaces. Simply run:
```bash
openenv push
```

---

## Project Structure
| File | Description |
| :--- | :--- |
| `server/PharmaDDIEnv_environment.py` | Core RL environment and grading engine. |
| `server/drug_data.py` | Knowledge base of 30+ drugs and 60+ DDI pairs. |
| `models.py` | Pydantic schemas for Actions and Observations. |
| `inference.py` | Baseline inference script with Hackathon-compliant logging. |
| `openenv.yaml` | Environment manifest for the OpenEnv framework. |

---

##  License & Acknowledgements
Built for the **Meta x HuggingFace OpenEnv Hackathon**.
Knowledge base curated from publicly available clinical guidelines and drug interaction databases.

*Disclaimer: This environment is for research and benchmarking purposes only. It is not a substitute for clinical judgment.*
