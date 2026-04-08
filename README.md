---
title: PharmaDDIEnv
emoji: 💊
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 8000
---

# PharmaDDI: Clinical Drug-Drug Interaction RL Environment

### Empowering AI Agents to Prevent Adverse Drug Events in Complex Polypharmacy

---

## Motivation & Real-World Utility

Drug-Drug Interactions (DDIs) are a major cause of preventable medical errors, hospitalizations, and mortality worldwide. As patients age and develop multiple chronic conditions, the complexity of their medication regimens (polypharmacy) increases significantly.

In a typical clinical workflow, a pharmacist must:

1. Identify all interacting drug pairs within a list of medications
2. Assess the clinical severity of each interaction
3. Recommend appropriate actions to reduce patient risk while maintaining treatment effectiveness

PharmaDDI provides a clinically grounded reinforcement learning (RL) environment to benchmark AI agents on these critical healthcare decision-making tasks.

---

## Environment Overview

The PharmaDDI environment provides an agent with a structured patient profile that includes:

* Demographics (age and medical conditions)
* Medication list (drug name, class, dosage, and frequency)

The goal of the agent is to detect and report all clinically significant drug-drug interactions, simulating the reasoning process of a clinical pharmacist.

---

## Observation Space

The agent receives a structured observation object:

```python
class MedicationInfo(BaseModel):
    name: str
    therapeutic_class: str
    common_dose: str
    frequency: str

class PharmaDDIObservation(Observation):
    patient_id: str
    patient_age: int
    patient_conditions: List[str]
    medications: List[MedicationInfo]
    num_medications: int
    instructions: str
```

---

## Action Space

The agent returns detected interactions in the following format:

```python
class InteractionReport(BaseModel):
    drug_a: str
    drug_b: str
    severity: str  # minor | moderate | major | contraindicated
    clinical_effect: str
    recommendation: str  # monitor | adjust_dose | substitute | discontinue

class PharmaDDIAction(Action):
    interactions_found: List[InteractionReport]
    done: bool  # Set to True when the agent is confident and wants to end the episode
```

---

## Tasks & Evaluation

The environment includes three levels of difficulty:

| Task ID           | Medications | Objective                           | Difficulty |
| ----------------- | ----------- | ----------------------------------- | ---------- |
| easy_pair_check   | 2           | Detect interaction for a given pair | Easy       |
| medium_multi_drug | 5           | Identify all interacting pairs      | Medium     |
| hard_polypharmacy | 8           | Handle complex multi-drug cases     | Hard       |

---

## Reward Function

## Reward Function

The environment supports **multi‑step episodes** with incremental reward shaping:

- At each step, the agent receives a **score** (0.0–1.0) based on:
  - Identification Accuracy (30–60%)
  - Severity Classification (25–40%)
  - Recommendation Quality (10–25%)
- The **reward** for the current step is the **improvement** over the agent’s previous best score.
- Penalties are applied for:
  - False positives (incorrect interaction detection)
  - Missing critical interactions (major or contraindicated)

This design encourages the agent to iteratively refine its answer using the detailed feedback provided after each attempt.

---

### Multi‑Step Refinement Example

The agent can improve its answer across multiple steps:

```text
[START] task=medium_multi_drug env=PharmaDDIEnv model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action={"interactions_found":[...],"done":false} reward=0.10 done=false error=null
[STEP] step=2 action={"interactions_found":[...],"done":true}  reward=0.89 done=true error=null
[END] success=true steps=2 score=0.990 rewards=0.10,0.89
```
---

## Example Workflow

1. Input: Patient with multiple medications
2. Agent analyzes all possible drug pairs
3. Outputs structured interaction reports
4. Environment scores the response based on correctness

---

## Setup & Usage

### Local Development

```bash
pip install -e .
uvicorn server.app:app --host 0.0.0.0 --port 8000
python vall.py
```

---

### Docker Deployment

```bash
docker build -t pharmaddi-env:latest -f Dockerfile .
docker run -p 8000:8000 pharmaddi-env:latest
```

---

### Hugging Face Deployment

```bash
openenv push
```

---

## Project Structure

| File                               | Description                               |
| ---------------------------------- | ----------------------------------------- |
| server/PharmaDDIEnv_environment.py | Core RL environment and grading logic     |
| server/drug_data.py                | Drug database and interaction rules       |
| models.py                          | Data schemas for observations and actions |
| inference.py                       | Baseline agent implementation             |
| openenv.yaml                       | OpenEnv configuration                     |

---

## License & Acknowledgements

This project was developed for the Meta x Hugging Face OpenEnv Hackathon.

The drug interaction knowledge base is derived from publicly available clinical sources and research literature.

Disclaimer: This project is intended for research and educational purposes only. It should not be used for real clinical decision-making.
---