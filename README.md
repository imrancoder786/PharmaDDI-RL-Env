---
title: PharmaDDI Environment - Drug Interaction Checker
emoji: 💊
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# PharmaDDI Environment - Pharmaceutical Drug Interaction Checker

A clinical pharmacist RL environment where an AI agent reviews patient medication lists and identifies dangerous **drug-drug interactions (DDIs)**. Medication errors cause ~7,000 deaths/year in the US, making automated DDI checking a high-impact real-world application.

## Motivation

Polypharmacy (multiple concurrent medications) is extremely common in elderly patients. Manual drug interaction checking is time-consuming and error-prone. This environment trains and evaluates AI agents on this critical clinical task.

## Quick Start

```python
from PharmaDDIEnv import PharmaDDIAction, PharmaDDIEnv, InteractionReport

# Create environment from Docker image
env = PharmaDDIEnv.from_docker_image("PharmaDDIEnv-env:latest")

try:
    result = env.reset()
    print(f"Patient: {result.observation.patient_id}")
    print(f"Medications: {[m.name for m in result.observation.medications]}")

    # Submit interaction analysis
    action = PharmaDDIAction(interactions_found=[
        InteractionReport(
            drug_a="warfarin", drug_b="aspirin",
            severity="major",
            clinical_effect="Increased bleeding risk",
            recommendation="monitor"
        )
    ])
    result = env.step(action)
    print(f"Score: {result.observation.score}")
    print(f"Feedback: {result.observation.feedback}")
finally:
    env.close()
```

## Building & Running

```bash
# Build Docker image
docker build -t PharmaDDIEnv-env:latest -f server/Dockerfile .

# Run locally (development)
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

# Deploy to HF Spaces
openenv push
```

## Action Space

**`PharmaDDIAction`** - Agent's drug interaction analysis:

| Field | Type | Description |
|-------|------|-------------|
| `interactions_found` | `List[InteractionReport]` | List of identified DDI pairs |

Each **`InteractionReport`** contains:

| Field | Type | Values |
|-------|------|--------|
| `drug_a` | `str` | Drug name (lowercase) |
| `drug_b` | `str` | Drug name (lowercase) |
| `severity` | `str` | `minor` \| `moderate` \| `major` \| `contraindicated` |
| `clinical_effect` | `str` | Description of the interaction mechanism |
| `recommendation` | `str` | `monitor` \| `adjust_dose` \| `substitute` \| `discontinue` |

## Observation Space

**`PharmaDDIObservation`** - Patient scenario:

| Field | Type | Description |
|-------|------|-------------|
| `task_name` | `str` | `easy_pair_check` \| `medium_multi_drug` \| `hard_polypharmacy` |
| `task_difficulty` | `str` | `easy` \| `medium` \| `hard` |
| `patient_id` | `str` | Unique patient identifier |
| `patient_age` | `int` | Patient age in years |
| `patient_conditions` | `List[str]` | Medical conditions |
| `medications` | `List[MedicationInfo]` | Current medication list |
| `num_medications` | `int` | Number of medications |
| `instructions` | `str` | Task-specific instructions |
| `feedback` | `str` | Grading feedback after submission |
| `score` | `float` | Score achieved (0.01.0) |
| `total_interactions` | `int` | Hint for easy mode only |

## Tasks

### Task 1: Easy - Pair Check (2 drugs)
- Given 2 medications, identify if a DDI exists
- Classify severity level
- **Scoring**: 60% pair identification + 40% severity  false positive penalty

### Task 2: Medium - Multi-Drug Review (5 drugs)
- Review 5 concurrent medications
- Find ALL interacting pairs and classify severity
- **Scoring**: 40% pairs + 35% severity + 10% recommendations + 15% precision  critical miss penalty

### Task 3: Hard - Polypharmacy (8 drugs)
- Complex elderly patient with 8 medications
- Find all interactions, classify severity, AND recommend clinical actions
- **Scoring**: 30% pairs + 25% severity + 25% recommendations + 10% precision + 10% completeness  critical miss penalty

## Reward Function

Rewards range from **0.0 to 1.0** per task with partial credit:

-  **Correct pair identification**: Points for each real DDI found
-  **Severity accuracy**: Full points for exact match, 50% for adjacent severity
-  **Recommendation quality** (medium/hard): Points for correct clinical action
-  **False positive penalty**: Deduction per hallucinated interaction
-  **Critical miss penalty**: Extra penalty for missing major/contraindicated interactions

## Drug Knowledge Base

The environment includes **30 common drugs** across therapeutic classes:
- Cardiovascular (warfarin, aspirin, clopidogrel, lisinopril, atorvastatin, etc.)
- CNS/Psychiatric (fluoxetine, sertraline, diazepam, tramadol, lithium, etc.)
- Antibiotics (ciprofloxacin, metronidazole, erythromycin, rifampin, etc.)
- Analgesics (ibuprofen, naproxen, acetaminophen)
- Diabetes (metformin, glipizide)
- Other (omeprazole, levothyroxine, spironolactone)

With **60+ known DDI pairs** based on real clinical pharmacology.

## Baseline Inference

```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4"
export HF_TOKEN="your-api-key"
python inference.py
```

### Expected Baseline Scores (GPT-4)

| Task | Expected Score | Difficulty |
|------|---------------|------------|
| `easy_pair_check` | ~0.851.0 | Easy |
| `medium_multi_drug` | ~0.550.75 | Medium |
| `hard_polypharmacy` | ~0.350.55 | Hard |

## Project Structure

```
PharmaDDIEnv/
-- inference.py              # Baseline inference script
-- openenv.yaml              # OpenEnv manifest with 3 tasks
-- pyproject.toml             # Project metadata & dependencies
-- README.md                  # This file
-- models.py                  # Action/Observation Pydantic models
-- client.py                  # PharmaDDIEnv client
-- __init__.py                # Module exports
-- server/
    -- drug_data.py           # Drug & interaction knowledge base
    -- PharmaDDIEnv_environment.py  # Core environment + grading logic
    -- app.py                 # FastAPI application
    -- Dockerfile             # Container image definition
    -- requirements.txt       # Server dependencies
```

## Setup Instructions

1. **Install dependencies**: `pip install openenv-core[core]`
2. **Build Docker**: `docker build -t PharmaDDIEnv-env:latest -f server/Dockerfile .`
3. **Run server**: `uvicorn server.app:app --host 0.0.0.0 --port 8000`
4. **Validate**: `openenv validate`
5. **Deploy**: `openenv push`

## Deploying to Hugging Face Spaces

```bash
openenv push
# or with options:
openenv push --repo-id your-username/PharmaDDIEnv --private
```

The deployed space includes:
- **Web Interface** at `/web`
- **API Documentation** at `/docs`
- **Health Check** at `/health`
- **WebSocket** at `/ws`
