"""
Baseline inference script for PharmaDDIEnv.
Multi-step version with feedback refinement.
"""

import asyncio
import os
import json
import re
import textwrap
from typing import List, Optional

from openai import OpenAI
import requests

# -- Config ------------------------------------------------------------------
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
ENV_BASE_URL = os.getenv("ENV_BASE_URL") or "http://localhost:8000"

BENCHMARK = "PharmaDDIEnv"
MAX_STEPS = 8          # allow up to 8 refinement steps
TEMPERATURE = 0.1
MAX_TOKENS = 1200

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert clinical pharmacist AI assistant.
    You will review a patient's medication list and identify drug-drug interactions.

    You can submit your findings in multiple steps:
    - On each step, provide a JSON with "interactions_found" list.
    - After each submission, you will receive a score (0-1) and detailed feedback.
    - Use the feedback to improve your next submission: correct severity, add missing interactions, remove false positives.
    - When you are confident you have the best possible answer, set "done": true.

    Example action JSON:
    {
      "interactions_found": [
        {
          "drug_a": "lisinopril",
          "drug_b": "losartan",
          "severity": "major",
          "clinical_effect": "increased risk of hyperkalemia",
          "recommendation": "discontinue"
        }
      ],
      "done": false
    }

    Use only lowercase generic drug names.
    Severity levels: minor | moderate | major | contraindicated
    Recommendations: monitor | adjust_dose | substitute | discontinue

    Respond ONLY with valid JSON. No other text.
""").strip()


# -- Logging Helpers ---------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


# -- JSON Extraction ---------------------------------------------------------

def extract_json(text: str) -> dict:
    """Robust JSON extraction from model output."""
    # Try direct parse
    try:
        return json.loads(text.strip())
    except:
        pass

    # Try to find JSON object via regex
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except:
            pass

    # Fallback: empty interactions
    print("[WARN] Could not parse JSON. Returning empty interactions.", flush=True)
    return {"interactions_found": [], "done": False}


# -- Prompt Builder ----------------------------------------------------------

def build_user_prompt(step: int, obs: dict, last_feedback: str, last_score: float) -> str:
    meds = obs.get("medications", [])
    med_list = "\n".join([f"- {m['name']} ({m.get('therapeutic_class', '')})" for m in meds])

    feedback_section = last_feedback if last_feedback else "No feedback yet. This is your first attempt."

    return textwrap.dedent(f"""
        Step: {step}
        Patient Age: {obs.get('patient_age')}
        Conditions: {', '.join(obs.get('patient_conditions', []))}
        Medications:
        {med_list}

        Previous Score: {last_score:.3f}
        Feedback from previous attempt:
        {feedback_section}

        Provide an improved JSON response with "interactions_found" and a "done" flag (true/false).
    """).strip()


# -- Task Runner -------------------------------------------------------------

async def run_task(task_name: str):
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    session = requests.Session()

    # Reset environment
    try:
        reset_resp = session.post(f"{ENV_BASE_URL}/reset", json={"task_name": task_name})
        reset_resp.raise_for_status()
        reset_data = reset_resp.json()
        obs = reset_data.get("observation", {})
    except Exception as e:
        print(f"[DEBUG] Failed to reset task {task_name}: {e}", flush=True)
        return

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    last_feedback = ""
    done = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            # Build prompt with current feedback
            user_prompt = build_user_prompt(step, obs, last_feedback, score)

            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                )
                action_text = completion.choices[0].message.content or ""
            except Exception as e:
                print(f"[DEBUG] LLM error: {e}", flush=True)
                action_text = "{}"

            action_json = extract_json(action_text)
            # Ensure required fields
            if "interactions_found" not in action_json:
                action_json["interactions_found"] = []
            if "done" not in action_json:
                action_json["done"] = False

            # Send to environment
            payload = {"action": action_json}
            try:
                step_resp = session.post(f"{ENV_BASE_URL}/step", json=payload)
                step_resp.raise_for_status()
                step_data = step_resp.json()
            except Exception as e:
                print(f"[DEBUG] Step error: {e}", flush=True)
                step_data = {"observation": obs, "reward": 0.0, "done": True}

            obs = step_data.get("observation", obs)
            reward = step_data.get("reward", 0.0)
            done = step_data.get("done", False)
            error = step_data.get("error")

            rewards.append(reward)
            steps_taken = step
            score = obs.get("score", score)
            last_feedback = obs.get("feedback", "")

            compact_action = json.dumps(action_json, separators=(',', ':'))
            log_step(step, compact_action, reward, done, error)

            history.append(f"Step {step}: score {score:.3f}, reward +{reward:.3f}")

            if done:
                break

        success = score >= 0.5

    except Exception as e:
        print(f"[DEBUG] Task {task_name} fatal error: {e}", flush=True)

    finally:
        log_end(success, steps_taken, score, rewards)


# -- Main --------------------------------------------------------------------

async def main() -> None:
    for task in ["easy_pair_check", "medium_multi_drug", "hard_polypharmacy"]:
        await run_task(task)


if __name__ == "__main__":
    asyncio.run(main())