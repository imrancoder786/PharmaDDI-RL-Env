"""
Baseline inference script for PharmaDDIEnv.
Uses the OpenAI Python client with HF-provided API key.
Strictly follows the Meta x HuggingFace OpenEnv Hackathon [START], [STEP], [END] format.
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
MAX_STEPS = 15
TEMPERATURE = 0.1
MAX_TOKENS = 1000

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert clinical pharmacist AI assistant.
    Your job is to assess drug interaction safety for patients.

    Examine the patient's current medications and identify any dangerous drug-drug interactions.
    Check EVERY possible pair of medications. Be thorough and accurate.
    Use only lowercase generic drug names.
    Severity levels: minor | moderate | major | contraindicated
    Recommendations: monitor | adjust_dose | substitute | discontinue

    IMPORTANT: Always respond with a single valid JSON object representing the interactions found.
    Example format:
    {
      "interactions_found": [
        {
          "drug_a": "drug_name_lowercase",
          "drug_b": "drug_name_lowercase",
          "severity": "major",
          "clinical_effect": "brief description of the interaction",
          "recommendation": "substitute"
        }
      ]
    }
    No explanations outside the JSON.
""").strip()


# -- Logging Helpers (exact hackathon format) --------------------------------

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


# -- LLM Interaction --------------------------------------------------------

def get_model_message(client: OpenAI, step: int, last_obs: dict, last_reward: float, history: List[str]) -> str:
    history_block = "\n".join(history[-6:]) if history else "None"
    
    meds = last_obs.get("medications", [])
    med_lines = []
    for m in meds:
        med_lines.append(f"- {m.get('name', 'Unknown')} ({m.get('therapeutic_class', '')}) {m.get('common_dose', '')} {m.get('frequency', '')}")
    med_list = "\n".join(med_lines)
    
    user_prompt = textwrap.dedent(f"""
        Patient Age: {last_obs.get('patient_age')}
        Patient Conditions: {', '.join(last_obs.get('patient_conditions', []))}
        
        Current Medications:
        {med_list}
        
        Instructions:
        {last_obs.get('instructions', '')}
        
        Respond with your findings as a single JSON object.
    """).strip()

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        match = re.search(r'\{.*\}', text, re.DOTALL)
        return match.group() if match else text
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return json.dumps({"interactions_found": []})


# -- Task Runner --------------------------------------------------------------

async def run_task(task_name: str):
    # Create client inside function
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Reset environment using a requests Session to handle cookies/state if any
    session = requests.Session()
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

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    last_obs = obs
    last_reward = 0.0
    done = False

    try:
        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            action_json_str = get_model_message(client, step, last_obs, last_reward, history)

            try:
                action_parsed = json.loads(action_json_str)
                if "interactions_found" not in action_parsed:
                    action_parsed = {"interactions_found": []}
            except json.JSONDecodeError:
                action_parsed = {"interactions_found": []}

            # Our environment expects payload nested in "action" key
            payload = {"action": action_parsed}

            try:
                step_resp = session.post(f"{ENV_BASE_URL}/step", json=payload)
                step_resp.raise_for_status()
                step_data = step_resp.json()

                obs = step_data.get("observation", {})
                reward = step_data.get("reward", 0.0)
                if reward is None:
                    reward = 0.0
                done = step_data.get("done", False)
                error = None
            except Exception as e:
                obs = {}
                reward = 0.0
                done = True
                error = str(e)

            rewards.append(reward)
            steps_taken = step
            last_obs = obs
            last_reward = reward

            # Make action compact for single-line logging
            compact_action = json.dumps(action_parsed, separators=(',', ':'))
            log_step(step=step, action=compact_action, reward=reward, done=done, error=error)
            
            num_ix = len(action_parsed.get("interactions_found", []))
            history.append(f"Step {step}: Submitted {num_ix} interactions -> reward {reward:+.2f}")

            if done:
                break

        if rewards:
            score = max(rewards)  # Single episode, so max/final reward is the score
        score = min(max(float(score), 0.0), 1.0)
        success = score >= 0.5

    except Exception as e:
        print(f"[DEBUG] Task {task_name} error: {e}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# -- Main --------------------------------------------------------------------

async def main() -> None:
    # Test all 3 tasks
    for level in ["easy_pair_check", "medium_multi_drug", "hard_polypharmacy"]:
        await run_task(level)

if __name__ == "__main__":
    asyncio.run(main())
