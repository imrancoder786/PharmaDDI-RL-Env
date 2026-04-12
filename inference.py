"""
Baseline inference script for PharmaDDIEnv.
Multi-step with feedback refinement + curriculum-aware prompting.
"""

import asyncio
import os
import json
import re
import textwrap
from typing import List, Optional

from openai import OpenAI
import requests

API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME   = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
ENV_BASE_URL = os.getenv("ENV_BASE_URL") or "http://localhost:8000"

BENCHMARK  = "PharmaDDIEnv"
MAX_STEPS  = 8
TEMPERATURE = 0.1
MAX_TOKENS  = 1500

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert clinical pharmacist AI with deep knowledge of drug-drug interactions (DDIs).

    You will review a patient's medication list and identify ALL clinically significant interactions.

    KEY KNOWLEDGE TO APPLY:
    1. CYP Enzyme interactions (pharmacokinetic):
       - CYP3A4 inhibitors: clarithromycin, erythromycin, fluconazole, itraconazole, verapamil,
         diltiazem, amlodipine — RAISE levels of CYP3A4 substrates (statins, tacrolimus, colchicine)
       - CYP2C9 inhibitors: fluconazole, metronidazole, amiodarone — RAISE warfarin, phenytoin, glipizide
       - CYP2D6 inhibitors: fluoxetine, paroxetine, duloxetine — RAISE metoprolol, tramadol
       - CYP inducers: rifampin, carbamazepine — LOWER levels of many drugs
    2. Pharmacodynamic interactions:
       - Serotonin syndrome: SSRIs/SNRIs + tramadol (contraindicated)
       - Bleeding: anticoagulants + NSAIDs + antiplatelets (major/contraindicated)
       - Bradycardia/heart block: beta-blockers + verapamil/diltiazem/amiodarone
       - Hyperkalemia: ACE inhibitors + ARBs, or either + spironolactone
       - Lithium toxicity: NSAIDs, ACE inhibitors, loop diuretics raise lithium levels
    3. Narrow therapeutic index drugs need extra care:
       warfarin, lithium, digoxin, phenytoin, theophylline, tacrolimus, cyclosporine

    STRATEGY:
    - Step 1: Identify all drug pairs. Flag any CYP inhibitors/substrates.
    - Step 2: Use feedback to add missed interactions or correct severity.
    - Final step: Set done=true when confident.

    Severity: minor | moderate | major | contraindicated
    Recommendation: monitor | adjust_dose | substitute | discontinue

    Respond ONLY with valid JSON. No other text.
""").strip()


def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error):
    error_val = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)

def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def extract_json(text: str) -> dict:
    try:
        return json.loads(text.strip())
    except:
        pass
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except:
            pass
    return {"interactions_found": [], "done": False}


def build_user_prompt(step: int, obs: dict, last_feedback: str, last_score: float) -> str:
    meds = obs.get("medications", [])
    med_lines = []
    for m in meds:
        cyp_note = ""
        # Identify CYP inhibitors from known classes in the prompt context
        if m.get("therapeutic_class") in ("antifungal", "macrolide", "fluoroquinolone",
                                           "antibiotic", "antiarrhythmic", "anticonvulsant"):
            cyp_note = " ← CHECK CYP interactions"
        med_lines.append(f"  - {m['name']} ({m.get('therapeutic_class','')}){cyp_note}")

    curriculum_focus = obs.get("metadata", {}).get("curriculum_focus", "general")
    focus_note = f"\nScenario focus area: {curriculum_focus.replace('_',' ')}" if curriculum_focus != "general" else ""

    feedback_section = last_feedback if last_feedback else "No feedback yet. This is your first attempt."

    return textwrap.dedent(f"""
        Step {step} | Previous Score: {last_score:.3f}
        {focus_note}

        Patient Age: {obs.get('patient_age')}
        Conditions: {', '.join(obs.get('patient_conditions', []))}
        Medications:
        {chr(10).join(med_lines)}

        Previous Feedback:
        {feedback_section}

        Instructions: {obs.get('instructions', '')}

        Submit improved JSON with "interactions_found" list and "done" (true when confident).
        For each interaction: drug_a, drug_b, severity, clinical_effect, recommendation.
    """).strip()


async def run_task(task_name: str, client: OpenAI, session: requests.Session):
    try:
        reset_resp = session.post(f"{ENV_BASE_URL}/reset", json={"task_name": task_name})
        reset_resp.raise_for_status()
        obs = reset_resp.json().get("observation", {})
    except Exception as e:
        print(f"[DEBUG] Reset failed for {task_name}: {e}", flush=True)
        return

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
            if "interactions_found" not in action_json:
                action_json["interactions_found"] = []
            if "done" not in action_json:
                action_json["done"] = False

            # On final step always set done=true
            if step == MAX_STEPS:
                action_json["done"] = True

            try:
                step_resp = session.post(f"{ENV_BASE_URL}/step", json={"action": action_json})
                step_resp.raise_for_status()
                step_data = step_resp.json()
            except Exception as e:
                print(f"[DEBUG] Step error: {e}", flush=True)
                step_data = {"observation": obs, "reward": 0.0, "done": True}

            obs       = step_data.get("observation", obs)
            reward    = step_data.get("reward", 0.0)
            done      = step_data.get("done", False)
            error     = step_data.get("error")

            rewards.append(reward)
            steps_taken = step
            score = obs.get("score", score)
            last_feedback = obs.get("feedback", "")

            compact_action = json.dumps(action_json, separators=(',', ':'))
            log_step(step, compact_action, reward, done, error)

            if done:
                break

        success = score >= 0.5

    except Exception as e:
        print(f"[DEBUG] Fatal error in {task_name}: {e}", flush=True)
    finally:
        log_end(success, steps_taken, score, rewards)


async def main():
    client  = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    session = requests.Session()
    for task in ["easy_pair_check", "medium_multi_drug", "hard_polypharmacy"]:
        await run_task(task, client, session)


if __name__ == "__main__":
    asyncio.run(main())