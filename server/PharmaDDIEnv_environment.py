"""
PharmaDDI Environment — with Adaptive Curriculum Learning and DDI-Bench inspired data.
"""

import random as _random
from uuid import uuid4
from typing import Optional, Dict, Any

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import PharmaDDIAction, PharmaDDIObservation, MedicationInfo
except ImportError:
    from models import PharmaDDIAction, PharmaDDIObservation, MedicationInfo

try:
    from .drug_data import (
        TASK_DEFINITIONS, PatientScenario, drug_to_dict,
        interaction_to_dict, lookup_interaction, SEVERITY_ORDER, DRUGS,
    )
    from .curriculum import CurriculumEngine
except ImportError:
    from drug_data import (
        TASK_DEFINITIONS, PatientScenario, drug_to_dict,
        interaction_to_dict, lookup_interaction, SEVERITY_ORDER, DRUGS,
    )
    from curriculum import CurriculumEngine


INSTRUCTIONS = {
    "easy_pair_check": (
        "You are a clinical pharmacist. A patient is prescribed the following 2 medications. "
        "Determine if there is a drug-drug interaction between them.\n\n"
        "Respond with a JSON object containing an 'interactions_found' array. "
        "Each interaction: drug_a, drug_b, severity (minor/moderate/major/contraindicated), "
        "clinical_effect, recommendation (monitor/adjust_dose/substitute/discontinue).\n"
        "If no interaction, return empty interactions_found."
    ),
    "medium_multi_drug": (
        "You are a clinical pharmacist reviewing 5 medications. "
        "Identify ALL drug-drug interactions. Note: total count is not disclosed — be thorough.\n\n"
        "For each interaction: drug_a, drug_b (lowercase), severity, clinical_effect, recommendation.\n"
        "Check all possible pairs."
    ),
    "hard_polypharmacy": (
        "You are a clinical pharmacist reviewing a complex polypharmacy case with 8 medications. "
        "This elderly patient requires careful medication review.\n\n"
        "Identify ALL drug-drug interactions:\n"
        "- drug_a and drug_b (lowercase generic names)\n"
        "- severity: minor | moderate | major | contraindicated\n"
        "- clinical_effect: mechanism and clinical consequence\n"
        "- recommendation: monitor | adjust_dose | substitute | discontinue\n\n"
        "Note: total interactions NOT disclosed. Accuracy of severity AND recommendation is critical.\n"
        "Hint: check CYP enzyme interactions carefully — many interactions are pharmacokinetic."
    ),
}


class PharmaDDIEnvironment(Environment):
    """
    PharmaDDI RL Environment with Adaptive Curriculum Learning.

    New in this version:
    - 60+ drugs, 120+ interactions (DDI-Bench inspired coverage)
    - CurriculumEngine tracks agent weaknesses per drug class
    - Scenarios are generated to target the agent's weakest areas
    - Mechanism-aware hints in instructions (CYP pathways)
    - Random seed per episode (never same patient twice)
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = False

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._scenario: Optional[PatientScenario] = None
        self._current_task: str = "easy_pair_check"
        self._done: bool = False
        self._last_score: float = 0.0
        self._best_score: float = 0.0
        self._best_submission = None
        self._max_steps: int = 10
        self._current_focus: str = "general"

        # Adaptive curriculum engine — persists across episodes
        self._curriculum = CurriculumEngine()
        self._rng = _random.Random()

    def reset(self, task_name: str = None, seed: int = None) -> PharmaDDIObservation:
        """
        Reset environment. Each call generates a fresh random patient.
        Curriculum engine selects which drug class to focus on based on
        the agent's previous performance weaknesses.
        """
        # Record previous episode result in curriculum before resetting
        if self._scenario is not None and self._best_score > 0:
            self._curriculum.record_episode(
                curriculum_focus=self._current_focus,
                score=self._best_score,
                task_name=self._current_task,
            )

        # Reset episode state
        self._best_score = 0.0
        self._best_submission = None
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._done = False
        self._last_score = 0.0

        if task_name and task_name in TASK_DEFINITIONS:
            self._current_task = task_name

        # Always use a fresh random seed unless caller specifies one
        episode_seed = seed if seed is not None else _random.randint(0, 999999)

        # Ask curriculum which drug class to focus on
        self._current_focus = self._curriculum.select_focus(self._rng)

        # Generate scenario with curriculum focus
        generator = TASK_DEFINITIONS[self._current_task]["generator"]
        self._scenario = generator(seed=episode_seed, curriculum_focus=self._current_focus)

        curriculum_hint = (
            f"[Curriculum focus: {self._current_focus.replace('_', ' ')}]"
            if self._current_focus != "general" else ""
        )

        return self._build_observation(
            feedback=(
                f"Environment reset. Review the patient's medications and identify drug interactions. "
                f"{curriculum_hint}"
            ),
            score=0.0,
        )

    def step(self, action: PharmaDDIAction) -> PharmaDDIObservation:
        """
        Grade the agent's submission. Supports multi-step refinement.
        Reward = improvement over best score so far (delta reward shaping).
        """
        if self._done:
            return self._build_observation(
                feedback="Episode complete. Call reset() to start a new episode.",
                score=self._best_score, done=True, reward=0.0,
            )

        self._state.step_count += 1

        if self._scenario is None:
            return self._build_observation(
                feedback="No scenario loaded. Call reset() first.",
                score=0.0, done=True, reward=0.0,
            )

        raw_score, feedback = self._grade_submission(action)

        # Allow full 0.0 and 1.0 — no artificial clamping
        final_score = min(max(raw_score, 0.0), 1.0)

        # Delta reward: only reward genuine improvement
        improvement = max(0.0, final_score - self._best_score)
        if final_score > self._best_score:
            self._best_score = final_score
            self._best_submission = action

        done = action.done or (self._state.step_count >= self._max_steps)
        if done:
            self._done = True
            # Record final score in curriculum
            self._curriculum.record_episode(
                curriculum_focus=self._current_focus,
                score=self._best_score,
                task_name=self._current_task,
            )

        return self._build_observation(
            feedback=feedback,
            score=self._best_score,
            reward=improvement,
            done=done,
        )

    @property
    def state(self) -> State:
        return self._state

    def get_curriculum_status(self) -> Dict:
        """Expose curriculum state — useful for monitoring agent learning."""
        return self._curriculum.get_status()

    # ------------------------------------------------------------------
    # Grading Logic (improved partial credit + capped penalties)
    # ------------------------------------------------------------------

    def _grade_submission(self, action: PharmaDDIAction) -> tuple:
        scenario = self._scenario
        ground_truth = scenario.ground_truth_interactions
        submitted = action.interactions_found

        gt_map: Dict[frozenset, Any] = {}
        for ix in ground_truth:
            key = frozenset({ix.drug_a.lower(), ix.drug_b.lower()})
            gt_map[key] = ix

        total_gt = len(ground_truth)
        if total_gt == 0:
            if len(submitted) == 0:
                return 1.0, "Perfect! Correctly identified no interactions."
            penalty = min(len(submitted) * 0.2, 1.0)
            return max(0.0, 1.0 - penalty), f"False positives: {len(submitted)} reported but none exist."

        correct_pairs = 0
        correct_severity = 0
        correct_recommendation = 0
        false_positives = 0
        matched_keys = set()
        feedback_lines = []

        rec_order = {"monitor": 1, "adjust_dose": 2, "substitute": 3, "discontinue": 4}

        for sub in submitted:
            key = frozenset({sub.drug_a.lower(), sub.drug_b.lower()})
            if key in gt_map:
                matched_keys.add(key)
                gt_ix = gt_map[key]
                correct_pairs += 1

                # Severity scoring with partial credit
                sub_sev = sub.severity.lower().strip()
                gt_sev = gt_ix.severity.lower()
                if sub_sev == gt_sev:
                    correct_severity += 1.0
                    feedback_lines.append(f"  ✓ {sub.drug_a}+{sub.drug_b}: correct pair & severity ({gt_sev})")
                else:
                    sub_ord = SEVERITY_ORDER.get(sub_sev, 0)
                    gt_ord = SEVERITY_ORDER.get(gt_sev, 0)
                    dist = abs(sub_ord - gt_ord)
                    if dist == 1:
                        correct_severity += 0.5
                        feedback_lines.append(f"  ~ {sub.drug_a}+{sub.drug_b}: pair correct, severity close (said {sub_sev}, actual {gt_sev})")
                    else:
                        correct_severity += 0.1   # at least named the pair
                        feedback_lines.append(f"  ✗ {sub.drug_a}+{sub.drug_b}: pair correct but wrong severity (said {sub_sev}, actual {gt_sev})")

                # Recommendation scoring with graduated partial credit
                sub_rec = sub.recommendation.lower().strip()
                gt_rec = gt_ix.recommendation.lower()
                if sub_rec == gt_rec:
                    correct_recommendation += 1.0
                elif sub_rec in rec_order:
                    distance = abs(rec_order.get(sub_rec, 0) - rec_order.get(gt_rec, 0))
                    if distance == 1:
                        correct_recommendation += 0.5
                    elif distance == 2:
                        correct_recommendation += 0.25
                    else:
                        correct_recommendation += 0.1
            else:
                false_positives += 1
                feedback_lines.append(f"  ✗ {sub.drug_a}+{sub.drug_b}: FALSE POSITIVE — no known interaction")

        missed = []
        for key, gt_ix in gt_map.items():
            if key not in matched_keys:
                missed.append(gt_ix)
                tag = "CRITICAL" if gt_ix.severity in ("major", "contraindicated") else "missed"
                feedback_lines.append(f"  ✗ MISSED ({tag}): {gt_ix.drug_a}+{gt_ix.drug_b} ({gt_ix.severity})")

        task = self._current_task

        if task == "easy_pair_check":
            pair_score = correct_pairs / total_gt
            sev_score = correct_severity / total_gt
            fp_penalty = false_positives * 0.15
            raw_score = (0.6 * pair_score) + (0.4 * sev_score) - fp_penalty

        elif task == "medium_multi_drug":
            pair_score = correct_pairs / total_gt
            sev_score = correct_severity / total_gt
            rec_score = correct_recommendation / total_gt
            fp_penalty = false_positives * 0.1
            critical_missed = sum(1 for m in missed if m.severity in ("major", "contraindicated"))
            critical_penalty = min(critical_missed * 0.1, 0.25)   # CAPPED
            no_fp_bonus = 1.0 if false_positives == 0 else max(0, 1.0 - fp_penalty)
            raw_score = (0.4 * pair_score) + (0.35 * sev_score) + (0.1 * rec_score) + (0.15 * no_fp_bonus) - critical_penalty

        else:  # hard_polypharmacy
            pair_score = correct_pairs / total_gt
            sev_score = correct_severity / total_gt
            rec_score = correct_recommendation / total_gt
            
            # INCREASE false positive penalty slightly to punish guessing
            fp_penalty = false_positives * 0.10 
            completeness = 1.0 - (len(missed) / total_gt)
            critical_missed = sum(1 for m in missed if m.severity in ("major", "contraindicated"))
            
            # UNCAPPED or severely raised penalty for missing lethal interactions
            critical_penalty = critical_missed * 0.25  # Lose 25% of score per missed critical!
            
            no_fp_bonus = 1.0 if false_positives == 0 else max(0, 1.0 - fp_penalty)
            raw_score = (
                0.30 * pair_score + 0.25 * sev_score + 0.25 * rec_score
                + 0.10 * no_fp_bonus + 0.10 * completeness - critical_penalty
            )
            

        final_score = min(max(raw_score, 0.0), 1.0)

        # Add mechanism hint in feedback for learning
        mechanism_hints = []
        for gt_ix in missed:
            if gt_ix.mechanism == "pharmacokinetic":
                mechanism_hints.append(f"  Hint: {gt_ix.drug_a}+{gt_ix.drug_b} is a CYP enzyme interaction.")
        if mechanism_hints:
            feedback_lines.extend(["", "Mechanism hints:"] + mechanism_hints)

        summary = (
            f"Score: {final_score:.3f}/1.000 | "
            f"Pairs: {correct_pairs}/{total_gt} | "
            f"Severity: {correct_severity:.1f}/{total_gt} | "
            f"Recommendations: {correct_recommendation:.1f}/{total_gt} | "
            f"FP: {false_positives} | Missed: {len(missed)}\n"
            + "\n".join(feedback_lines)
        )

        return final_score, summary

    def _build_observation(self, feedback="", score=0.0, reward=0.0, done=False) -> PharmaDDIObservation:
        scenario = self._scenario
        if scenario is None:
            return PharmaDDIObservation(
                task_name="", task_difficulty="", patient_id="", patient_age=0,
                patient_conditions=[], medications=[], num_medications=0,
                instructions="Call reset() to start.", feedback=feedback,
                score=score, total_interactions=-1, done=done, reward=reward,
            )

        meds = [
            MedicationInfo(
                name=d.name, therapeutic_class=d.therapeutic_class,
                common_dose=d.common_dose, frequency=d.frequency,
            )
            for d in scenario.medications
        ]

        hint_count = len(scenario.ground_truth_interactions) if scenario.task_name == "easy_pair_check" else -1

        return PharmaDDIObservation(
            task_name=scenario.task_name,
            task_difficulty=scenario.task_difficulty,
            patient_id=scenario.patient_id,
            patient_age=scenario.age,
            patient_conditions=scenario.conditions,
            medications=meds,
            num_medications=len(meds),
            instructions=INSTRUCTIONS.get(scenario.task_name, ""),
            feedback=feedback,
            score=score,
            total_interactions=hint_count,
            done=done,
            reward=reward,
            metadata={
                "episode_id": self._state.episode_id,
                "step": self._state.step_count,
                "task": scenario.task_name,
                "curriculum_focus": self._current_focus,
                "curriculum_status": self._curriculum.get_status(),
            },
        )

# ---------------------------------------------------------------------------
# Direct testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    env = PharmaDDIEnvironment()

    for task_name in ["easy_pair_check", "medium_multi_drug", "hard_polypharmacy"]:
        print(f"\n{'='*60}")
        print(f"Testing task: {task_name}")
        print(f"{'='*60}")

        env._current_task = task_name
        obs = env.reset(task_name=task_name, seed=42)
        print(f"Patient: {obs.patient_id}, age {obs.patient_age}")
        print(f"Conditions: {obs.patient_conditions}")
        print(f"Medications: {[m.name for m in obs.medications]}")
        print(f"Instructions: {obs.instructions[:100]}...")

        # Simulate a perfect submission using ground truth
        from models import InteractionReport
        from drug_data import get_all_interactions_for_drugs

        drug_names = [m.name for m in obs.medications]
        gt = get_all_interactions_for_drugs(drug_names)
        print(f"Ground truth interactions: {len(gt)}")

        reports = [
            InteractionReport(
                drug_a=ix.drug_a,
                drug_b=ix.drug_b,
                severity=ix.severity,
                clinical_effect=ix.clinical_effect,
                recommendation=ix.recommendation,
            )
            for ix in gt
        ]

        action = PharmaDDIAction(interactions_found=reports)
        result = env.step(action)
        print(f"\nResult score: {result.score}")
        print(f"Feedback:\n{result.feedback}")

    print("\n All tasks tested successfully!")
