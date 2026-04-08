# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
PharmaDDI Environment Implementation.

A clinical pharmacist drug-drug interaction (DDI) checking environment.
The agent receives patient medication lists and must identify interactions,
classify their severity, and recommend clinical actions.

Three tasks with increasing difficulty:
  - easy_pair_check:    2 drugs - identify if interaction exists + severity
  - medium_multi_drug:  5 drugs - find ALL interacting pairs + severity
  - hard_polypharmacy:  8 drugs - interactions + severity + recommendations
"""

import json
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
        TASK_DEFINITIONS,
        PatientScenario,
        drug_to_dict,
        interaction_to_dict,
        lookup_interaction,
        SEVERITY_ORDER,
    )
except ImportError:
    from drug_data import (
        TASK_DEFINITIONS,
        PatientScenario,
        drug_to_dict,
        interaction_to_dict,
        lookup_interaction,
        SEVERITY_ORDER,
    )


# ---------------------------------------------------------------------------
# Instructions templates
# ---------------------------------------------------------------------------

INSTRUCTIONS = {
    "easy_pair_check": (
        "You are a clinical pharmacist. A patient is prescribed the following 2 medications. "
        "Determine if there is a drug-drug interaction between them.\n\n"
        "Respond with a JSON object containing an 'interactions_found' array. "
        "Each interaction should have: drug_a, drug_b, severity (minor/moderate/major/contraindicated), "
        "clinical_effect (brief description), recommendation (monitor/adjust_dose/substitute/discontinue).\n\n"
        "If no interaction exists, return an empty interactions_found array."
    ),
    "medium_multi_drug": (
        "You are a clinical pharmacist reviewing a patient's medication list of 5 drugs. "
        "Identify ALL drug-drug interactions among these medications.\n\n"
        "Respond with a JSON object containing an 'interactions_found' array. "
        "For each interaction: drug_a, drug_b (lowercase), severity (minor/moderate/major/contraindicated), "
        "clinical_effect (brief description), recommendation (monitor/adjust_dose/substitute/discontinue).\n\n"
        "Be thorough - check all possible pairs."
    ),
    "hard_polypharmacy": (
        "You are a clinical pharmacist reviewing a complex polypharmacy case with 8 medications. "
        "This elderly patient requires careful medication review.\n\n"
        "Identify ALL drug-drug interactions. For each interaction provide:\n"
        "- drug_a and drug_b (lowercase generic names)\n"
        "- severity: minor | moderate | major | contraindicated\n"
        "- clinical_effect: explain the mechanism and clinical consequence\n"
        "- recommendation: monitor | adjust_dose | substitute | discontinue\n\n"
        "Respond with a JSON object containing an 'interactions_found' array.\n"
        "Accuracy of severity classification AND recommendation is critical."
    ),
}


class PharmaDDIEnvironment(Environment):
    """
    Pharmaceutical Drug-Drug Interaction checking environment.

    The agent reviews patient medication lists and identifies dangerous
    drug interactions, simulating a real clinical pharmacist workflow.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = False

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._scenario: Optional[PatientScenario] = None
        self._current_task: str = "easy_pair_check"
        self._done: bool = False
        self._last_score: float = 0.0
        self._episode_seed: int = 42

    def reset(self, task_name: str = None, seed: int = None) -> PharmaDDIObservation:
        """Reset the environment with a new patient scenario."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._done = False
        self._last_score = 0.0

        if task_name and task_name in TASK_DEFINITIONS:
            self._current_task = task_name
        # Default to easy_pair_check if not specified

        if seed is not None:
            self._episode_seed = seed

        # Generate scenario
        generator = TASK_DEFINITIONS[self._current_task]["generator"]
        self._scenario = generator(self._episode_seed)

        return self._build_observation(
            feedback="Environment reset. Review the patient's medications and identify drug interactions.",
            score=0.0,
        )

    def step(self, action: PharmaDDIAction) -> PharmaDDIObservation:
        """
        Grade the agent's drug interaction analysis.

        The agent submits its identified interactions and the environment
        scores them against ground truth.
        """
        if self._done:
            return self._build_observation(
                feedback="Episode already complete. Call reset() to start a new episode.",
                score=self._last_score,
                done=True,
            )

        self._state.step_count += 1

        if self._scenario is None:
            return self._build_observation(
                feedback="No scenario loaded. Call reset() first.",
                score=0.0,
                done=True,
            )

        # Grade the submission
        score, feedback = self._grade_submission(action)
        self._last_score = score
        self._done = True

        return self._build_observation(
            feedback=feedback,
            score=score,
            done=True,
        )

    @property
    def state(self) -> State:
        return self._state

    # ------------------------------------------------------------------
    # Grading logic
    # ------------------------------------------------------------------

    def _grade_submission(self, action: PharmaDDIAction) -> tuple:
        """Grade the agent's interaction analysis against ground truth."""
        scenario = self._scenario
        ground_truth = scenario.ground_truth_interactions
        submitted = action.interactions_found

        # Build ground truth lookup
        gt_map: Dict[frozenset, Any] = {}
        for ix in ground_truth:
            key = frozenset({ix.drug_a.lower(), ix.drug_b.lower()})
            gt_map[key] = ix

        total_gt = len(ground_truth)
        if total_gt == 0:
            # No interactions scenario
            if len(submitted) == 0:
                return 1.0, "Perfect! Correctly identified no interactions."
            else:
                penalty = min(len(submitted) * 0.2, 1.0)
                return max(0.0, 1.0 - penalty), f"False positives: {len(submitted)} interactions reported but none exist."

        # Scoring components
        correct_pairs = 0
        correct_severity = 0
        correct_recommendation = 0
        false_positives = 0
        matched_keys = set()
        feedback_lines = []

        for sub in submitted:
            key = frozenset({sub.drug_a.lower(), sub.drug_b.lower()})
            if key in gt_map:
                matched_keys.add(key)
                gt_ix = gt_map[key]
                correct_pairs += 1

                # Severity scoring
                if sub.severity.lower().strip() == gt_ix.severity.lower():
                    correct_severity += 1
                    feedback_lines.append(
                        f"  [OK] {sub.drug_a}{sub.drug_b}: correct pair & severity ({gt_ix.severity})"
                    )
                else:
                    # Partial credit for close severity
                    sub_order = SEVERITY_ORDER.get(sub.severity.lower().strip(), 0)
                    gt_order = SEVERITY_ORDER.get(gt_ix.severity.lower(), 0)
                    if abs(sub_order - gt_order) == 1:
                        correct_severity += 0.5
                        feedback_lines.append(
                            f"  ~ {sub.drug_a}{sub.drug_b}: correct pair, severity close "
                            f"(said {sub.severity}, actual {gt_ix.severity})"
                        )
                    else:
                        feedback_lines.append(
                            f"   {sub.drug_a}{sub.drug_b}: correct pair but wrong severity "
                            f"(said {sub.severity}, actual {gt_ix.severity})"
                        )

                # Recommendation scoring (hard task gets extra weight)
                if sub.recommendation.lower().strip() == gt_ix.recommendation.lower():
                    correct_recommendation += 1
                elif sub.recommendation.lower().strip() in ("monitor", "adjust_dose", "substitute", "discontinue"):
                    correct_recommendation += 0.25  # partial credit for valid but wrong
            else:
                false_positives += 1
                feedback_lines.append(
                    f"   {sub.drug_a}{sub.drug_b}: FALSE POSITIVE - no known interaction"
                )

        # Missed interactions
        missed = []
        for key, gt_ix in gt_map.items():
            if key not in matched_keys:
                missed.append(gt_ix)
                severity_tag = " CRITICAL" if gt_ix.severity in ("major", "contraindicated") else "missed"
                feedback_lines.append(
                    f"   MISSED ({severity_tag}): {gt_ix.drug_a}{gt_ix.drug_b} ({gt_ix.severity})"
                )

        # Calculate score based on task difficulty
        task = self._current_task

        if task == "easy_pair_check":
            # Easy: 60% pair identification, 40% severity
            pair_score = correct_pairs / total_gt if total_gt > 0 else 0
            sev_score = correct_severity / total_gt if total_gt > 0 else 0
            fp_penalty = false_positives * 0.15
            raw_score = (0.6 * pair_score) + (0.4 * sev_score) - fp_penalty

        elif task == "medium_multi_drug":
            # Medium: 40% pair identification, 35% severity, 10% recommendations, 15% no false positives
            pair_score = correct_pairs / total_gt if total_gt > 0 else 0
            sev_score = correct_severity / total_gt if total_gt > 0 else 0
            rec_score = correct_recommendation / total_gt if total_gt > 0 else 0
            fp_penalty = false_positives * 0.1
            # Extra penalty for missed critical interactions
            critical_missed = sum(1 for m in missed if m.severity in ("major", "contraindicated"))
            critical_penalty = critical_missed * 0.1
            raw_score = (0.4 * pair_score) + (0.35 * sev_score) + (0.1 * rec_score) + (0.15 * (1.0 if false_positives == 0 else max(0, 1.0 - fp_penalty))) - critical_penalty

        else:  # hard_polypharmacy
            # Hard: 30% pair, 25% severity, 25% recommendations, 10% no FP, 10% completeness
            pair_score = correct_pairs / total_gt if total_gt > 0 else 0
            sev_score = correct_severity / total_gt if total_gt > 0 else 0
            rec_score = correct_recommendation / total_gt if total_gt > 0 else 0
            fp_penalty = false_positives * 0.08
            completeness = 1.0 - (len(missed) / total_gt) if total_gt > 0 else 0
            critical_missed = sum(1 for m in missed if m.severity in ("major", "contraindicated"))
            critical_penalty = critical_missed * 0.12
            raw_score = (
                0.30 * pair_score +
                0.25 * sev_score +
                0.25 * rec_score +
                0.10 * (1.0 if false_positives == 0 else max(0, 1.0 - fp_penalty)) +
                0.10 * completeness -
                critical_penalty
            )

        final_score = min(max(raw_score, 0.0), 1.0)

        # Build summary feedback
        summary = (
            f"Score: {final_score:.3f}/1.000\n"
            f"Interactions found: {correct_pairs}/{total_gt} | "
            f"Severity correct: {correct_severity}/{total_gt} | "
            f"Recommendations correct: {correct_recommendation}/{total_gt} | "
            f"False positives: {false_positives} | "
            f"Missed: {len(missed)}\n"
            f"Details:\n" + "\n".join(feedback_lines)
        )

        return final_score, summary

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_observation(
        self,
        feedback: str = "",
        score: float = 0.0,
        done: bool = False,
    ) -> PharmaDDIObservation:
        """Build an observation from the current scenario."""
        scenario = self._scenario

        if scenario is None:
            return PharmaDDIObservation(
                task_name="",
                task_difficulty="",
                patient_id="",
                patient_age=0,
                patient_conditions=[],
                medications=[],
                num_medications=0,
                instructions="Call reset() to start.",
                feedback=feedback,
                score=score,
                total_interactions=0,
                done=done,
                reward=score,
            )

        meds = [
            MedicationInfo(
                name=d.name,
                therapeutic_class=d.therapeutic_class,
                common_dose=d.common_dose,
                frequency=d.frequency,
            )
            for d in scenario.medications
        ]

        # Only hint interaction count for easy mode
        hint_count = len(scenario.ground_truth_interactions) if scenario.task_name == "easy_pair_check" else 0

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
            reward=score,
            metadata={
                "episode_id": self._state.episode_id,
                "step": self._state.step_count,
                "task": scenario.task_name,
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
