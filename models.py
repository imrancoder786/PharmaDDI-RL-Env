"""
Data models for the PharmaDDI Environment.

The PharmaDDI environment simulates clinical pharmacist drug interaction
checking. An agent receives patient medication lists and must identify
drug-drug interactions, classify severity, and recommend clinical actions.

Database: 60+ drugs, 120+ interactions across all major therapeutic classes.
Grounded in DDI-Bench methodology (Shen et al., Bioinformatics 2025).
"""

from typing import List, Optional, Dict, Any
from openenv.core.env_server.types import Action, Observation
from pydantic import Field, BaseModel


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class MedicationInfo(BaseModel):
    """A single medication in the patient's list."""
    name: str = Field(..., description="Drug generic name (lowercase)")
    therapeutic_class: str = Field(..., description="Therapeutic class (e.g. SSRI, NSAID, statin)")
    common_dose: str = Field(..., description="Common dose (e.g. 20mg)")
    frequency: str = Field(..., description="Dosing frequency (e.g. once daily)")


class InteractionReport(BaseModel):
    """Agent's report of a single drug-drug interaction."""
    drug_a: str = Field(..., description="First drug name (lowercase generic)")
    drug_b: str = Field(..., description="Second drug name (lowercase generic)")
    severity: str = Field(
        ...,
        description="Interaction severity: minor | moderate | major | contraindicated"
    )
    clinical_effect: str = Field(
        default="",
        description="Description of the clinical mechanism and consequence"
    )
    recommendation: str = Field(
        default="",
        description="Recommended clinical action: monitor | adjust_dose | substitute | discontinue"
    )


# ---------------------------------------------------------------------------
# OpenEnv Action & Observation
# ---------------------------------------------------------------------------

class PharmaDDIAction(Action):
    """
    Agent's response analyzing drug interactions in a patient scenario.

    The agent submits a list of detected drug-drug interactions.
    Multi-step: the agent can refine its answer over multiple steps
    using feedback from the grader. Set done=True when confident.
    """
    interactions_found: List[InteractionReport] = Field(
        default_factory=list,
        description="List of identified drug-drug interaction pairs with clinical details"
    )
    done: bool = Field(
        default=False,
        description="Set True when the agent is confident and wants to end the episode"
    )


class PharmaDDIObservation(Observation):
    """
    Patient scenario presented to the agent for drug interaction analysis.

    Returned by both reset() and step(). After step(), the feedback field
    contains the grader's detailed assessment of the previous submission,
    and score reflects the best score achieved so far in this episode.
    """

    # --- Task identity ---
    task_name: str = Field(
        default="",
        description="Task identifier: easy_pair_check | medium_multi_drug | hard_polypharmacy"
    )
    task_difficulty: str = Field(
        default="",
        description="Difficulty level: easy | medium | hard"
    )

    # --- Patient profile ---
    patient_id: str = Field(default="", description="Unique patient identifier for this episode")
    patient_age: int = Field(default=0, description="Patient age in years")
    patient_conditions: List[str] = Field(
        default_factory=list,
        description="Patient's active medical conditions (e.g. 'atrial_fibrillation', 'type2_diabetes')"
    )
    medications: List[MedicationInfo] = Field(
        default_factory=list,
        description="Current medication list — agent must check all pairs for interactions"
    )
    num_medications: int = Field(default=0, description="Number of medications in the list")

    # --- Task guidance ---
    instructions: str = Field(
        default="",
        description="Task-specific clinical instructions for the agent"
    )

    # --- Step feedback ---
    feedback: str = Field(
        default="",
        description=(
            "Grader feedback from the most recent step. Includes: score breakdown, "
            "which pairs were correct, missed interactions, false positives, "
            "and mechanism hints for missed pharmacokinetic interactions."
        )
    )
    score: float = Field(
        default=0.0,
        description="Best score achieved so far this episode (0.0–1.0, cumulative best)"
    )

    # --- Hints ---
    total_interactions: int = Field(
        default=-1,
        description=(
            "Hint: total number of real interactions in this scenario. "
            "Provided only for easy_pair_check. "
            "-1 means not disclosed (medium and hard tasks)."
        )
    )
    actions_remaining: int = Field(
        default=10,
        description="Number of steps remaining before the episode ends automatically"
    )

    # --- Curriculum metadata ---
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Episode metadata including: episode_id, step count, task name, "
            "curriculum_focus (which drug class is being targeted this episode), "
            "and curriculum_status (agent performance per drug class category)."
        )
    )

    # --- Compatibility (kept for backward compat with older clients) ---
    done: bool = Field(default=False, description="Whether the episode has ended")
    reward: Optional[float] = Field(
        default=None,
        description="Step reward (improvement over best score so far). None after reset()."
    )