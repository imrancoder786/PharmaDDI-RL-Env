# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the PharmaDDI Environment.

The PharmaDDI environment simulates clinical pharmacist drug interaction
checking. An agent receives patient medication lists and must identify
drug-drug interactions, classify severity, and recommend clinical actions.
"""

from typing import List, Optional, Dict, Any
from openenv.core.env_server.types import Action, Observation
from pydantic import Field, BaseModel


# ---------------------------------------------------------------------------
# Sub-models used inside Action / Observation
# ---------------------------------------------------------------------------

class MedicationInfo(BaseModel):
    """A single medication in the patient's list."""
    name: str = Field(..., description="Drug generic name")
    therapeutic_class: str = Field(..., description="Therapeutic class (e.g. SSRI, NSAID)")
    common_dose: str = Field(..., description="Common dose (e.g. 20mg)")
    frequency: str = Field(..., description="Dosing frequency (e.g. once daily)")


class InteractionReport(BaseModel):
    """Agent's report of a single drug-drug interaction."""
    drug_a: str = Field(..., description="First drug name (lowercase)")
    drug_b: str = Field(..., description="Second drug name (lowercase)")
    severity: str = Field(..., description="Severity: minor | moderate | major | contraindicated")
    clinical_effect: str = Field(default="", description="Description of the clinical effect")
    recommendation: str = Field(default="", description="Recommended action: monitor | adjust_dose | substitute | discontinue")


# ---------------------------------------------------------------------------
# OpenEnv Action & Observation
# ---------------------------------------------------------------------------

class PharmaDDIAction(Action):
    """Agent's response analyzing drug interactions in a patient scenario."""
    interactions_found: List[InteractionReport] = Field(
        default_factory=list,
        description="List of identified drug-drug interaction pairs with details"
    )


class PharmaDDIObservation(Observation):
    """Patient scenario presented to the agent for drug interaction analysis."""
    task_name: str = Field(default="", description="Task identifier: easy_pair_check | medium_multi_drug | hard_polypharmacy")
    task_difficulty: str = Field(default="", description="Difficulty level: easy | medium | hard")
    patient_id: str = Field(default="", description="Unique patient identifier")
    patient_age: int = Field(default=0, description="Patient age in years")
    patient_conditions: List[str] = Field(default_factory=list, description="Patient medical conditions")
    medications: List[MedicationInfo] = Field(default_factory=list, description="Current medication list")
    num_medications: int = Field(default=0, description="Number of medications in the list")
    instructions: str = Field(default="", description="Task-specific instructions for the agent")
    feedback: str = Field(default="", description="Grader feedback from the evaluation")
    score: float = Field(default=0.0, description="Score achieved (0.0-1.0)")
    total_interactions: int = Field(default=0, description="Hint: total number of real interactions (only shown in easy mode)")
