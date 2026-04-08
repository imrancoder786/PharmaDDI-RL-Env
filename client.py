# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""PharmaDDI Environment Client."""

from typing import Dict, List

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import (
    PharmaDDIAction,
    PharmaDDIObservation,
    MedicationInfo,
    InteractionReport,
)


class PharmaDDIEnv(
    EnvClient[PharmaDDIAction, PharmaDDIObservation, State]
):
    """
    Client for the PharmaDDI Environment.

    This client interacts with a drug interaction checking environment where
    an AI agent reviews patient medication lists and identifies dangerous
    drug-drug interactions.

    Example:
        >>> with PharmaDDIEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.medications)
        ...
        ...     action = PharmaDDIAction(interactions_found=[
        ...         InteractionReport(
        ...             drug_a="warfarin", drug_b="aspirin",
        ...             severity="major",
        ...             clinical_effect="Increased bleeding risk",
        ...             recommendation="monitor"
        ...         )
        ...     ])
        ...     result = client.step(action)
        ...     print(result.observation.score)
    """

    def _step_payload(self, action: PharmaDDIAction) -> Dict:
        """Convert PharmaDDIAction to JSON payload for step message."""
        return {
            "interactions_found": [
                {
                    "drug_a": r.drug_a,
                    "drug_b": r.drug_b,
                    "severity": r.severity,
                    "clinical_effect": r.clinical_effect,
                    "recommendation": r.recommendation,
                }
                for r in action.interactions_found
            ],
        }

    def _parse_result(self, payload: Dict) -> StepResult[PharmaDDIObservation]:
        """Parse server response into StepResult[PharmaDDIObservation]."""
        obs_data = payload.get("observation", {})

        medications = [
            MedicationInfo(**m) for m in obs_data.get("medications", [])
        ]

        observation = PharmaDDIObservation(
            task_name=obs_data.get("task_name", ""),
            task_difficulty=obs_data.get("task_difficulty", ""),
            patient_id=obs_data.get("patient_id", ""),
            patient_age=obs_data.get("patient_age", 0),
            patient_conditions=obs_data.get("patient_conditions", []),
            medications=medications,
            num_medications=obs_data.get("num_medications", 0),
            instructions=obs_data.get("instructions", ""),
            feedback=obs_data.get("feedback", ""),
            score=obs_data.get("score", 0.0),
            total_interactions=obs_data.get("total_interactions", 0),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """Parse server response into State object."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
