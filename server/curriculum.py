"""
Adaptive Curriculum Engine for PharmaDDI.

Tracks agent performance per drug class and generates harder scenarios
in areas where the agent is weakest — learning what the agent doesn't know.

This is the key differentiator from random scenario generation.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
import random

from .drug_data import CURRICULUM_DRUG_GROUPS


@dataclass
class ClassPerformance:
    """Tracks agent accuracy per drug interaction category."""
    drug_class: str
    total_episodes: int = 0
    total_score: float = 0.0
    recent_scores: List[float] = field(default_factory=list)
    last_focus_episode: int = 0

    @property
    def average_score(self) -> float:
        if not self.recent_scores:
            return 0.5  # assume moderate difficulty for unseen classes
        return sum(self.recent_scores) / len(self.recent_scores)

    @property
    def weakness_score(self) -> float:
        """Higher = weaker = should be focused on more."""
        return 1.0 - self.average_score

    def record(self, score: float):
        self.total_episodes += 1
        self.total_score += score
        self.recent_scores.append(score)
        # Keep only last 5 episodes for recency weighting
        if len(self.recent_scores) > 5:
            self.recent_scores.pop(0)


class CurriculumEngine:
    """
    Adaptive curriculum that tracks agent weaknesses and generates
    targeted scenarios to address them.

    How it works:
    1. Every episode is tagged with a `curriculum_focus` (drug class group)
    2. After each episode completes, the score is recorded for that class
    3. When generating the next scenario, the weakest classes are prioritized
    4. This creates a feedback loop: agent improves where it's weakest

    This implements a simplified version of the distribution-change-aware
    evaluation from DDI-Bench — the environment adapts to test the agent
    on drug classes it hasn't mastered yet.
    """

    EXPLORATION_RATE = 0.20  # 20% random exploration to avoid over-specialization

    def __init__(self):
        self.performance: Dict[str, ClassPerformance] = {
            cls: ClassPerformance(drug_class=cls)
            for cls in CURRICULUM_DRUG_GROUPS.keys()
        }
        self.episode_count = 0
        self.history: List[Dict] = []

    def select_focus(self, rng: Optional[random.Random] = None) -> str:
        """
        Select which drug class to focus the next scenario on.
        Uses weighted random selection biased toward weak areas.
        """
        if rng is None:
            rng = random.Random()

        self.episode_count += 1

        # Exploration: occasionally pick randomly to stay generalized
        if rng.random() < self.EXPLORATION_RATE:
            return "general"

        # Exploitation: weight by weakness score
        classes = list(self.performance.keys())
        weights = [max(self.performance[c].weakness_score, 0.05) for c in classes]
        total = sum(weights)
        normalized = [w / total for w in weights]

        # Weighted random choice
        r = rng.random()
        cumulative = 0.0
        for cls, weight in zip(classes, normalized):
            cumulative += weight
            if r <= cumulative:
                return cls

        return classes[-1]

    def record_episode(self, curriculum_focus: str, score: float, task_name: str):
        """Record agent performance after an episode ends."""
        if curriculum_focus in self.performance:
            self.performance[curriculum_focus].record(score)

        self.history.append({
            "episode": self.episode_count,
            "focus": curriculum_focus,
            "score": score,
            "task": task_name,
        })

    def get_status(self) -> Dict:
        """Return current curriculum state — useful for debugging and README."""
        return {
            "episode_count": self.episode_count,
            "class_performance": {
                cls: {
                    "average_score": round(p.average_score, 3),
                    "weakness_score": round(p.weakness_score, 3),
                    "episodes": p.total_episodes,
                }
                for cls, p in self.performance.items()
            },
            "weakest_class": min(
                self.performance.items(),
                key=lambda x: x[1].average_score
            )[0] if any(p.total_episodes > 0 for p in self.performance.values()) else "none_yet",
        }

    def reset_history(self):
        """Reset curriculum state (useful for evaluation vs training modes)."""
        for p in self.performance.values():
            p.total_episodes = 0
            p.total_score = 0.0
            p.recent_scores = []
        self.episode_count = 0
        self.history = []