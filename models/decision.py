from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class Classification(str, Enum):
    BUG = "bug"
    FEATURE = "feature"
    DUPLICATE = "duplicate"
    UNCLEAR = "unclear"


class DecisionAction(str, Enum):
    OPENED_PR = "opened_pr"
    COMMENTED = "commented"
    LABELED = "labeled"
    NO_ACTION = "no_action"


@dataclass
class Decision:
    issue_id: int
    classification: Classification
    action: DecisionAction
    explanation: str
    confidence: float = 0.0
    pr_url: str | None = None
    created_at: datetime | None = None
