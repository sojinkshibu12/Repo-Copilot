from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Issue:
    id: int
    repo: str
    number: int
    title: str
    body: str
    author: str
    labels: list[str] = field(default_factory=list)
    created_at: datetime | None = None
    url: str = ""
