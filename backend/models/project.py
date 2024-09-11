from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class Project:
    roll_no: str
    github_link: str
    complexity: float
    total_lines: int
    halstead_metrics: Dict[str, float]
    files_content: Dict[str, str]
    score: float = 0.0
    plagiarism_report: Dict[str, Any] = field(default_factory=dict)