from __future__ import annotations

from dataclasses import asdict, dataclass

SEVERITY_PENALTY = {"error": 40, "warning": 15, "info": 3}
SEVERITY_ORDER = {"error": 0, "warning": 1, "info": 2}


@dataclass
class Issue:
    episode: int
    check: str
    severity: str  # "error" | "warning" | "info"
    message: str
    start_s: float | None = None  # seconds from episode start
    end_s: float | None = None
    camera: str | None = None

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


def episode_score(issues: list[Issue]) -> int:
    """100 = nothing found. Each issue type counts once per episode."""
    seen: dict[tuple[str, str | None], str] = {}
    for i in issues:
        key = (i.check, i.camera)
        if key not in seen or SEVERITY_ORDER[i.severity] < SEVERITY_ORDER[seen[key]]:
            seen[key] = i.severity
    penalty = sum(SEVERITY_PENALTY[s] for s in seen.values())
    return max(0, 100 - penalty)


def grade(issues: list[Issue]) -> str:
    """bad = at least one error, review = at least one warning, ok = info only or nothing."""
    sev = {i.severity for i in issues}
    if "error" in sev:
        return "bad"
    if "warning" in sev:
        return "review"
    return "ok"
