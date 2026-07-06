import json
import uuid
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    case_id: str
    category: str
    classification_correct: bool
    action_correct: bool
    pr_correctness: float | None
    expected: dict
    actual: dict
    errors: list[str] = field(default_factory=list)
    judge_scores: dict | None = None


@dataclass
class CategoryBreakdown:
    name: str
    total: int
    classification_correct: int
    action_correct: int


@dataclass
class EvalReport:
    run_id: str
    timestamp: str
    total: int
    classification_accuracy: float
    action_accuracy: float
    pr_correctness: float | None
    overall_score: float
    results: list[EvalResult]
    failures: list[EvalResult]
    categories: list[CategoryBreakdown] = field(default_factory=list)


class Scorer:
    WEIGHTS = {"classification": 0.4, "action": 0.4, "pr_correctness": 0.2}

    def __init__(self, test_cases_dir: str):
        self.test_cases_dir = Path(test_cases_dir)

    def load_test_cases(self) -> list[dict]:
        cases = []
        for path in sorted(self.test_cases_dir.glob("*.json")):
            with open(path) as f:
                cases.append(json.load(f))
        return cases

    def evaluate(
        self,
        agent_outputs: dict[str, dict],
        judge_scores: dict[str, dict] | None = None,
    ) -> EvalReport:
        test_cases = self.load_test_cases()
        results = []
        category_stats: dict[str, dict] = {}

        for case in test_cases:
            case_id = case["id"]
            category = case.get("category", "unknown")
            expected = case["expected"]
            actual = agent_outputs.get(case_id, {})

            errors = []

            classification_correct = actual.get("classification") == expected["classification"]
            if not classification_correct:
                errors.append(
                    f"Classification: expected '{expected['classification']}', got '{actual.get('classification')}'"
                )

            action_correct = actual.get("action") == expected["action"]
            if not action_correct:
                errors.append(
                    f"Action: expected '{expected['action']}', got '{actual.get('action')}'"
                )

            if expected.get("requires_comment") and not actual.get("commented"):
                errors.append("Expected a comment on the issue, but none was posted")
                action_correct = False

            pr_correctness = None
            if expected["action"] == "opened_pr":
                pr_correctness = actual.get("pr_correctness")
                if pr_correctness is not None and pr_correctness < 0.5:
                    errors.append(f"PR correctness score too low: {pr_correctness:.2f}")

            cat = category_stats.setdefault(category, {"total": 0, "class_ok": 0, "action_ok": 0})
            cat["total"] += 1
            if classification_correct:
                cat["class_ok"] += 1
            if action_correct:
                cat["action_ok"] += 1

            results.append(EvalResult(
                case_id=case_id,
                category=category,
                classification_correct=classification_correct,
                action_correct=action_correct,
                pr_correctness=pr_correctness,
                expected=expected,
                actual=actual,
                errors=errors,
                judge_scores=judge_scores.get(case_id) if judge_scores else None,
            ))

        total = len(results)
        correct_classifications = sum(1 for r in results if r.classification_correct)
        correct_actions = sum(1 for r in results if r.action_correct)
        failures = [r for r in results if r.errors]

        pr_scores = [r.pr_correctness for r in results if r.pr_correctness is not None]
        avg_pr_correctness = sum(pr_scores) / len(pr_scores) if pr_scores else None

        cls_acc = correct_classifications / total if total else 0.0
        act_acc = correct_actions / total if total else 0.0

        overall = (
            cls_acc * self.WEIGHTS["classification"]
            + act_acc * self.WEIGHTS["action"]
            + (avg_pr_correctness if avg_pr_correctness else 0.0) * self.WEIGHTS["pr_correctness"]
        )

        categories = [
            CategoryBreakdown(
                name=name,
                total=stats["total"],
                classification_correct=stats["class_ok"],
                action_correct=stats["action_ok"],
            )
            for name, stats in sorted(category_stats.items())
        ]

        return EvalReport(
            run_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            total=total,
            classification_accuracy=cls_acc,
            action_accuracy=act_acc,
            pr_correctness=avg_pr_correctness,
            overall_score=overall,
            results=results,
            failures=failures,
            categories=categories,
        )

    @staticmethod
    def print_report(report: EvalReport):
        print(f"\n{'='*60}")
        print(f"  EVAL REPORT  ·  {report.run_id}")
        print(f"{'='*60}")
        print(f"  Overall score:         {report.overall_score:.1%}")
        print(f"  Classification acc:    {report.classification_accuracy:.1%}  ({int(report.classification_accuracy * report.total)}/{report.total})")
        print(f"  Action accuracy:       {report.action_accuracy:.1%}  ({int(report.action_accuracy * report.total)}/{report.total})")
        if report.pr_correctness is not None:
            print(f"  PR correctness:        {report.pr_correctness:.1%}")
        print(f"  Total cases:           {report.total}")
        print(f"  Failures:              {len(report.failures)}")
        print()

        if report.categories:
            print("  Per-category breakdown:")
            for cat in report.categories:
                cls_pct = cat.classification_correct / cat.total if cat.total else 0
                act_pct = cat.action_correct / cat.total if cat.total else 0
                print(f"    {cat.name:20s}  {cat.total:2d} cases  "
                      f"cls:{cls_pct:5.0%}  act:{act_pct:5.0%}")
            print()

        if report.failures:
            print("  Failed cases:")
            for f in report.failures:
                print(f"    [{f.case_id}] ({f.category})")
                for err in f.errors:
                    print(f"      - {err}")

        print(f"{'='*60}\n")
