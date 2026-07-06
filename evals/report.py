"""Generate rich eval reports in Markdown, JSON, and HTML formats."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from evals.scorer import EvalReport

logger = logging.getLogger(__name__)

REPORT_DIR = Path(__file__).resolve().parent.parent / "eval-reports"


class ReportGenerator:
    def __init__(self, output_dir: str | None = None):
        self.output_dir = Path(output_dir) if output_dir else REPORT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, report: EvalReport, formats: list[str] | None = None) -> dict[str, Path]:
        formats = formats or ["markdown", "json"]
        generated = {}

        if "json" in formats:
            path = self._write_json(report)
            generated["json"] = path

        if "markdown" in formats:
            path = self._write_markdown(report)
            generated["markdown"] = path

        if "html" in formats:
            path = self._write_html(report)
            generated["html"] = path

        logger.info("Reports generated: %s", {k: str(v) for k, v in generated.items()})
        return generated

    def _write_json(self, report: EvalReport) -> Path:
        path = self.output_dir / f"eval-{report.run_id}.json"
        data = {
            "run_id": report.run_id,
            "timestamp": report.timestamp,
            "overall_score": report.overall_score,
            "classification_accuracy": report.classification_accuracy,
            "action_accuracy": report.action_accuracy,
            "pr_correctness": report.pr_correctness,
            "total": report.total,
            "failures": len(report.failures),
            "categories": [
                {"name": c.name, "total": c.total,
                 "classification_correct": c.classification_correct,
                 "action_correct": c.action_correct}
                for c in report.categories
            ],
            "results": [
                {
                    "case_id": r.case_id,
                    "category": r.category,
                    "classification_correct": r.classification_correct,
                    "action_correct": r.action_correct,
                    "errors": r.errors,
                    "judge_scores": r.judge_scores,
                }
                for r in report.results
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return path

    def _write_markdown(self, report: EvalReport) -> Path:
        path = self.output_dir / f"eval-{report.run_id}.md"
        lines = [
            f"# Eval Report — {report.run_id}",
            f"",
            f"- **Date:** {report.timestamp}",
            f"- **Total cases:** {report.total}",
            f"- **Overall score:** {report.overall_score:.1%}",
            f"",
            f"## Scores",
            f"",
            f"| Metric | Score | Details |",
            f"|--------|-------|---------|",
            f"| Classification Accuracy | {report.classification_accuracy:.1%} | {int(report.classification_accuracy * report.total)}/{report.total} |",
            f"| Action Accuracy | {report.action_accuracy:.1%} | {int(report.action_accuracy * report.total)}/{report.total} |",
        ]
        if report.pr_correctness is not None:
            lines.append(f"| PR Correctness | {report.pr_correctness:.1%} | avg of PR-generating cases |")
        lines.append(f"| Failures | {len(report.failures)} | cases with at least one error |")
        lines.append(f"")

        if report.categories:
            lines.append(f"## Per-Category Breakdown")
            lines.append(f"")
            lines.append(f"| Category | Cases | Classification | Action |")
            lines.append(f"|----------|-------|---------------|--------|")
            for cat in report.categories:
                cls_pct = cat.classification_correct / cat.total if cat.total else 0
                act_pct = cat.action_correct / cat.total if cat.total else 0
                lines.append(f"| {cat.name} | {cat.total} | {cls_pct:.0%} | {act_pct:.0%} |")
            lines.append(f"")

        if report.failures:
            lines.append(f"## Failed Cases")
            lines.append(f"")
            for f in report.failures:
                lines.append(f"### {f.case_id} ({f.category})")
                for err in f.errors:
                    lines.append(f"- {err}")
                lines.append(f"")
                if f.judge_scores:
                    lines.append(f"**Judge scores:** {json.dumps(f.judge_scores)}")
                    lines.append(f"")

        with open(path, "w") as f:
            f.write("\n".join(lines))
        return path

    def _write_html(self, report: EvalReport) -> Path:
        path = self.output_dir / f"eval-{report.run_id}.html"

        rows = ""
        for r in report.results:
            status = "✅" if not r.errors else "❌"
            cls = "✓" if r.classification_correct else "✗"
            act = "✓" if r.action_correct else "✗"
            errors = "<br>".join(r.errors) if r.errors else ""
            judge = ""
            if r.judge_scores:
                judge = f"<small>{json.dumps(r.judge_scores)}</small>"
            rows += f"""<tr>
                <td>{status}</td>
                <td>{r.case_id}</td>
                <td>{r.category}</td>
                <td>{cls}</td>
                <td>{act}</td>
                <td>{errors}{judge}</td>
            </tr>"""

        cat_rows = ""
        for cat in report.categories:
            cls_pct = cat.classification_correct / cat.total if cat.total else 0
            act_pct = cat.action_correct / cat.total if cat.total else 0
            cat_rows += f"""<tr>
                <td>{cat.name}</td>
                <td>{cat.total}</td>
                <td>{cls_pct:.0%}</td>
                <td>{act_pct:.0%}</td>
            </tr>"""

        pr_line = ""
        if report.pr_correctness is not None:
            pr_line = f"<p><strong>PR Correctness:</strong> {report.pr_correctness:.1%}</p>"

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Eval Report {report.run_id}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 960px; margin: 2em auto; padding: 0 1em; color: #333; }}
  h1 {{ border-bottom: 2px solid #eee; padding-bottom: 0.3em; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
  th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
  th {{ background: #f5f5f5; }}
  .score {{ font-size: 2em; font-weight: bold; }}
  .pass {{ color: #2e7d32; }} .fail {{ color: #c62828; }}
  small {{ color: #666; }}
</style>
</head>
<body>
<h1>Eval Report</h1>
<p><strong>Run ID:</strong> {report.run_id}<br>
<strong>Date:</strong> {report.timestamp}<br>
<strong>Total cases:</strong> {report.total}</p>

<div class="score" style="text-align:center; margin:2em 0;">
  Overall: <span class="pass">{report.overall_score:.1%}</span>
</div>

<h2>Scores</h2>
<table>
  <tr><th>Metric</th><th>Score</th><th>Details</th></tr>
  <tr><td>Classification Accuracy</td><td>{report.classification_accuracy:.1%}</td><td>{int(report.classification_accuracy * report.total)}/{report.total}</td></tr>
  <tr><td>Action Accuracy</td><td>{report.action_accuracy:.1%}</td><td>{int(report.action_accuracy * report.total)}/{report.total}</td></tr>
</table>
{pr_line}
<p><strong>Failures:</strong> {len(report.failures)}</p>

<h2>Per-Category Breakdown</h2>
<table><tr><th>Category</th><th>Cases</th><th>Classification</th><th>Action</th></tr>
{cat_rows}</table>

<h2>All Results</h2>
<table><tr><th></th><th>Case</th><th>Category</th><th>Cls</th><th>Act</th><th>Errors / Judge</th></tr>
{rows}</table>
</body>
</html>"""
        with open(path, "w") as f:
            f.write(html)
        return path
