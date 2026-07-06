import json
import tempfile
from pathlib import Path

import pytest
from evals.scorer import Scorer, EvalResult, EvalReport
from evals.runner import EvalRunner
from evals.report import ReportGenerator
from evals.judge_llm import LLMJudge


# ── Helpers ──────────────────────────────────────────────────────────────────

def _write_test_case(dir: Path, case_id: str, category: str, expected: dict):
    path = dir / f"{case_id}.json"
    with open(path, "w") as f:
        json.dump({
            "id": case_id,
            "category": category,
            "issue": {"title": "Test", "body": "Body", "repo": "owner/repo"},
            "expected": expected,
        }, f)


@pytest.fixture
def test_cases_dir():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _write_test_case(d, "001-bug", "bug-scoped", {"classification": "bug", "action": "opened_pr", "requires_comment": False})
        _write_test_case(d, "002-feature", "feature", {"classification": "feature", "action": "commented", "requires_comment": True})
        _write_test_case(d, "003-unclear", "unclear", {"classification": "unclear", "action": "commented", "requires_comment": True})
        _write_test_case(d, "004-bug-ambiguous", "bug-ambiguous", {"classification": "unclear", "action": "commented", "requires_comment": True})
        yield str(d)


# ── Scorer ───────────────────────────────────────────────────────────────────

class TestScorer:
    def test_all_correct(self, test_cases_dir):
        scorer = Scorer(test_cases_dir)
        outputs = {
            "001-bug": {"classification": "bug", "action": "opened_pr", "commented": False, "pr_correctness": 1.0},
            "002-feature": {"classification": "feature", "action": "commented", "commented": True},
            "003-unclear": {"classification": "unclear", "action": "commented", "commented": True},
            "004-bug-ambiguous": {"classification": "unclear", "action": "commented", "commented": True},
        }
        report = scorer.evaluate(outputs)
        assert report.classification_accuracy == 1.0
        assert report.action_accuracy == 1.0
        assert len(report.failures) == 0

    def test_all_wrong(self, test_cases_dir):
        scorer = Scorer(test_cases_dir)
        outputs = {
            "001-bug": {"classification": "feature", "action": "commented", "commented": True},
            "002-feature": {"classification": "bug", "action": "opened_pr", "commented": False},
            "003-unclear": {"classification": "bug", "action": "opened_pr", "commented": False},
            "004-bug-ambiguous": {"classification": "feature", "action": "opened_pr", "commented": False},
        }
        report = scorer.evaluate(outputs)
        assert report.classification_accuracy == 0.0
        assert report.action_accuracy == 0.0
        assert len(report.failures) == 4

    def test_missing_comment_is_failure(self, test_cases_dir):
        scorer = Scorer(test_cases_dir)
        outputs = {
            "001-bug": {"classification": "bug", "action": "opened_pr", "commented": False, "pr_correctness": 1.0},
            "002-feature": {"classification": "feature", "action": "commented", "commented": False},
            "003-unclear": {"classification": "unclear", "action": "commented", "commented": True},
            "004-bug-ambiguous": {"classification": "unclear", "action": "commented", "commented": True},
        }
        report = scorer.evaluate(outputs)
        assert report.action_accuracy < 1.0
        assert any("comment" in str(r.errors).lower() for r in report.failures)

    def test_categories_breakdown(self, test_cases_dir):
        scorer = Scorer(test_cases_dir)
        outputs = {
            "001-bug": {"classification": "bug", "action": "opened_pr", "commented": False, "pr_correctness": 1.0},
            "002-feature": {"classification": "feature", "action": "commented", "commented": True},
            "003-unclear": {"classification": "unclear", "action": "commented", "commented": True},
            "004-bug-ambiguous": {"classification": "unclear", "action": "commented", "commented": True},
        }
        report = scorer.evaluate(outputs)
        categories = {c.name: c for c in report.categories}
        assert "bug-scoped" in categories
        assert categories["bug-scoped"].classification_correct == 1

    def test_partial_accuracy(self, test_cases_dir):
        scorer = Scorer(test_cases_dir)
        outputs = {
            "001-bug": {"classification": "bug", "action": "opened_pr", "commented": False, "pr_correctness": 1.0},
            "002-feature": {"classification": "bug", "action": "opened_pr", "commented": False},
            "003-unclear": {"classification": "unclear", "action": "commented", "commented": True},
            "004-bug-ambiguous": {"classification": "unclear", "action": "commented", "commented": True},
        }
        report = scorer.evaluate(outputs)
        assert report.classification_accuracy == 0.75
        assert report.action_accuracy == 0.75


# ── Runner ───────────────────────────────────────────────────────────────────

class TestEvalRunner:
    def test_mock_run_returns_perfect(self, test_cases_dir):
        runner = EvalRunner(test_cases_dir)
        report = runner.run()
        assert report.classification_accuracy == 1.0
        assert report.action_accuracy == 1.0
        assert report.total == 4

    def test_agent_fn_called(self, test_cases_dir):
        call_log = []

        def fake_agent(title, body):
            call_log.append((title, body))
            return {"classification": "bug", "action": "opened_pr", "commented": False, "pr_correctness": 1.0}

        runner = EvalRunner(test_cases_dir, agent_fn=fake_agent)
        report = runner.run()
        assert len(call_log) == 4
        assert report.total == 4

    def test_agent_fn_error_handling(self, test_cases_dir):
        def broken_agent(title, body):
            raise RuntimeError("crash")

        runner = EvalRunner(test_cases_dir, agent_fn=broken_agent)
        report = runner.run()
        assert report.total == 4
        # 2 of 4 match 'unclear' (003, 004), so accuracy is 0.5
        assert report.classification_accuracy == 0.5

    def test_results_path(self, test_cases_dir):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({
                "001-bug": {"classification": "bug", "action": "opened_pr", "commented": False, "pr_correctness": 1.0},
                "002-feature": {"classification": "bug", "action": "opened_pr", "commented": False},
            }, f)
            results_path = f.name

        runner = EvalRunner(test_cases_dir)
        report = runner.run(results_path=results_path)
        assert report.total == 4  # still loads all test cases
        # Missing entries default to empty dict → wrong classification
        assert report.classification_accuracy < 1.0


# ── Report Generator ─────────────────────────────────────────────────────────

class TestReportGenerator:
    def test_json_report(self, test_cases_dir):
        scorer = Scorer(test_cases_dir)
        outputs = {
            "001-bug": {"classification": "bug", "action": "opened_pr", "commented": False, "pr_correctness": 1.0},
            "002-feature": {"classification": "feature", "action": "commented", "commented": True},
            "003-unclear": {"classification": "unclear", "action": "commented", "commented": True},
            "004-bug-ambiguous": {"classification": "unclear", "action": "commented", "commented": True},
        }
        report = scorer.evaluate(outputs)

        with tempfile.TemporaryDirectory() as tmp:
            gen = ReportGenerator(output_dir=tmp)
            files = gen.generate(report, formats=["json"])
            assert "json" in files
            with open(files["json"]) as f:
                data = json.load(f)
            assert data["total"] == 4
            assert data["classification_accuracy"] == 1.0

    def test_markdown_report(self, test_cases_dir):
        scorer = Scorer(test_cases_dir)
        report = scorer.evaluate({})
        with tempfile.TemporaryDirectory() as tmp:
            gen = ReportGenerator(output_dir=tmp)
            files = gen.generate(report, formats=["markdown"])
            assert "markdown" in files
            content = Path(files["markdown"]).read_text()
            assert "Eval Report" in content

    def test_html_report(self, test_cases_dir):
        scorer = Scorer(test_cases_dir)
        report = scorer.evaluate({})
        with tempfile.TemporaryDirectory() as tmp:
            gen = ReportGenerator(output_dir=tmp)
            files = gen.generate(report, formats=["html"])
            assert "html" in files
            content = Path(files["html"]).read_text()
            assert "DOCTYPE html" in content


# ── Judge (mock client) ──────────────────────────────────────────────────────

class MockJudgeLLM:
    """Mock LLM client that returns a valid judge response via the chat() interface."""
    def chat(self, messages, system=None, tools=None):
        from core.llm import LLMResponse
        return LLMResponse(
            content='{"clarity": 4, "safety": 5, "completeness": 4, "testing": 3, "overall": 4.0, "explanation": "Good PR"}',
            usage={"input_tokens": 10, "output_tokens": 5},
            finish_reason="end_turn",
        )


class TestLLMJudge:
    def test_rate_pr_parses_response(self):
        judge = LLMJudge(llm_client=MockJudgeLLM())
        scores = judge.rate_pr("issue", "title", "body", "diff")
        assert scores["clarity"] == 4
        assert scores["safety"] == 5
        assert scores["overall"] == 4.0

    def test_rate_comment_parses_response(self):
        judge = LLMJudge(llm_client=MockJudgeLLM())
        scores = judge.rate_comment("issue", "comment")
        assert "helpfulness" in scores
        assert "tone" in scores

    def test_parse_fallback_on_garbage(self):
        class GarbageLLM:
            def chat(self, messages, system=None, tools=None):
                from core.llm import LLMResponse
                return LLMResponse(content="not json", usage={}, finish_reason="end_turn")

        judge = LLMJudge(llm_client=GarbageLLM())
        scores = judge.rate_pr("issue", "title", "body", "diff")
        assert scores["clarity"] == 0
        assert scores["overall"] == 0.0
        assert "Parse failed" in scores.get("explanation", "")

    def test_handles_markdown_fence(self):
        class FenceLLM:
            def chat(self, messages, system=None, tools=None):
                from core.llm import LLMResponse
                return LLMResponse(
                    content='```json\n{"clarity": 5, "safety": 5, "completeness": 5, "testing": 5, "overall": 5.0}\n```',
                    usage={}, finish_reason="end_turn",
                )

        judge = LLMJudge(llm_client=FenceLLM())
        scores = judge.rate_pr("issue", "title", "body", "diff")
        assert scores["clarity"] == 5
        assert scores["overall"] == 5.0

    def test_version_property(self):
        judge = LLMJudge(llm_client=MockJudgeLLM())
        assert judge.version == "1.0.0"
