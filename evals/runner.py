"""Eval runner — drives the agent or mock through every test case and collects results."""

import json
import logging
import time
import sys
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.scorer import Scorer, EvalReport

logger = logging.getLogger(__name__)


class EvalRunner:
    """Runs the agent (or a mock/replay function) against all test cases.

    Two modes:
      1. **Agent mode** — pass a callable that takes (title, body) and returns a dict.
      2. **Replay mode** — pass a path to pre-recorded results JSON.

    After running, it scores results and optionally runs LLM-as-judge.
    """

    def __init__(
        self,
        test_cases_dir: str,
        agent_fn: Callable | None = None,
        judge_fn: Callable | None = None,
    ):
        self.scorer = Scorer(test_cases_dir)
        self.agent_fn = agent_fn
        self.judge_fn = judge_fn

    def run(self, results_path: str | None = None) -> EvalReport:
        test_cases = self.scorer.load_test_cases()
        logger.info("EvalRunner: %d test cases loaded", len(test_cases))

        if results_path:
            with open(results_path) as f:
                agent_outputs = json.load(f)
            logger.info("Loaded pre-recorded results from %s", results_path)
        elif self.agent_fn:
            agent_outputs = self._run_agent(test_cases)
        else:
            logger.warning("No agent_fn or results_path — using mock responses")
            agent_outputs = self._mock_run(test_cases)

        judge_scores = None
        if self.judge_fn:
            judge_scores = self._run_judge(test_cases, agent_outputs)

        report = self.scorer.evaluate(agent_outputs, judge_scores=judge_scores)
        return report

    def _run_agent(self, test_cases: list[dict]) -> dict[str, dict]:
        outputs = {}
        for case in test_cases:
            case_id = case["id"]
            issue = case["issue"]
            logger.info("Running agent on %s: %s", case_id, issue["title"][:60])
            try:
                result = self.agent_fn(issue["title"], issue["body"])
                outputs[case_id] = result
            except Exception as e:
                logger.error("Agent failed on %s: %s", case_id, e)
                outputs[case_id] = {
                    "classification": "unclear",
                    "action": "no_action",
                    "commented": False,
                    "error": str(e),
                }
        return outputs

    def _run_judge(self, test_cases: list[dict], agent_outputs: dict) -> dict[str, dict]:
        scores = {}
        for case in test_cases:
            case_id = case["id"]
            actual = agent_outputs.get(case_id, {})
            try:
                if actual.get("action") == "opened_pr":
                    scores[case_id] = self.judge_fn(
                        issue_body=case["issue"]["body"],
                        pr_title=actual.get("pr_title", ""),
                        pr_body=actual.get("pr_body", ""),
                        diff=actual.get("diff", ""),
                    )
                elif actual.get("action") in ("commented",):
                    scores[case_id] = self.judge_fn(
                        issue_body=case["issue"]["body"],
                        comment_body=actual.get("comment_body", ""),
                    )
            except Exception as e:
                logger.error("Judge failed on %s: %s", case_id, e)
        return scores

    def _mock_run(self, test_cases: list[dict]) -> dict[str, dict]:
        outputs = {}
        for case in test_cases:
            case_id = case["id"]
            expected = case["expected"]
            # Mock returns perfect answers for baseline scoring
            outputs[case_id] = {
                "classification": expected["classification"],
                "action": expected["action"],
                "commented": expected.get("requires_comment", False),
                "pr_correctness": 1.0 if expected["action"] == "opened_pr" else None,
                "confidence": 0.95,
                "explanation": f"Mock returned expected {expected['classification']}",
            }
        return outputs
