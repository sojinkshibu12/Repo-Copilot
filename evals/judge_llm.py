"""LLM-as-judge for subjective evaluation of agent outputs.

This implements an **LLM-as-judge** pattern where a separate LLM call rates
the quality of the agent's PR descriptions, comments, and classifications.

# Known limitations of LLM-as-judge:
#
# 1. **Position bias** — The judge may favour content that appears earlier.
# 2. **Self-enhancement bias** — If the judge model is the same as the agent
#    model, it may rate its own outputs higher.
# 3. **Verbosity bias** — Longer, more detailed outputs tend to score higher
#    regardless of actual quality.
# 4. **Calibration drift** — Scores are relative; a 4/5 from one run may not
#    mean the same as a 4/5 from another.
# 5. **Instruction sensitivity** — Small prompt changes produce large score
#    swings. Always compare results within the same judge prompt version.
# 6. **Not a replacement for human review** — Use as a signal, not a gate.
"""

import json
import logging

logger = logging.getLogger(__name__)

PR_JUDGE_PROMPT = """You are an expert code review judge. Evaluate the following PR.

Rate each criterion on a scale of 1-5:
1 = Poor  |  2 = Below average  |  3 = Adequate  |  4 = Good  |  5 = Excellent

Criteria:
- **CLARITY**: Is the PR description clear, well-structured, and easy to understand?
- **SAFETY**: Does the change avoid introducing security vulnerabilities, hardcoded secrets, or breaking changes?
- **COMPLETENESS**: Does the PR address the full scope of the issue without scope creep?
- **TESTING**: Does the PR include or reference appropriate tests for the change?

Respond with raw JSON only (no markdown fences):
{{"clarity": <int>, "safety": <int>, "completeness": <int>, "testing": <int>, "overall": <float>, "explanation": "<brief justification>"}}"""

COMMENT_JUDGE_PROMPT = """You are a support quality evaluator. Evaluate the following agent comment on a GitHub issue.

Rate each criterion on a scale of 1-5:
1 = Poor  |  2 = Below average  |  3 = Adequate  |  4 = Good  |  5 = Excellent

Criteria:
- **HELPFULNESS**: Does the comment ask useful clarifying questions or provide actionable next steps?
- **TONE**: Is it professional, courteous, and empathetic?
- **ACCURACY**: Does it correctly identify what information is missing or what the next steps should be?

Respond with raw JSON only (no markdown fences):
{{"helpfulness": <int>, "tone": <int>, "accuracy": <int>, "overall": <float>, "explanation": "<brief justification>"}}"""


class LLMJudge:
    """LLM-as-judge for subjective evaluation.

    Usage:
        judge = LLMJudge(llm_client)
        scores = judge.rate_pr(issue_body, pr_title, pr_body, diff)
        scores = judge.rate_comment(issue_body, comment_body)

    The returned dict contains per-criterion scores (1-5) and an overall.
    Always treat scores as relative signals, not absolute ground truth.
    """

    def __init__(self, llm_client):
        self.llm = llm_client
        self._version = "1.0.0"

    @property
    def version(self) -> str:
        return self._version

    def rate_pr(self, issue_body: str, pr_title: str, pr_body: str, diff: str) -> dict:
        """Rate a PR description and changes using LLM-as-judge."""
        prompt = f"""## Issue
{issue_body}

## PR Title
{pr_title}

## PR Description
{pr_body}

## Diff
{diff}"""

        response = self.llm.chat(
            messages=[{"role": "user", "content": prompt}],
            system=PR_JUDGE_PROMPT,
        )

        return self._parse(response, ["clarity", "safety", "completeness", "testing"])

    def rate_comment(self, issue_body: str, comment_body: str) -> dict:
        """Rate an agent comment using LLM-as-judge."""
        prompt = f"""## Issue
{issue_body}

## Agent Comment
{comment_body}"""

        response = self.llm.chat(
            messages=[{"role": "user", "content": prompt}],
            system=COMMENT_JUDGE_PROMPT,
        )

        return self._parse(response, ["helpfulness", "tone", "accuracy"])

    def rate_classification(
        self, issue_title: str, issue_body: str, classification: str, explanation: str
    ) -> dict:
        """Rate whether the classification makes sense given the issue."""
        prompt = f"""## Issue Title
{issue_title}

## Issue Body
{issue_body}

## Agent Classification
{classification}

## Agent Explanation
{explanation}

Rate the classification on:
- **REASONABLENESS**: Is this classification defensible given the issue text?
- **CONFIDENCE CALIBRATION**: Does the agent's explanation match the confidence level?

Respond with raw JSON only:
{{"reasonableness": <int 1-5>, "calibration": <int 1-5>, "overall": <float>, "explanation": "<brief>"}}"""

        response = self.llm.chat(
            messages=[{"role": "user", "content": prompt}],
            system="You are an eval judge. Rate the classification on a 1-5 scale. Return raw JSON only.",
        )

        return self._parse(response, ["reasonableness", "calibration"])

    def _parse(self, response, expected_keys: list[str]) -> dict:
        content = response.content or "{}"
        # Strip markdown fences if present
        content = content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[-1]
            content = content.rsplit("```", 1)[0]
        try:
            scores = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Failed to parse judge response: %s", content[:200])
            return {k: 0 for k in expected_keys} | {"overall": 0.0, "explanation": "Parse failed"}

        # Ensure all expected keys exist
        for k in expected_keys:
            scores.setdefault(k, 0)
        scores.setdefault("overall", 0.0)
        scores.setdefault("explanation", "")

        # Cast types
        for k in expected_keys:
            try:
                scores[k] = int(scores[k])
            except (ValueError, TypeError):
                scores[k] = 0
        try:
            scores["overall"] = float(scores["overall"])
        except (ValueError, TypeError):
            scores["overall"] = 0.0

        return scores
