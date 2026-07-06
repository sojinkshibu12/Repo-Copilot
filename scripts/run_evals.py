#!/usr/bin/env python3
"""CLI to run the evaluation suite against the agent.

Usage:
    # Run with mock (baseline — all answers perfect):
    python scripts/run_evals.py --mock

    # Run with pre-recorded results:
    python scripts/run_evals.py --results path/to/outputs.json

    # Run with a live agent (pass a module path):
    python scripts/run_evals.py --agent my_agent:run

    # Full run with report generation:
    python scripts/run_evals.py --mock --report --format html --format markdown

    # CI mode — fail if accuracy below threshold:
    python scripts/run_evals.py --mock --min-accuracy 0.8
"""

import argparse
import importlib
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.runner import EvalRunner
from evals.report import ReportGenerator

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def resolve_agent_fn(agent_spec: str):
    """Resolve 'module:function' string to a callable."""
    if ":" not in agent_spec:
        raise ValueError("Agent spec must be in 'module:function' format")
    mod_path, fn_name = agent_spec.split(":", 1)
    module = importlib.import_module(mod_path)
    return getattr(module, fn_name)


def load_results(results_path: str) -> dict:
    with open(results_path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Run Repo Copilot evaluation suite")
    parser.add_argument(
        "--test-cases", default="evals/test_cases",
        help="Directory containing JSON test cases",
    )
    parser.add_argument(
        "--agent", default=None,
        help="Agent function as 'module:function' (e.g., 'my_agent:run')",
    )
    parser.add_argument(
        "--results", default=None,
        help="Path to pre-recorded agent outputs JSON",
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="Run with mock agent (all answers perfect — for baseline)",
    )
    parser.add_argument(
        "--judge", default=None,
        help="Judge function as 'module:function' for LLM-as-judge",
    )
    parser.add_argument(
        "--report", action="store_true",
        help="Generate report files in eval-reports/",
    )
    parser.add_argument(
        "--format", nargs="+", default=["markdown", "json"],
        choices=["markdown", "json", "html"],
        help="Report output formats",
    )
    parser.add_argument(
        "--min-accuracy", type=float, default=None,
        help="Minimum required overall score (exit code 1 if below)",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Directory for report output (default: eval-reports/)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Print detailed results")
    args = parser.parse_args()

    # Determine agent source
    agent_fn = None
    results_path = args.results

    if args.agent:
        agent_fn = resolve_agent_fn(args.agent)
        logger.info("Using agent: %s", args.agent)
    elif args.mock:
        logger.info("Using mock agent (perfect answers)")
    elif not results_path:
        logger.warning("No agent or results specified — using mock fallback")
        args.mock = True

    # Determine judge
    judge_fn = None
    if args.judge:
        judge_fn = resolve_agent_fn(args.judge)
        logger.info("Using judge: %s", args.judge)

    # Run
    runner = EvalRunner(
        test_cases_dir=args.test_cases,
        agent_fn=agent_fn,
        judge_fn=judge_fn,
    )
    report = runner.run(results_path=results_path)

    # Print report
    runner.scorer.print_report(report)

    # Generate report files
    if args.report:
        gen = ReportGenerator(output_dir=args.output_dir)
        paths = gen.generate(report, formats=args.format)
        for fmt, path in paths.items():
            print(f"  Report ({fmt}): {path}")

    # CI gate
    if args.min_accuracy is not None:
        if report.overall_score < args.min_accuracy:
            logger.error(
                "Overall score %.1f%% below minimum %.1f%% — FAILING",
                report.overall_score * 100, args.min_accuracy * 100,
            )
            sys.exit(1)
        else:
            logger.info(
                "Overall score %.1f%% meets minimum %.1f%% — PASSING",
                report.overall_score * 100, args.min_accuracy * 100,
            )

    # Exit code: 0 = no failures, 1 = failures exist
    sys.exit(0 if len(report.failures) == 0 else 1)


if __name__ == "__main__":
    main()
