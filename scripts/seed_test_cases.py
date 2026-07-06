#!/usr/bin/env python3
"""Generate 25+ synthetic test cases for the eval harness.

Usage:
    python scripts/seed_test_cases.py          # generate all cases
    python scripts/seed_test_cases.py --force  # overwrite existing
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TEST_CASES_DIR = Path(__file__).resolve().parent.parent / "evals" / "test_cases"

BUG_SCOPED = [
    {
        "title": "Typo in README: 'instalation' should be 'installation'",
        "body": "On line 42 of README.md, there's a typo: 'instalation' should be 'installation'. Quick fix.",
        "classification": "bug", "action": "opened_pr", "requires_comment": False,
    },
    {
        "title": "Off-by-one error in pagination when page > total",
        "body": "In `src/utils/pagination.py`, the `get_page` function returns an extra empty page when the requested page number exceeds the total number of pages. The condition on line 23 should use `>=` instead of `>`.",
        "classification": "bug", "action": "opened_pr", "requires_comment": False,
    },
    {
        "title": "Missing null check in user login endpoint",
        "body": "In `src/auth/login.py`, the `authenticate_user` function on line 34 doesn't check if the user object is None before accessing its properties. This causes a `AttributeError: 'NoneType' object has no attribute 'password'` when a non-existent username is submitted.",
        "classification": "bug", "action": "opened_pr", "requires_comment": False,
    },
    {
        "title": "Database connection timeout on deployment",
        "body": "The DATABASE_URL environment variable in production points to `localhost:5432` instead of the RDS endpoint. Change it in the deployment config.",
        "classification": "bug", "action": "opened_pr", "requires_comment": False,
    },
    {
        "title": "Broken link in footer documentation",
        "body": "The 'API Reference' link in the footer points to `/docs/api/v1` which returns a 404. It should point to `/docs/api/v2`.",
        "classification": "bug", "action": "opened_pr", "requires_comment": False,
    },
    {
        "title": "Unclosed file handle in log writer",
        "body": "In `src/utils/logger.py`, the `write_log` function opens a file but never closes it. This causes 'too many open files' errors under load. Add a `with` statement or explicit `.close()`.",
        "classification": "bug", "action": "opened_pr", "requires_comment": False,
    },
    {
        "title": "Incorrect import path for ConfigParser",
        "body": "Line 3 of `src/config.py` says `from configparser import ConfigParser` but the module is at `src/utils/config_parser.py`. The import should be `from src.utils.config_parser import ConfigParser`.",
        "classification": "bug", "action": "opened_pr", "requires_comment": False,
    },
    {
        "title": "Hardcoded API key in source code",
        "body": "In `src/services/external.py`, line 15 has `api_key = 'sk-1234567890abcdef'` hardcoded. This should be read from an environment variable.",
        "classification": "bug", "action": "opened_pr", "requires_comment": False,
    },
    {
        "title": "Rate limiter uses integer division instead of ceiling",
        "body": "In `src/middleware/ratelimit.py`, the calculation `remaining = max_burst // elapsed` should use `math.ceil` division to avoid granting extra requests on fractional seconds.",
        "classification": "bug", "action": "opened_pr", "requires_comment": False,
    },
    {
        "title": "Missing __init__.py in tests directory",
        "body": "The `tests/` directory is missing an `__init__.py` file, so `pytest` can't discover test modules in subdirectories. Add an empty `__init__.py`.",
        "classification": "bug", "action": "opened_pr", "requires_comment": False,
    },
    {
        "title": "Memory leak in WebSocket handler",
        "body": "In `src/handlers/websocket.py`, connected clients are stored in a `set` but never removed on disconnect. This causes unbounded memory growth. Add a `finally` block to remove the client.",
        "classification": "bug", "action": "opened_pr", "requires_comment": False,
    },
    {
        "title": "Sorting function mutates input list",
        "body": "The `sort_results` function in `src/utils/sorting.py` calls `.sort()` on the input list in-place instead of using `sorted()`. This mutates the caller's data unexpectedly.",
        "classification": "bug", "action": "opened_pr", "requires_comment": False,
    },
]

BUG_AMBIGUOUS = [
    {
        "title": "App crashes when I upload a file",
        "body": "I tried uploading a CSV file and the app crashed. The error was something about memory. Please fix.",
        "classification": "bug", "action": "commented", "requires_comment": True,
    },
    {
        "title": "Login button sometimes doesn't work",
        "body": "Sometimes when I click the login button nothing happens. It seems random. I'm using Chrome on Windows.",
        "classification": "bug", "action": "commented", "requires_comment": True,
    },
    {
        "title": "Performance is slow in production",
        "body": "Since the last deploy, the app feels sluggish. Pages take 5-10 seconds to load. Not sure what changed.",
        "classification": "bug", "action": "commented", "requires_comment": True,
    },
]

FEATURES = [
    {
        "title": "Add dark mode support",
        "body": "It would be great if the app had a dark mode toggle. Many users have requested this feature for better nighttime usability.",
        "classification": "feature", "action": "commented", "requires_comment": True,
    },
    {
        "title": "Export data as CSV from dashboard",
        "body": "It would be useful if users could export their data as CSV from the dashboard. This would help with data analysis workflows.",
        "classification": "feature", "action": "commented", "requires_comment": True,
    },
    {
        "title": "Add webhook support for new user signups",
        "body": "Please add a webhook that fires when a new user signs up. We want to integrate with our CRM in real time.",
        "classification": "feature", "action": "commented", "requires_comment": True,
    },
    {
        "title": "Implement two-factor authentication",
        "body": "Security is a growing concern. We need TOTP-based two-factor authentication for admin accounts.",
        "classification": "feature", "action": "commented", "requires_comment": True,
    },
    {
        "title": "Add pagination to the search results page",
        "body": "Search currently returns all results at once. For large result sets this is slow. Add pagination with 20 items per page.",
        "classification": "feature", "action": "commented", "requires_comment": True,
    },
]

DUPLICATES = [
    {
        "title": "This was already reported in issue #42",
        "body": "This is the same issue as #42 — the button alignment problem on mobile. Please close this and track it there.",
        "classification": "duplicate", "action": "commented", "requires_comment": True,
    },
    {
        "title": "Duplicate: Login crash (same as #18)",
        "body": "I'm also getting the login crash that was described in issue #18. Same error message and everything.",
        "classification": "duplicate", "action": "commented", "requires_comment": True,
    },
    {
        "title": "Another report of slow queries",
        "body": "This is the same slow query issue reported in #37. The dashboard reports take 30s to load.",
        "classification": "duplicate", "action": "commented", "requires_comment": True,
    },
]

UNCLEAR = [
    {
        "title": "My computer won't turn on",
        "body": "Since I installed your software, my computer won't turn on. Please help.",
        "classification": "unclear", "action": "commented", "requires_comment": True,
    },
    {
        "title": "App crashes sometimes",
        "body": "Hi, I was using the app and it crashed. I don't remember what I was doing. Can you fix it?",
        "classification": "unclear", "action": "commented", "requires_comment": True,
    },
    {
        "title": "Can you add more features?",
        "body": "Your app is good but it needs more features. Make it better.",
        "classification": "unclear", "action": "commented", "requires_comment": True,
    },
    {
        "title": "I have an idea",
        "body": "I think you should improve the app. Let me know if you want to hear my idea.",
        "classification": "unclear", "action": "commented", "requires_comment": True,
    },
    {
        "title": "It doesn't work",
        "body": "It doesn't work. Fix it.",
        "classification": "unclear", "action": "commented", "requires_comment": True,
    },
]

ALL_CASES = [
    *[(c, f"bug-scoped") for c in BUG_SCOPED],
    *[(c, f"bug-ambiguous") for c in BUG_AMBIGUOUS],
    *[(c, f"feature") for c in FEATURES],
    *[(c, f"duplicate") for c in DUPLICATES],
    *[(c, f"unclear") for c in UNCLEAR],
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Overwrite existing test cases")
    args = parser.parse_args()

    TEST_CASES_DIR.mkdir(parents=True, exist_ok=True)

    existing = {p.stem for p in TEST_CASES_DIR.glob("*.json")}
    count = 0

    for i, (case, category) in enumerate(ALL_CASES, start=1):
        case_id = f"{i:03d}-{category}"

        if case_id in existing and not args.force:
            print(f"  SKIP {case_id} (exists, use --force to overwrite)")
            continue

        # Remove existing numbered prefix if re-saving
        test_case = {
            "id": case_id,
            "category": category,
            "issue": {
                "title": case["title"],
                "body": case["body"],
                "repo": "owner/repo",
            },
            "expected": {
                "classification": case["classification"],
                "action": case["action"],
                "requires_comment": case["requires_comment"],
            },
        }

        path = TEST_CASES_DIR / f"{case_id}.json"
        with open(path, "w") as f:
            json.dump(test_case, f, indent=2)
        count += 1
        print(f"  CREATE {case_id} ({category}): {case['title'][:50]}...")

    print(f"\nDone. {count} test cases generated. Total: {len(list(TEST_CASES_DIR.glob('*.json')))} files.")


if __name__ == "__main__":
    main()
