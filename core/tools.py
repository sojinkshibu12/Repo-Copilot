from pydantic import BaseModel


class ToolSchema(BaseModel):
    name: str
    description: str
    input_schema: dict


TOOL_DEFINITIONS: list[dict] = [
    {
        "name": "read_file",
        "description": "Read a file from the local repository. Optionally specify offset and limit to read a slice.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path relative to repo root"},
                "offset": {"type": "integer", "description": "Starting line (1-indexed)", "default": 1},
                "limit": {"type": "integer", "description": "Max lines to read", "default": 2000},
            },
            "required": ["path"],
        },
    },
    {
        "name": "search_code",
        "description": "Search codebase using regex. Returns matching files and line numbers.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regex pattern"},
                "include": {"type": "string", "description": "File glob filter (e.g., '*.py')", "default": ""},
                "path": {"type": "string", "description": "Subdirectory to scope search", "default": ""},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "glob",
        "description": "List files matching a glob pattern.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Glob pattern (e.g., 'src/**/*.py')"},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "semantic_search",
        "description": "Search codebase by semantic meaning. Use this when regex search doesn't find relevant code.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural language query about the code"},
                "k": {"type": "integer", "description": "Number of results to return", "default": 5},
            },
            "required": ["query"],
        },
    },
    {
        "name": "run_tests",
        "description": "Run project tests. Always run this after making changes to verify correctness.",
        "input_schema": {
            "type": "object",
            "properties": {
                "test_command": {"type": "string", "description": "Test command to run", "default": "pytest"},
            },
            "required": [],
        },
    },
    {
        "name": "run_command",
        "description": "Run an arbitrary command in the sandboxed environment for exploration or verification.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to run"},
                "timeout": {"type": "integer", "description": "Timeout in seconds", "default": 30},
            },
            "required": ["command"],
        },
    },
    {
        "name": "comment_on_issue",
        "description": "Post a comment on the current GitHub issue.",
        "input_schema": {
            "type": "object",
            "properties": {
                "body": {"type": "string", "description": "Comment text (use Markdown)"},
            },
            "required": ["body"],
        },
    },
    {
        "name": "add_label",
        "description": "Add a label to the current GitHub issue.",
        "input_schema": {
            "type": "object",
            "properties": {
                "label": {"type": "string", "description": "Label name (e.g., 'bug', 'enhancement', 'question')"},
            },
            "required": ["label"],
        },
    },
    {
        "name": "create_branch",
        "description": "Create a new branch from the current HEAD and switch to it.",
        "input_schema": {
            "type": "object",
            "properties": {
                "branch_name": {"type": "string", "description": "Name for the new branch"},
            },
            "required": ["branch_name"],
        },
    },
    {
        "name": "push_branch",
        "description": "Push the current branch to the remote repository. Must be called after commit_changes and before open_draft_pr.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "commit_changes",
        "description": "Stage and commit all local changes.",
        "input_schema": {
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "Commit message"},
            },
            "required": ["message"],
        },
    },
    {
        "name": "open_draft_pr",
        "description": "Open a draft pull request from the current branch to the base branch.",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "PR title"},
                "body": {"type": "string", "description": "PR description (include summary of changes, link to issue)"},
                "base": {"type": "string", "description": "Base branch (usually 'main')", "default": "main"},
            },
            "required": ["title", "body"],
        },
    },
    {
        "name": "classify_issue",
        "description": "Classify the current issue into one of: bug, feature, duplicate, unclear.",
        "input_schema": {
            "type": "object",
            "properties": {
                "classification": {
                    "type": "string",
                    "enum": ["bug", "feature", "duplicate", "unclear"],
                    "description": "Issue classification",
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Confidence in this classification",
                },
                "explanation": {"type": "string", "description": "Brief reasoning for the classification"},
            },
            "required": ["classification", "confidence", "explanation"],
        },
    },
]
