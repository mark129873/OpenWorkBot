import fnmatch
from dataclasses import dataclass
from pathlib import PurePosixPath


IGNORE_PATTERNS = (
    ".git", ".svn", ".hg", ".bzr", ".tool-results", "node_modules", "__pycache__", ".venv", "venv",
    ".env", "env", ".tox", ".nox", ".eggs", "*.egg-info", "site-packages", "dist", "build",
    ".next", ".nuxt", ".output", ".turbo", "target", "out", ".idea", ".vscode", "*.swp",
    "*.swo", "*~", ".project", ".classpath", ".settings", ".DS_Store", "Thumbs.db", "desktop.ini",
    "*.lnk", "*.log", "*.tmp", "*.temp", ".upload-*.part", "*.bak", "*.cache", ".cache", "logs",
    ".coverage", "coverage", ".nyc_output", "htmlcov", ".pytest_cache", ".mypy_cache", ".ruff_cache",
)


@dataclass(frozen=True)
class GrepMatch:
    path: str
    line_number: int
    line: str


def ignored(path: str, root: str) -> bool:
    root = root.rstrip("/") or "/"
    if path == root:
        return False
    prefix = "/" if root == "/" else root + "/"
    relative = path[len(prefix):] if path.startswith(prefix) else path
    return any(fnmatch.fnmatchcase(part, pattern) for part in relative.split("/")
               for pattern in IGNORE_PATTERNS)


def path_matches(pattern: str, path: str) -> bool:
    if "/" not in pattern:
        return PurePosixPath(path).match(pattern)
    parts = PurePosixPath(path).parts
    positions = {0}
    for part in PurePosixPath(pattern).parts:
        if part == "**":
            positions = set(range(min(positions), len(parts) + 1))
        else:
            positions = {index + 1 for index in positions if index < len(parts)
                         and fnmatch.fnmatchcase(parts[index], part)}
        if not positions:
            return False
    return len(parts) in positions
