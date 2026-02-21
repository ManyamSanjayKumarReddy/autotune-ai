import pathlib
import subprocess
from typing import Tuple, Optional

from langchain.tools import tool

# Project Root Configuration
_PROJECT_ROOT : Optional[pathlib.Path] = None

def set_project_root(path: str):
    global _PROJECT_ROOT
    _PROJECT_ROOT = pathlib.Path(path).resolve()

def get_project_root() -> pathlib.Path:
    if _PROJECT_ROOT is None:
        raise RuntimeError("Project root not initialized")
    return _PROJECT_ROOT

# Path Safety Utilities
def safe_path_for_project(path: str) -> pathlib.Path:
    if not path:
        raise ValueError("Path cannot be empty")

    root = get_project_root()
    candidate = (root / path).resolve()

    if candidate != root and root not in candidate.parents:
        raise ValueError("Attempt to access outside project root")

    return candidate

# File System Tools
@tool()
def write_file(path: str, content: str) -> str:
    """
    Writes full content to a file within the project root.
    Overwrites existing content.
    """
    p = safe_path_for_project(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    with open(p, "w", encoding="utf-8") as f:
        f.write(content)

    return f"WROTE: {p.relative_to(get_project_root())}"

@tool()
def read_file(path: str) -> str:
    """
    Reads content from a file within the project root.
    Returns empty string if file does not exist.
    """
    p = safe_path_for_project(path)

    if not p.exists():
        return ""

    if not p.is_file():
        return f"ERROR: {path} is not a file"

    with open(p, "r", encoding="utf-8") as f:
        return f.read()

@tool()
def list_files(directory: str = ".") -> str:
    """
    Recursively lists all files within a directory in the project root.
    """
    p = safe_path_for_project(directory)

    if not p.exists():
        return f"ERROR: Directory does not exist: {directory}"

    if not p.is_dir():
        return f"ERROR: {directory} is not a directory"

    files = sorted(
        str(f.relative_to(get_project_root()))
        for f in p.glob("**/*")
        if f.is_file()
    )

    return "\n".join(files) if files else "No files found"


@tool()
def get_current_directory() -> str:
    """
    Returns the project root directory path.
    """
    return str(get_project_root())

# Shell Execution Tool
@tool()
def run_cmd(
    cmd: str,
    cwd: Optional[str] = None,
    timeout: int = 30
) -> Tuple[int, str, str]:
    """
    Executes a shell command inside the project root or subdirectory.

    Returns:
        (return_code, stdout, stderr)
    """
    if not cmd:
        return 1, "", "ERROR: Command is empty"

    try:
        cwd_path = safe_path_for_project(cwd) if cwd else get_project_root()

        result = subprocess.run(
            cmd,
            shell=True,
            cwd=str(cwd_path),
            capture_output=True,
            text=True,
            timeout=timeout
        )

        return (
            result.returncode,
            result.stdout.strip(),
            result.stderr.strip()
        )

    except subprocess.TimeoutExpired:
        return 1, "", "ERROR: Command timed out"

    except Exception as e:
        return 1, "", f"ERROR: {str(e)}"
