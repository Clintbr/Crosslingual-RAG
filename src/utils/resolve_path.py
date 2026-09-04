from pathlib import Path


def resolve_project_path(relative_path: str) -> Path:
    current = Path(__file__).resolve()

    for parent in [current, *current.parents]:
        if (parent/".checkpoint").exists():
            return parent/relative_path

    raise RuntimeError("Project root could not be determined.")