"""Manifest loader.

Supports:
- JSON: {"documents": ["path1", "path2"]}
- YAML: see manifest.example.yaml

YAML supports per-document metadata so we can later implement role-based filtering and nicer citations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ManifestDoc:
    id: Optional[str]
    title: Optional[str]
    path: str
    tags: List[str]
    allowed_roles: List[str]


def load_manifest(path: str) -> List[ManifestDoc]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    if p.suffix.lower() in {".yaml", ".yml"}:
        return _load_yaml(p)
    return _load_json(p)


def _load_json(p: Path) -> List[ManifestDoc]:
    data = json.loads(p.read_text(encoding="utf-8"))
    docs = []
    for item in data.get("documents", []):
        docs.append(ManifestDoc(id=None, title=None, path=str(item), tags=[], allowed_roles=[]))
    return docs


def _load_yaml(p: Path) -> List[ManifestDoc]:
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError("PyYAML is required for YAML manifests. Install pyyaml.") from e

    data: Dict[str, Any] = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    out: List[ManifestDoc] = []
    for d in data.get("documents", []) or []:
        out.append(
            ManifestDoc(
                id=d.get("id"),
                title=d.get("title"),
                path=str(d.get("path")),
                tags=list(d.get("tags") or []),
                allowed_roles=list(d.get("allowed_roles") or []),
            )
        )
    return out
