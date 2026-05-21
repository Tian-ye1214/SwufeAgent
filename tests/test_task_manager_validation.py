from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tools.ManagementTools import TaskManager


def test_create_todo_list_rejects_empty_description() -> None:
    manager = TaskManager()

    result = manager.create_todo_list('[{"id": "1", "description": "   "}]')

    assert "description must not be empty" in result
    assert manager.tasks == {}


def test_create_todo_list_rejects_duplicate_ids() -> None:
    manager = TaskManager()

    result = manager.create_todo_list(
        '[{"id": "1", "description": "first"}, {"id": "1", "description": "second"}]'
    )

    assert "duplicate task id" in result
    assert manager.tasks == {}


def test_create_todo_list_rejects_missing_dependencies() -> None:
    manager = TaskManager()

    result = manager.create_todo_list(
        '[{"id": "1", "description": "first", "dependencies": ["missing"]}]'
    )

    assert "unknown dependency" in result
    assert manager.tasks == {}


def test_create_todo_list_rejects_dependency_cycles() -> None:
    manager = TaskManager()

    result = manager.create_todo_list(
        "["
        '{"id": "1", "description": "first", "dependencies": ["2"]},'
        '{"id": "2", "description": "second", "dependencies": ["1"]}'
        "]"
    )

    assert "dependency cycle" in result
    assert manager.tasks == {}
