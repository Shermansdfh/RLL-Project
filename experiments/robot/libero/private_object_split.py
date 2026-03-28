"""Shared Stage 1 / Stage 2 private split definitions for LIBERO-Object."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional


LIBERO_OBJECT_DATASET_NAME = "libero_object_no_noops"
LIBERO_OBJECT_TASK_SUITE = "libero_object"


@dataclass(frozen=True)
class LiberoTaskInfo:
    task_id: int
    task_name: str
    language: str


@dataclass(frozen=True)
class LiberoObjectPrivateSplitDefinition:
    split_name: str
    task_suite_name: str
    dataset_name: str
    task_ids: tuple[int, ...]
    train_demo_ids: tuple[int, ...]
    val_demo_ids: tuple[int, ...]


PRIVATE_LIBERO_OBJECT_SPLITS: Dict[str, LiberoObjectPrivateSplitDefinition] = {
    "stage1": LiberoObjectPrivateSplitDefinition(
        split_name="stage1",
        task_suite_name=LIBERO_OBJECT_TASK_SUITE,
        dataset_name=LIBERO_OBJECT_DATASET_NAME,
        task_ids=(0, 1, 2),
        train_demo_ids=tuple(range(40)),
        val_demo_ids=tuple(range(10)),
    ),
    "stage2": LiberoObjectPrivateSplitDefinition(
        split_name="stage2",
        task_suite_name=LIBERO_OBJECT_TASK_SUITE,
        dataset_name=LIBERO_OBJECT_DATASET_NAME,
        task_ids=tuple(range(10)),
        train_demo_ids=tuple(range(40)),
        val_demo_ids=tuple(range(10)),
    ),
}


def get_private_libero_object_split(split_name: str) -> LiberoObjectPrivateSplitDefinition:
    try:
        split = PRIVATE_LIBERO_OBJECT_SPLITS[split_name]
    except KeyError as exc:
        valid = ", ".join(sorted(PRIVATE_LIBERO_OBJECT_SPLITS))
        raise ValueError(f"Unknown LIBERO-Object private split `{split_name}`. Expected one of: {valid}.") from exc

    _validate_demo_ids(split.train_demo_ids, "train_demo_ids")
    _validate_demo_ids(split.val_demo_ids, "val_demo_ids")
    return split


def get_private_libero_object_task_infos(
    split_name: str,
    task_provider: Optional[Callable[[str], Iterable[LiberoTaskInfo]]] = None,
) -> List[LiberoTaskInfo]:
    split = get_private_libero_object_split(split_name)
    provider = task_provider or _default_task_provider
    tasks_by_id = {task.task_id: task for task in provider(split.task_suite_name)}

    missing_task_ids = [task_id for task_id in split.task_ids if task_id not in tasks_by_id]
    if missing_task_ids:
        raise ValueError(
            f"LIBERO suite `{split.task_suite_name}` is missing task IDs required by split "
            f"`{split_name}`: {missing_task_ids}"
        )

    task_infos = [tasks_by_id[task_id] for task_id in split.task_ids]
    _validate_unique_task_languages(task_infos, split_name)
    return task_infos


def build_rlds_private_split_selection(
    split_name: str,
    split_kind: str,
    task_provider: Optional[Callable[[str], Iterable[LiberoTaskInfo]]] = None,
) -> Dict[str, object]:
    split = get_private_libero_object_split(split_name)
    if split_kind not in {"train", "val"}:
        raise ValueError(f"Unsupported split_kind `{split_kind}`. Expected `train` or `val`.")

    task_infos = get_private_libero_object_task_infos(split_name, task_provider=task_provider)
    selected_demo_ids = split.train_demo_ids if split_kind == "train" else split.val_demo_ids

    return {
        "split_name": split.split_name,
        "task_suite_name": split.task_suite_name,
        "dataset_name": split.dataset_name,
        "split_kind": split_kind,
        "task_ids": list(split.task_ids),
        "task_languages": [task.language for task in task_infos],
        "selected_demo_ids": list(selected_demo_ids),
        "max_demos_per_task": len(selected_demo_ids),
    }


def build_private_split_summary(
    split_name: str,
    task_provider: Optional[Callable[[str], Iterable[LiberoTaskInfo]]] = None,
) -> Dict[str, object]:
    split = get_private_libero_object_split(split_name)
    task_infos = get_private_libero_object_task_infos(split_name, task_provider=task_provider)
    return {
        "split_name": split.split_name,
        "task_suite_name": split.task_suite_name,
        "dataset_name": split.dataset_name,
        "task_ids": list(split.task_ids),
        "train_demo_ids": list(split.train_demo_ids),
        "val_demo_ids": list(split.val_demo_ids),
        "task_names": [task.task_name for task in task_infos],
        "task_languages": [task.language for task in task_infos],
    }

def _validate_demo_ids(demo_ids: tuple[int, ...], field_name: str) -> None:
    expected = tuple(range(len(demo_ids)))
    if demo_ids != expected:
        raise ValueError(
            f"{field_name} must be a zero-based contiguous range for deterministic RLDS filtering. "
            f"Expected {list(expected)}, got {list(demo_ids)}."
        )


def _validate_unique_task_languages(task_infos: List[LiberoTaskInfo], split_name: str) -> None:
    languages_to_task_ids: Dict[str, List[int]] = {}
    for task in task_infos:
        languages_to_task_ids.setdefault(task.language, []).append(task.task_id)

    duplicates = {
        language: task_ids
        for language, task_ids in languages_to_task_ids.items()
        if len(task_ids) > 1
    }
    if duplicates:
        raise ValueError(
            f"LIBERO-Object private split `{split_name}` requires unique task languages, "
            f"but found duplicates: {duplicates}"
        )


def _default_task_provider(task_suite_name: str) -> Iterable[LiberoTaskInfo]:
    try:
        from libero.libero import benchmark
    except ImportError as exc:
        raise ImportError(
            "Resolving a private LIBERO-Object split requires the LIBERO benchmark package. "
            "Install LIBERO as described in the README before using `--libero_object_private_split`."
        ) from exc

    benchmark_dict = benchmark.get_benchmark_dict()
    if task_suite_name not in benchmark_dict:
        raise ValueError(f"Unknown LIBERO task suite `{task_suite_name}`.")

    task_suite = benchmark_dict[task_suite_name]()
    task_infos = []
    for task_id in range(task_suite.n_tasks):
        task = task_suite.get_task(task_id)
        task_infos.append(LiberoTaskInfo(task_id=task_id, task_name=task.name, language=task.language))
    return task_infos
