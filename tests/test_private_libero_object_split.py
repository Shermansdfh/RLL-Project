from experiments.robot.libero.private_object_split import (
    LiberoTaskInfo,
    build_private_split_summary,
    build_rlds_private_split_selection,
    get_private_libero_object_split,
)


def fake_task_provider(task_suite_name):
    assert task_suite_name == "libero_object"
    return [
        LiberoTaskInfo(task_id=task_id, task_name=f"task_{task_id}", language=f"language_{task_id}")
        for task_id in range(10)
    ]


def test_stage1_private_split_definition():
    split = get_private_libero_object_split("stage1")

    assert list(split.task_ids) == [0, 1, 2]
    assert list(split.train_demo_ids) == list(range(40))
    assert list(split.val_demo_ids) == list(range(10))

    summary = build_private_split_summary("stage1", task_provider=fake_task_provider)
    assert summary["task_names"] == ["task_0", "task_1", "task_2"]
    assert summary["task_languages"] == ["language_0", "language_1", "language_2"]


def test_stage2_private_split_definition():
    split = get_private_libero_object_split("stage2")

    assert list(split.task_ids) == list(range(10))
    assert list(split.train_demo_ids) == list(range(40))
    assert list(split.val_demo_ids) == list(range(10))

    train_selection = build_rlds_private_split_selection("stage2", "train", task_provider=fake_task_provider)
    val_selection = build_rlds_private_split_selection("stage2", "val", task_provider=fake_task_provider)

    assert train_selection["task_ids"] == list(range(10))
    assert train_selection["max_demos_per_task"] == 40
    assert val_selection["max_demos_per_task"] == 10


def test_duplicate_task_languages_are_rejected():
    def duplicate_language_provider(task_suite_name):
        assert task_suite_name == "libero_object"
        return [
            LiberoTaskInfo(task_id=task_id, task_name=f"task_{task_id}", language="shared_language")
            for task_id in range(10)
        ]

    import pytest

    with pytest.raises(ValueError, match="unique task languages"):
        build_rlds_private_split_selection("stage1", "train", task_provider=duplicate_language_provider)
