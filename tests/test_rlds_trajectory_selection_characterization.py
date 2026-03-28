"""Regression tests for RLDS trajectory selection wiring."""

import types

from prismatic.vla.datasets.rlds import dataset as rlds_dataset


def _make_fake_tf():
    return types.SimpleNamespace(
        int64="int64",
        string="string",
        constant=lambda value, dtype=None: value,
        range=lambda n, dtype=None: list(range(n)),
        zeros=lambda shape, dtype=None: [0] * (shape[0] if isinstance(shape, (list, tuple)) else shape),
        data=types.SimpleNamespace(AUTOTUNE="AUTOTUNE"),
        lookup=types.SimpleNamespace(
            StaticHashTable=lambda initializer, default_value=None: object(),
            KeyValueTensorInitializer=lambda keys, values: (keys, values),
        ),
    )


class _FakeTFDataset:
    def filter(self, fn):
        return self

    def map(self, fn, num_parallel_calls=None):
        return self


class _FakeDLDataset:
    def scan(self, initial_state=None, scan_func=None):
        return _FakeTFDataset()


class _FakeDataset:
    def __init__(self, name):
        self.name = name
        self.sample_weights = None

    def repeat(self):
        return self

    def flatten(self, num_parallel_calls=None):
        return self

    def shuffle(self, shuffle_buffer_size):
        return self

    def with_ram_budget(self, budget):
        return self

    def take(self, n):
        return self

    def cache(self):
        return self


class _FakeSelectableDataset(_FakeDataset):
    def __init__(self, name, scan_calls):
        super().__init__(name)
        self._scan_calls = scan_calls

    def scan(self, initial_state=None, scan_func=None):
        self._scan_calls.append(self.name)
        return _FakeTFDataset()


class _WrappedDLDataset:
    def __init__(self, dataset):
        self.dataset = dataset

    def traj_map(self, *args, **kwargs):
        return self


def test_apply_trajectory_selection_rewraps_scan_output(monkeypatch):
    monkeypatch.setattr(rlds_dataset, "tf", _make_fake_tf())
    monkeypatch.setattr(
        rlds_dataset,
        "dl",
        types.SimpleNamespace(DLataset=_WrappedDLDataset),
    )

    result = rlds_dataset.apply_trajectory_selection(
        _FakeDLDataset(),
        {
            "dataset_name": "libero_object_no_noops",
            "task_languages": ["language_0"],
            "max_demos_per_task": 40,
            "split_name": "stage1",
            "split_kind": "train",
        },
        dataset_name="libero_object_no_noops",
    )

    assert isinstance(result, _WrappedDLDataset)
    assert isinstance(result.dataset, _FakeTFDataset)
    assert hasattr(result, "traj_map")


def test_make_interleaved_dataset_only_disables_shuffle_for_selected_dataset(monkeypatch):
    shuffle_calls = []

    monkeypatch.setattr(rlds_dataset, "allocate_threads", lambda total, weights: [1] * len(weights))
    monkeypatch.setattr(rlds_dataset, "pprint_data_mixture", lambda dataset_kwargs_list, sample_weights: None)
    monkeypatch.setattr(rlds_dataset.overwatch, "info", lambda *args, **kwargs: None)

    def fake_make_dataset_from_rlds(name, train, shuffle=True, **kwargs):
        shuffle_calls.append((name, shuffle))
        return _FakeDataset(name), {"num_transitions": 1}

    monkeypatch.setattr(rlds_dataset, "make_dataset_from_rlds", fake_make_dataset_from_rlds)
    monkeypatch.setattr(
        rlds_dataset,
        "apply_trajectory_selection",
        lambda dataset, trajectory_selection, dataset_name: dataset,
    )
    monkeypatch.setattr(rlds_dataset, "apply_trajectory_transforms", lambda dataset, **kwargs: dataset)
    monkeypatch.setattr(rlds_dataset, "apply_per_dataset_frame_transforms", lambda dataset, **kwargs: dataset)
    monkeypatch.setattr(rlds_dataset, "apply_frame_transforms", lambda dataset, **kwargs: dataset)
    monkeypatch.setattr(
        rlds_dataset,
        "dl",
        types.SimpleNamespace(
            DLataset=types.SimpleNamespace(
                sample_from_datasets=lambda datasets, sample_weights: _FakeDataset("sampled")
            )
        ),
    )

    rlds_dataset.make_interleaved_dataset(
        dataset_kwargs_list=[
            {"name": "libero_object_no_noops", "data_dir": "unused"},
            {"name": "bridge_orig", "data_dir": "unused"},
        ],
        sample_weights=[1.0, 1.0],
        train=True,
        shuffle_buffer_size=8,
        traj_transform_kwargs={},
        frame_transform_kwargs={},
        trajectory_selection={
            "dataset_name": "libero_object_no_noops",
            "task_languages": ["language_0"],
            "max_demos_per_task": 40,
            "split_name": "stage1",
            "split_kind": "train",
        },
    )

    second_pass_calls = shuffle_calls[2:]
    assert second_pass_calls == [
        ("libero_object_no_noops", False),
        ("bridge_orig", True),
    ]


def test_make_interleaved_dataset_only_applies_selection_to_selected_dataset(monkeypatch):
    scan_calls = []

    monkeypatch.setattr(rlds_dataset, "allocate_threads", lambda total, weights: [1] * len(weights))
    monkeypatch.setattr(rlds_dataset, "pprint_data_mixture", lambda dataset_kwargs_list, sample_weights: None)
    monkeypatch.setattr(rlds_dataset.overwatch, "info", lambda *args, **kwargs: None)
    monkeypatch.setattr(rlds_dataset, "tf", _make_fake_tf())
    monkeypatch.setattr(
        rlds_dataset,
        "make_dataset_from_rlds",
        lambda name, train, shuffle=True, **kwargs: (_FakeSelectableDataset(name, scan_calls), {"num_transitions": 1}),
    )
    monkeypatch.setattr(rlds_dataset, "apply_trajectory_transforms", lambda dataset, **kwargs: dataset)
    monkeypatch.setattr(rlds_dataset, "apply_per_dataset_frame_transforms", lambda dataset, **kwargs: dataset)
    monkeypatch.setattr(rlds_dataset, "apply_frame_transforms", lambda dataset, **kwargs: dataset)
    monkeypatch.setattr(
        rlds_dataset,
        "dl",
        types.SimpleNamespace(
            DLataset=type(
                "FakeDLDataset",
                (),
                {
                    "__init__": lambda self, dataset: setattr(self, "dataset", dataset),
                    "repeat": lambda self: self,
                    "flatten": lambda self, num_parallel_calls=None: self,
                    "sample_from_datasets": staticmethod(lambda datasets, sample_weights: _FakeDataset("sampled")),
                },
            )
        ),
    )

    rlds_dataset.make_interleaved_dataset(
        dataset_kwargs_list=[
            {"name": "libero_object_no_noops", "data_dir": "unused"},
            {"name": "bridge_orig", "data_dir": "unused"},
        ],
        sample_weights=[1.0, 1.0],
        train=True,
        shuffle_buffer_size=8,
        traj_transform_kwargs={},
        frame_transform_kwargs={},
        trajectory_selection={
            "dataset_name": "libero_object_no_noops",
            "task_languages": ["language_0"],
            "max_demos_per_task": 40,
            "split_name": "stage1",
            "split_kind": "train",
        },
    )

    assert scan_calls == ["libero_object_no_noops"]
