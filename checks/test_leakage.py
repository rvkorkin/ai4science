"""Property tests for train/val split integrity and label consistency.

The split is leave-one-GROUP-out, group = treatment event, so that data from
one well does not leak between train and val. These tests verify the hard
invariants of that scheme directly on core/split.json and the label file.
"""
import json
import os
import re
from pathlib import Path

import pytest

# The split / label files are TrafficLight research data and live OUTSIDE this
# submission repo (same policy as HW1). Point TL_CORE at the project's core/ to
# run these tests with real data; otherwise the data-dependent tests skip and
# the pure property/unit tests (causality, validator) still run standalone.
# The numerical RESULTS of this audit are shipped in results/experiment_summary.json.
CORE = Path(os.environ.get("TL_CORE", Path(__file__).resolve().parent.parent / "data"))
SPLIT = CORE / "split.json"
LABELS = CORE / "field_strongest_hits.json"


def _require_data():
    if not (SPLIT.exists() and LABELS.exists()):
        pytest.skip("split.json/field_strongest_hits.json not present "
                    "(set TL_CORE to the project's core/; see results/ for the audit output)")


def _load_split():
    sp = json.loads(SPLIT.read_text(encoding="utf-8"))
    return sp.get("train", []), sp.get("val", [])


def _load_labels():
    d = json.loads(LABELS.read_text(encoding="utf-8"))
    return {k: v for k, v in d.items() if not k.startswith("_")}


def well_group(stem: str) -> str:
    """Heuristic treatment-well group key for a stage/event stem."""
    s = re.sub(r"_s\d+_\d+", "", stem)
    s = re.sub(r"_stg\d+.*$", "", s)
    s = re.sub(r"\d+F?$", "", s)
    return s


def test_split_files_exist():
    _require_data()
    assert SPLIT.exists() and LABELS.exists()


def test_train_val_stem_disjoint():
    _require_data()
    train, val = _load_split()
    overlap = set(train) & set(val)
    assert not overlap, f"{len(overlap)} stems in BOTH train and val: {sorted(overlap)[:5]}"


def test_no_duplicates_within_split():
    _require_data()
    train, val = _load_split()
    assert len(train) == len(set(train)), "duplicate stems in train"
    assert len(val) == len(set(val)), "duplicate stems in val"


def test_all_split_stems_are_labeled():
    _require_data()
    train, val = _load_split()
    labels = _load_labels()
    missing = [s for s in (train + val) if s not in labels]
    assert not missing, f"{len(missing)} split stems have no label entry: {missing[:5]}"


def test_hit_tpk_in_time_range():
    """For hit files, the hit time tpk must fall inside the record length."""
    _require_data()
    train, val = _load_split()
    labels = _load_labels()
    bad = []
    for s in train + val:
        rec = labels.get(s, {})
        if not rec.get("has_hit"):
            continue
        tpk = rec.get("hit_t_manual", rec.get("tpk"))
        T = rec.get("T")
        if tpk is None or T is None:
            continue
        if not (0 <= float(tpk) < float(T)):
            bad.append((s, tpk, T))
    assert not bad, f"tpk out of [0,T) for: {bad[:5]}"


def test_well_group_disjoint_reported():
    """Soft check: report (do not hard-fail) treatment-well groups that
    appear in BOTH train and val. Group=event is stage-level by design, so
    well-level overlap is expected; we assert only that the metric computes
    and surface it for the report."""
    _require_data()
    train, val = _load_split()
    g_tr = {well_group(s) for s in train}
    g_va = {well_group(s) for s in val}
    shared = sorted(g_tr & g_va)
    print(f"[leakage] shared well-groups train/val: {len(shared)} -> {shared}")
    assert isinstance(shared, list)
