import numpy as np
import pandas as pd

from riskml.oulad.features import FEATURE_GROUPS, build_features
from riskml.oulad.load import clean_tables
from tests.oulad.conftest import KEY, raw_frames


def _row(ff, student):
    return ff.X.loc[ff.meta["id_student"] == student].iloc[0]


def test_no_future_information():
    base = build_features(clean_tables(raw_frames()), cutoff=30)
    frames = raw_frames()
    frames["student_vle"] = pd.concat(
        [
            frames["student_vle"],
            pd.DataFrame(
                [
                    {**KEY, "id_student": 1, "id_site": 12, "date": 31, "sum_click": 99},
                    {**KEY, "id_student": 3, "id_site": 11, "date": 200, "sum_click": 50},
                ]
            ),
        ],
        ignore_index=True,
    )
    frames["student_assessment"] = pd.concat(
        [
            frames["student_assessment"],
            pd.DataFrame(
                [
                    {
                        "id_assessment": 101,
                        "id_student": 3,
                        "date_submitted": 31,
                        "is_banked": 0,
                        "score": "10",
                    },
                ]
            ),
        ],
        ignore_index=True,
    )
    future = build_features(clean_tables(frames), cutoff=30)
    pd.testing.assert_frame_equal(base.X, future.X)
    pd.testing.assert_series_equal(base.y, future.y)


def test_population_excludes_already_withdrawn():
    t = clean_tables(raw_frames())
    assert 2 not in set(build_features(t, cutoff=30).meta["id_student"])
    assert 2 in set(build_features(t, cutoff=0).meta["id_student"])


def test_labels_at_risk_for_fail_and_withdrawn():
    ff = build_features(clean_tables(raw_frames()), cutoff=0)
    labels = dict(zip(ff.meta["id_student"], ff.y, strict=True))
    assert labels == {1: 0, 2: 1, 3: 1}


def test_missed_assessments_counted():
    ff = build_features(clean_tables(raw_frames()), cutoff=30)
    assert _row(ff, 3)["assess_missed"] == 1
    assert _row(ff, 1)["assess_missed"] == 0
    assert _row(ff, 1)["assess_due"] == 1


def test_weighted_score_and_lateness():
    ff = build_features(clean_tables(raw_frames()), cutoff=60)
    s1, s3 = _row(ff, 1), _row(ff, 3)
    assert s1["assess_weighted_score"] == 75.0
    assert s1["assess_n_late"] == 0
    assert s3["assess_n_late"] == 1
    assert s3["assess_n_fail"] == 1


def test_vle_window_features():
    ff = build_features(clean_tables(raw_frames()), cutoff=30)
    s1 = _row(ff, 1)
    assert s1["vle_clicks_total"] == 8  # -5 (3) + 10 (5); day 40 is in the future
    assert s1["vle_clicks_precourse"] == 3
    assert s1["vle_clicks_last14"] == 0  # window is days 17..30
    assert s1["vle_days_since_last"] == 20
    assert s1["vle_clicks_forumng"] == 5
    assert _row(ff, 3)["vle_clicks_last14"] == 2


def test_inactive_student_has_zero_clicks_not_missing():
    frames = raw_frames()
    frames["student_vle"] = frames["student_vle"][frames["student_vle"]["id_student"] != 3]
    ff = build_features(clean_tables(frames), cutoff=30)
    s3 = _row(ff, 3)
    assert s3["vle_clicks_total"] == 0
    assert s3["vle_active_days"] == 0
    assert np.isnan(s3["vle_days_since_last"])


def test_feature_groups_cover_every_column():
    ff = build_features(clean_tables(raw_frames()), cutoff=30)
    grouped = [c for cols in FEATURE_GROUPS.values() for c in cols]
    assert sorted(grouped) == sorted(ff.X.columns)
    assert set(ff.categorical) | set(ff.numeric) == set(ff.X.columns)
