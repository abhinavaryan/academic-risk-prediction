import numpy as np

from riskml.oulad.load import clean_tables
from tests.oulad.conftest import raw_frames


def test_imd_band_normalised():
    t = clean_tables(raw_frames())
    bands = t.info.set_index("id_student")["imd_band"]
    assert bands[1] == "10-20%"
    assert bands.isna()[2]


def test_question_marks_become_nan_and_numeric():
    t = clean_tables(raw_frames())
    reg = t.registration.set_index("id_student")
    assert np.isnan(reg.loc[1, "date_unregistration"])
    assert reg.loc[2, "date_unregistration"] == 10
    assert np.isnan(reg.loc[3, "date_registration"])
    assert t.assessments["date"].isna().sum() == 1


def test_student_vle_has_activity_type():
    t = clean_tables(raw_frames())
    assert set(t.student_vle["activity_type"]) == {"oucontent", "forumng"}
