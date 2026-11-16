"""A tiny hand-built OULAD-shaped fixture (3 students, 1 course) for unit tests.

Values mimic the raw CSVs, including the '?' missing marker and the '10-20' IMD quirk.
"""

from __future__ import annotations

import pandas as pd
import pytest

from riskml.oulad.load import OuladTables, clean_tables

KEY = {"code_module": "AAA", "code_presentation": "2013J"}


def raw_frames() -> dict[str, pd.DataFrame]:
    courses = pd.DataFrame([{**KEY, "module_presentation_length": 268}])
    info = pd.DataFrame(
        [
            {
                **KEY,
                "id_student": 1,
                "gender": "M",
                "region": "Scotland",
                "highest_education": "HE Qualification",
                "imd_band": "10-20",
                "age_band": "0-35",
                "num_of_prev_attempts": 0,
                "studied_credits": 60,
                "disability": "N",
                "final_result": "Pass",
            },
            {
                **KEY,
                "id_student": 2,
                "gender": "F",
                "region": "Wales",
                "highest_education": "A Level or Equivalent",
                "imd_band": "?",
                "age_band": "35-55",
                "num_of_prev_attempts": 1,
                "studied_credits": 120,
                "disability": "Y",
                "final_result": "Withdrawn",
            },
            {
                **KEY,
                "id_student": 3,
                "gender": "F",
                "region": "Wales",
                "highest_education": "Lower Than A Level",
                "imd_band": "0-10%",
                "age_band": "0-35",
                "num_of_prev_attempts": 0,
                "studied_credits": 60,
                "disability": "N",
                "final_result": "Fail",
            },
        ]
    )
    registration = pd.DataFrame(
        [
            {**KEY, "id_student": 1, "date_registration": "-30", "date_unregistration": "?"},
            {**KEY, "id_student": 2, "date_registration": "-10", "date_unregistration": "10"},
            {**KEY, "id_student": 3, "date_registration": "?", "date_unregistration": "?"},
        ]
    )
    assessments = pd.DataFrame(
        [
            {**KEY, "id_assessment": 101, "assessment_type": "TMA", "date": "20", "weight": 50.0},
            {**KEY, "id_assessment": 102, "assessment_type": "TMA", "date": "50", "weight": 50.0},
            {**KEY, "id_assessment": 199, "assessment_type": "Exam", "date": "?", "weight": 100.0},
        ]
    )
    student_assessment = pd.DataFrame(
        [
            {
                "id_assessment": 101,
                "id_student": 1,
                "date_submitted": 18,
                "is_banked": 0,
                "score": "80",
            },
            {
                "id_assessment": 102,
                "id_student": 1,
                "date_submitted": 49,
                "is_banked": 0,
                "score": "70",
            },
            {
                "id_assessment": 102,
                "id_student": 3,
                "date_submitted": 55,
                "is_banked": 0,
                "score": "30",
            },
        ]
    )
    vle = pd.DataFrame(
        [
            {"id_site": 11, **KEY, "activity_type": "oucontent", "week_from": "?", "week_to": "?"},
            {"id_site": 12, **KEY, "activity_type": "forumng", "week_from": "?", "week_to": "?"},
        ]
    )
    student_vle = pd.DataFrame(
        [
            {**KEY, "id_student": 1, "id_site": 11, "date": -5, "sum_click": 3},
            {**KEY, "id_student": 1, "id_site": 12, "date": 10, "sum_click": 5},
            {**KEY, "id_student": 1, "id_site": 11, "date": 40, "sum_click": 4},
            {**KEY, "id_student": 2, "id_site": 11, "date": 5, "sum_click": 1},
            {**KEY, "id_student": 3, "id_site": 11, "date": 25, "sum_click": 2},
        ]
    )
    return {
        "courses": courses,
        "info": info,
        "registration": registration,
        "assessments": assessments,
        "student_assessment": student_assessment,
        "vle": vle,
        "student_vle": student_vle,
    }


@pytest.fixture
def tables() -> OuladTables:
    return clean_tables(raw_frames())
