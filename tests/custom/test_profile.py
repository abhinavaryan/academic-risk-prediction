import pandas as pd

from riskml.custom.profile import profile


def _roles(p):
    return {c["name"]: c["role"] for c in p["columns"]}


def test_profile_suggests_target_on_uci(uci_df):
    p = profile(uci_df)
    assert p["suggested_target"] == "Target"
    assert p["suggested_positive"] == ["Dropout"]
    assert p["suggested_unknown"] == ["Enrolled"]
    assert p["suggested_negative"] == ["Graduate"]
    roles = _roles(p)
    assert roles["Admission grade"] == "numeric"
    assert roles["Mother's qualification"] == "numeric"
    target = next(c for c in p["columns"] if c["name"] == "Target")
    assert {v["value"]: v["count"] for v in target["values"]}["Dropout"] == 1421


def test_pii_and_id_columns_detected():
    n = 40
    df = pd.DataFrame(
        {
            "Roll No": [f"CS{1000 + i}" for i in range(n)],
            "Student Name": [f"Student {chr(65 + i % 26)}{i}" for i in range(n)],
            "Email": [f"s{i}@college.edu" for i in range(n)],
            "Mobile": [f"98{i:08d}" for i in range(n)],
            "Attendance": [str(50 + i) + "%" for i in range(n)],
            "Section": ["A", "B"] * (n // 2),
            "Batch": ["2022"] * n,
            "Result": ["Pass", "Fail"] * (n // 2),
        }
    )
    roles = _roles(profile(df))
    assert roles["Roll No"] == "id"
    assert roles["Student Name"] == "pii"
    assert roles["Email"] == "pii"
    assert roles["Mobile"] == "pii"
    assert roles["Attendance"] == "numeric"
    assert roles["Section"] == "categorical"
    assert roles["Batch"] == "constant"
    p = profile(df)
    assert p["suggested_id"] == "Roll No"
    assert p["suggested_target"] == "Result"
    assert p["suggested_positive"] == ["Fail"]


def test_free_text_is_text_role():
    df = pd.DataFrame(
        {
            "remarks": [f"needs help with topic {i} and more" for i in range(200)],
            "y": ["Pass", "Fail"] * 100,
        }
    )
    assert _roles(profile(df))["remarks"] == "text"
