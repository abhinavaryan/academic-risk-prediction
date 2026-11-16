import io
import zipfile

import pandas as pd
import pytest

from riskml.config import MAX_COLS, MAX_UPLOAD_BYTES
from riskml.custom.io import UploadError, read_table


def test_reads_uci_semicolon_bom_tab_header(uci_df):
    assert uci_df.shape == (4424, 37)
    assert uci_df.columns[0] == "Marital status"  # BOM stripped
    assert "Daytime/evening attendance" in uci_df.columns  # trailing tab stripped
    assert uci_df["Target"].value_counts().to_dict() == {
        "Graduate": 2209,
        "Dropout": 1421,
        "Enrolled": 794,
    }


def test_cp1252():
    data = "Name;Résultat;Marks\nJosé;Pass;71\nRené;Fail;22\n".encode("cp1252")
    df = read_table(data, "marks.csv")
    assert list(df.columns) == ["Name", "Résultat", "Marks"]
    assert df.loc[0, "Name"] == "José"


def test_comma_csv_with_quotes_and_percent():
    data = b'roll,att,remark\n1,"85%","late, twice"\n2,"60%",ok\n'
    df = read_table(data, "a.csv")
    assert df.loc[0, "remark"] == "late, twice"
    assert df.loc[1, "att"] == "60%"


def test_xlsx_roundtrip():
    src = pd.DataFrame(
        {"Roll No": [1, 2, 3], "Attendance": [90.5, 40, 75], "Result": ["Pass", "Fail", None]}
    )
    buf = io.BytesIO()
    src.to_excel(buf, index=False)
    df = read_table(buf.getvalue(), "sheet.xlsx")
    assert list(df.columns) == ["Roll No", "Attendance", "Result"]
    assert df["Result"].isna().sum() == 1
    assert float(df.loc[0, "Attendance"]) == 90.5


def test_zip_bomb_rejected():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("xl/worksheets/sheet1.xml", b"0" * 70_000_000)
    assert len(buf.getvalue()) < MAX_UPLOAD_BYTES
    with pytest.raises(UploadError) as e:
        read_table(buf.getvalue(), "bomb.xlsx")
    assert e.value.code == "xlsx_too_large"


def test_oversize_rejected():
    with pytest.raises(UploadError) as e:
        read_table(b"a,b\n" + b"1,2\n" * (MAX_UPLOAD_BYTES // 4 + 10), "big.csv")
    assert e.value.code == "too_large"


def test_empty_rejected():
    with pytest.raises(UploadError) as e:
        read_table(b"", "e.csv")
    assert e.value.code == "empty"
    with pytest.raises(UploadError) as e:
        read_table(b"a,b,c\n", "h.csv")
    assert e.value.code == "no_rows"


def test_binary_garbage_rejected():
    with pytest.raises(UploadError) as e:
        read_table(bytes(range(256)) * 10, "x.csv")
    assert e.value.code == "unreadable"


def test_legacy_xls_rejected_with_help():
    with pytest.raises(UploadError) as e:
        read_table(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 100, "old.xls")
    assert e.value.code == "legacy_excel"
    assert ".xlsx" in e.value.message


def test_duplicate_and_blank_headers_normalised():
    df = read_table(b"a,a,,b\n1,2,3,4\n", "d.csv")
    assert list(df.columns) == ["a", "a_2", "column_3", "b"]


def test_too_many_columns_rejected():
    header = ",".join(f"c{i}" for i in range(MAX_COLS + 1))
    with pytest.raises(UploadError) as e:
        read_table((header + "\n" + ",".join("1" * (MAX_COLS + 1)) + "\n").encode(), "w.csv")
    assert e.value.code == "too_many_columns"


def test_fully_empty_rows_dropped():
    df = read_table(b"a,b\n1,2\n,\n3,4\n", "r.csv")
    assert len(df) == 2
