"""Safe reading of uploaded CSV / Excel files.

Defences: size cap before parsing, content sniffing (the extension is not trusted), zip-bomb
guard for .xlsx, XML-attack protection via defusedxml (used by openpyxl when installed),
formulas are never evaluated (cached values only), and every cell is read as text first.
"""

from __future__ import annotations

import csv
import io
import re
import zipfile

import numpy as np
import pandas as pd

from riskml.config import MAX_COLS, MAX_ROWS, MAX_UPLOAD_BYTES, MAX_XLSX_UNCOMPRESSED

NA_VALUES = ["", "NA", "N/A", "n/a", "na", "-", "--", "?", "null", "NULL", "None", "none",
             "nan", "NaN", "#N/A"]
_XLSX_MAGIC = b"PK\x03\x04"
_XLS_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
_WS = re.compile(r"\s+")


class UploadError(ValueError):
    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(message)


def _normalise_headers(cols) -> list[str]:
    out: list[str] = []
    seen: dict[str, int] = {}
    for i, c in enumerate(cols, start=1):
        name = "" if c is None or (isinstance(c, float) and np.isnan(c)) else str(c)
        name = _WS.sub(" ", name.replace("﻿", "")).strip()
        if not name or name.lower().startswith("unnamed:"):
            name = f"column_{i}"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 1
        out.append(name)
    return out


def _decode(data: bytes) -> str:
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16")
    if b"\x00" in data[:4096]:
        raise UploadError("unreadable", "This does not look like a text CSV or an Excel file.")
    for enc in ("utf-8-sig", "cp1252"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1")


def _looks_binary(text: str) -> bool:
    sample = text[:4096]
    ctrl = sum(1 for ch in sample if ord(ch) < 32 and ch not in "\r\n\t")
    return ctrl > len(sample) * 0.01


def _read_csv(data: bytes) -> pd.DataFrame:
    text = _decode(data)
    if _looks_binary(text):
        raise UploadError("unreadable", "This does not look like a text CSV or an Excel file.")
    sample = text[:65536]
    try:
        delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        delimiter = ","
    try:
        raw = pd.read_csv(io.StringIO(text), sep=delimiter, dtype=str, na_values=NA_VALUES,
                          keep_default_na=False, skip_blank_lines=True, header=None)
        # Take the header row ourselves so duplicate names are not silently mangled.
        df = raw.iloc[1:].reset_index(drop=True)
        df.columns = list(raw.iloc[0])
        return df
    except pd.errors.EmptyDataError as exc:
        raise UploadError("empty", "The file is empty.") from exc
    except (pd.errors.ParserError, ValueError) as exc:
        raise UploadError("malformed", f"Could not parse the CSV: {exc}") from exc


def _cell(v):
    if v is None:
        return np.nan
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    if hasattr(v, "isoformat"):
        return v.isoformat()
    s = str(v).strip()
    return np.nan if s in NA_VALUES else s


def _read_xlsx(data: bytes) -> pd.DataFrame:
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise UploadError("unreadable", "The Excel file is damaged.") from exc
    infos = zf.infolist()
    if len(infos) > 2000 or sum(i.file_size for i in infos) > MAX_XLSX_UNCOMPRESSED:
        raise UploadError("xlsx_too_large",
                          "The Excel file expands to more than 60 MB. Save it as CSV instead.")
    from openpyxl import load_workbook  # noqa: PLC0415

    try:
        wb = load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    except Exception as exc:
        raise UploadError("unreadable", "Could not open the Excel workbook.") from exc
    try:
        ws = wb.worksheets[0]
        rows = []
        for r in ws.iter_rows(values_only=True):
            rows.append([_cell(v) for v in r[: MAX_COLS + 1]])
            if len(rows) > MAX_ROWS + 1:
                raise UploadError("too_many_rows", f"At most {MAX_ROWS:,} rows are supported.")
    finally:
        wb.close()
    rows = [r for r in rows if any(isinstance(v, str) for v in r)]
    if not rows:
        raise UploadError("empty", "The first sheet of the workbook is empty.")
    width = max(len(r) for r in rows)
    rows = [r + [np.nan] * (width - len(r)) for r in rows]
    return pd.DataFrame(rows[1:], columns=rows[0])


def read_table(data: bytes, filename: str) -> pd.DataFrame:
    if not data:
        raise UploadError("empty", "The file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise UploadError("too_large",
                          f"Files up to {MAX_UPLOAD_BYTES // 1_000_000} MB are supported.")
    if data.startswith(_XLS_MAGIC):
        raise UploadError("legacy_excel",
                          "Old .xls files are not supported. In Excel choose File > Save As > "
                          ".xlsx (or CSV) and upload that.")
    df = _read_xlsx(data) if data.startswith(_XLSX_MAGIC) else _read_csv(data)
    if df.shape[1] > MAX_COLS:
        raise UploadError("too_many_columns", f"At most {MAX_COLS} columns are supported.")
    df.columns = _normalise_headers(df.columns)
    df = df.dropna(how="all").reset_index(drop=True)
    empty_unnamed = [c for c in df.columns if c.startswith("column_") and df[c].isna().all()]
    df = df.drop(columns=empty_unnamed)
    if len(df) == 0:
        raise UploadError("no_rows", "The file has a header but no data rows.")
    if len(df) > MAX_ROWS:
        raise UploadError("too_many_rows", f"At most {MAX_ROWS:,} rows are supported.")
    for c in df.columns:
        df[c] = df[c].astype("object").where(df[c].notna(), np.nan)
        df[c] = df[c].map(lambda v: v.strip() if isinstance(v, str) else v)
    _ = filename  # extension is informational only; content decides the format
    return df
