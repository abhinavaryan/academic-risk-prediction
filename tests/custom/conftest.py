import pandas as pd
import pytest

from riskml.config import SAMPLE_DATA_DIR

UCI_PATH = SAMPLE_DATA_DIR / "uci_student_dropout.csv"


@pytest.fixture(scope="session")
def uci_bytes() -> bytes:
    return UCI_PATH.read_bytes()


@pytest.fixture(scope="session")
def uci_df(uci_bytes) -> pd.DataFrame:
    from riskml.custom.io import read_table

    return read_table(uci_bytes, "uci_student_dropout.csv")
