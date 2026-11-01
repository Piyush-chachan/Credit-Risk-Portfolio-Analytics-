import sys
from pathlib import Path
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import assets, config as C


@pytest.fixture(scope="session")
def built():
    assets.ensure_built()
    return True


@pytest.fixture(scope="session")
def portfolio(built):
    return assets.load_portfolio()


@pytest.fixture(scope="session")
def collections(built):
    c = pd.read_csv(C.PROC_DIR / C.ART["collections"])
    c["origination_year"] = c["origination_year"].astype(str).str.replace(r"\.0$", "", regex=True)
    return c
